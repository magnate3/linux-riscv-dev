#  socket 低延迟选项：busy polling

socket 有个 `SO_BUSY_POLL` 选项，可以让内核在**阻塞式接收**（blocking receive）
时做 busy poll。这个选项会减少延迟，但会增加 CPU 使用量和耗电量。

**重要提示**：要使用此功能，首先要检查设备驱动是否支持。
如果驱动实现（并注册）了 `struct net_device_ops` 的 **<mark><code>ndo_busy_poll()</code></mark>** 方法，那就是支持。

Intel 有一篇非常好的[文章](http://www.intel.com/content/dam/www/public/us/en/documents/white-papers/open-source-kernel-enhancements-paper.pdf)介绍其原理。

对单个 socket 设置此选项，需要传一个以微秒（microsecond）为单位的时间，内核会
在这个时间内对设备驱动的接收队列做 busy poll。当在这个 socket 上触发一个阻塞式读请
求时，内核会 busy poll 来收数据。

全局设置此选项，可以修改 sysctl 配置项：

```shell
$ sudo sysctl -a | grep busy_poll
net.core.busy_poll = 0
```

注意这并不是一个 flag 参数，而是一个毫秒（microsecond）为单位的时间长度，当 `poll` 或 `select` 方
法以阻塞方式调用时，busy poll 的时长就是这个值。

<a name="chap_12.3"></a>


## ibv_poll_cq
oll策略：poll策略指的是poll CQE的方式，包括busy polling、event-triggered polling，busy polling以高CPU使用代价换取更快的CQE poll速度，event-triggered polling则相应的poll速度慢，但CPU使用代价低；       
We use ibv_poll_cq to poll a completion queue. It is a busy poll, so it consumes a CPU core, but provides lower latency.    
```
bool pollCompletion(struct ibv_cq* cq) {
  struct ibv_wc wc;
  int result;

  do {
    // ibv_poll_cq returns the number of WCs that are newly completed,
    // If it is 0, it means no new work completion is received.
    // Here, the second argument specifies how many WCs the poll should check,
    // however, giving more than 1 incurs stack smashing detection with g++8 compilation.
    result = ibv_poll_cq(cq, 1, &wc);
  } while (result == 0);

  if (result > 0 && wc.status == ibv_wc_status::IBV_WC_SUCCESS) {
    // success
    return true;
  }

  // You can identify which WR failed with wc.wr_id.
  printf("Poll failed with status %s (work request ID: %llu)\n", ibv_wc_status_str(wc.status), wc.wr_id);
  return false;
}

```
ibv_poll_cq returns the number of WCs. As we specify it has to wait at most one WC, the result must be either 0, 1, or negative if error occured.  

## RDMA完成队列CQ读取CQE的同步和异步方法
用RDMA读取CQ的操作为例展示如何基于`epoll`实现异步操作。先介绍下RDMA用轮询和阻塞的方式读取CQ，再介绍基于`epoll`的异步读取CQ的方法。下文的代码仅作为示例，并不能编译通过。

### RDMA轮询方式读取CQE

RDMA轮询方式读取CQ非常简单，就是不停调用`ibv_poll_cq`来读取CQ里的CQE。这种方式能够最快获得新的CQE，直接用户程序轮询CQ，而且也不需要内核参与，但是缺点也很明显，用户程序轮询消耗大量CPU资源。
```Rust
loop {
    // 尝试读取一个CQE
    poll_result = ibv_poll_cq(cq, 1, &mut cqe);
    if poll_result != 0 {
        // 处理CQE
    }
}
```

### RDMA完成事件通道方式读取CQE

RDMA用完成事件通道读取CQE的方式如下：
* 用户程序通过调用`ibv_create_comp_channel`创建完成事件通道；
* 接着在调用`ibv_create_cq`创建CQ时关联该完成事件通道；
* 再通过调用`ibv_req_notify_cq`来告诉CQ当有新的CQE产生时从完成事件通道来通知用户程序；
* 然后通过调用`ibv_get_cq_event`查询该完成事件通道，没有新的CQE时阻塞，有新的CQE时返回；
* 接下来用户程序从`ibv_get_cq_event`返回之后，还要再调用`ibv_poll_cq`从CQ里读取新的CQE，此时调用`ibv_poll_cq`一次就好，不需要轮询。

RDMA用完成事件通道读取CQE的代码示例如下：
```Rust
// 创建完成事件通道
let completion_event_channel = ibv_create_comp_channel(...);
// 创建完成队列，并关联完成事件通道
let cq = ibv_create_cq(completion_event_channel, ...);

loop {
    // 设置CQ从完成事件通道来通知下一个新CQE的产生
    ibv_req_notify_cq(cq, ...);
    // 通过完成事件通道查询CQ，有新的CQE就返回，没有新的CQE则阻塞
    ibv_get_cq_event(completion_event_channel, &mut cq, ...);
    // 读取一个CQE
    poll_result = ibv_poll_cq(cq, 1, &mut cqe);
    if poll_result != 0 {
        // 处理CQE
    }
    // 确认一个CQE
    ibv_ack_cq_events(cq, 1);
}
```

用RDMA完成事件通道的方式来读取CQE，本质是RDMA通过内核来通知用户程序CQ里有新的CQE。事件队列是通过一个设备文件，`/dev/infiniband/uverbs0`（如果有多个RDMA网卡，则每个网卡对应一个设备文件，序号从0开始递增），来让内核通过该设备文件通知用户程序有事件发生。用户程序调用`ibv_create_comp_channel`创建完成事件通道，其实就是打开上述设备文件；用户程序调用`ibv_get_cq_event`查询该完成事件通道，其实就是读取打开的设备文件。但是这个设备文件只用于做事件通知，通知用户程序有新的CQE可读，但并不能通过该设备文件读取CQE，用户程序还要是调用`ibv_poll_cq`来从CQ读取CQE。

用完成事件通道的方式来读取CQE，比轮询的方法要节省CPU资源，但是调用`ibv_get_cq_event`读取完成事件通道会发生阻塞，进而影响用户程序性能。

### 基于epoll异步读取CQE

上面提到用RDMA完成事件通道的方式来读取CQE，本质是用户程序通过事件队列打开`/dev/infiniband/uverbs0`设备文件，并读取内核产生的关于新CQE的事件通知。从完成事件通道`ibv_comp_channel`的定义可以看出，里面包含了一个Linux文件描述符，指向打开的设备文件：
```Rust
pub struct ibv_comp_channel {
    ...
    pub fd: RawFd,
    ...
}
```
于是可以借助`epoll`机制来检查该设备文件是否有新的事件产生，避免用户程序调用`ibv_get_cq_event`读取完成事件通道时（即读取该设备文件时）发生阻塞。

首先，用`fcntl`来修改完成事件通道里设备文件描述符的IO方式为非阻塞：
```Rust
// 创建完成事件通道
let completion_event_channel = ibv_create_comp_channel(...);
// 创建完成队列，并关联完成事件通道
let cq = ibv_create_cq(completion_event_channel, ...);
// 获取设备文件描述符当前打开方式
let flags =
    libc::fcntl((*completion_event_channel).fd, libc::F_GETFL);
// 为设备文件描述符增加非阻塞IO方式
libc::fcntl(
    (*completion_event_channel).fd,
    libc::F_SETFL,
    flags | libc::O_NONBLOCK
);
```

接着，创建`epoll`实例，并把要检查的事件注册给`epoll`实例：
```Rust
use nix::sys::epoll;

// 创建epoll实例
let epoll_fd = epoll::epoll_create()?;
// 完成事件通道里的设备文件描述符
let channel_dev_fd = (*completion_event_channel).fd;
// 创建epoll事件实例，并关联设备文件描述符，
// 当该设备文件有新数据可读时，用边沿触发的方式通知用户程序
let mut epoll_ev = epoll::EpollEvent::new(
    epoll::EpollFlags::EPOLLIN | epoll::EpollFlags::EPOLLET,
    channel_dev_fd
);
// 把创建好的epoll事件实例，注册到之前创建的epoll实例
epoll::epoll_ctl(
    epoll_fd,
    epoll::EpollOp::EpollCtlAdd,
    channel_dev_fd,
    &mut epoll_ev,
)
```
上面代码有两个注意的地方：
* `EPOLLIN`指的是要检查设备文件是否有新数据/事件可读；
* `EPOLLET`指的是epoll用边沿触发的方式来通知。

然后，循环调用`epoll_wait`检查设备文件是否有新数据可读，有新数据可读说明有新的CQE产生，再调用`ibv_poll_cq`来读取CQE：
```Rust
let timeout_ms = 10;
// 创建用于epoll_wait检查的事件列表
let mut event_list = [epoll_ev];
loop {
    // 设置CQ从完成事件通道来通知下一个新CQE的产生
    ibv_req_notify_cq(cq, ...);
    // 调用epoll_wait检查是否有期望的事件发生
    let nfds = epoll::epoll_wait(epoll_fd, &mut event_list, timeout_ms)?;
    // 有期望的事件发生
    if nfds > 0 {
        // 通过完成事件通道查询CQ，有新的CQE就返回，没有新的CQE则阻塞
        ibv_get_cq_event(completion_event_channel, &mut cq, ...);
        // 循环读取CQE，直到CQ读空
        loop {
            // 读取一个CQE
            poll_result = ibv_poll_cq(cq, 1, &mut cqe);
            if poll_result != 0 {
                // 处理CQE
                ...
                // 确认一个CQE
                ibv_ack_cq_events(cq, 1);
            } else {
                break;
            }
        }
    }
}
```
上面代码有个要注意的地方，因为`epoll`是用边沿触发，所以每次有新CQE产生时，都要调用`ibv_poll_cq`把CQ队列读空。考虑如下场景，同时有多个新的CQE产生，但是`epoll`边沿触发只通知一次，如果用户程序收到通知后没有读空CQ，那`epoll`也不会再产生新的通知，除非再有新的CQE产生，`epoll`才会再次通知用户程序。

总之，本文用`epoll`机制实现RDMA异步读取CQE的例子，展示了如何实现RDMA的异步操作。RDMA还有类似的操作，都可以基于`epoll`机制实现异步操作。
## epoll and ibv_poll_cq


```
uint32_t handle_completion_event(int32_t fd, uint32_t events, void *ctx)
{
	struct connection_struct *connection = ctx;
	struct ibv_cq *comp_queue = NULL;
	struct ibv_wc work_completion;
	void *ev_ctx = NULL;
	int res = 0;

	fprintf(stderr, "%s called\n", __func__);

	/*
	 * Retrieve the completion event and deal with it. We did not pass 
	 * a context when we created the completion queue, but we need a place
	 * for get_cq_events to place that NULL pointer anyway.
	 */
	if (ibv_get_cq_event(connection->comp_chan, &comp_queue, &ev_ctx)) {
		fprintf(stderr, "Could not get completion queue events: %s",
			strerror(errno));
		return 1;
	}

	ibv_ack_cq_events(comp_queue, 1);  /* Ack it */

	/* Re-arm ... we need the subsequent events, if any */
	if (ibv_req_notify_cq(comp_queue, 0)) {
		fprintf(stderr, "Could not request notifies on "
			"comp queue: %s\n",
			strerror(errno));
		return 1;
	}

	while ((res = ibv_poll_cq(comp_queue, 1, &work_completion)) > 0) {
		handle_completion(connection, &work_completion);
	}

	/* Did an error occur? */
	if (res < 0) {

	}

	return 0;
}
``` 

## X-RDMA
X-RDMA混合使用epoll和busy polling 来平衡CPU占用和响应速度，当有消息到来或者timer超时是切换到busy polling模式，当长时间没有事件时则切换到epoll模式   

## Introduction
![](assets/showcase.png)

We present RDMAbox, a set of low level RDMA optimizations that provide better performance than previous approaches. The optimizations are packaged in easy-to-use kernel and user space libraries for applications and systems in data centers. We demonstrate the flexibility and effectiveness of RDMAbox by implementing a kernel remote paging system and a user space file system using RDMAbox. RDMAbox employs two optimization techniques. First, we suggest RDMA request merging and chaining to reduce the total number of I/O operations to the RDMA NIC. The I/O merge queue at the same time functions as a traffic regulator to enforce admission control and avoid overloading the NIC. Second, we propose Adaptive Polling to achieve higher efficiency of polling Work Completion than existing busy polling while maintaining the low CPU overhead of event trigger. Our implementation of a remote paging system with RDMAbox outperforms existing representative solutions with up to 4x throughput improvement and up to 83% decrease in average tail latency in bigdata workloads, and up to 83% reduction in completion time in machine learning workloads. Our implementation of a user space file system based on RDMAbox achieves up to 5.9x higher throughput over existing representative solutions.

This repository contains the source code of the following paper:
Juhyun Bae, Ling Liu, Yanzhao Wu, Gong Su, Arun Iyengar "RDMAbox : Optimizing RDMA for Memory Intensive Workloads" In IEEE CIC, 2021. [[PDF]](https://arxiv.org/abs/2104.12197)

This repository is for Mellanox ConnectX-3 driver with kernel 3.13 for fair comparison with previous approach in evaluation.

RDMAbox for Inbox Kernel 4.x version is coming soon.

## Installation and Dependencies for Experiments in Paper (Kernel space Remote Paging System example)

1. Install MLNX_OFED driver
```bash
wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-3.1-1.0.3/MLNX_OFED_LINUX-3.1-1.0.3-ubuntu14.04-x86_64.tgz .
tar -xvf MLNX_OFED_LINUX-3.1-1.0.3-ubuntu14.04-x86_64.tgz
cd ~/MLNX_OFED_LINUX-3.1-1.0.3-ubuntu14.04-x86_64
sudo ./mlnxofedinstall --add-kernel-support --without-fw-update --force
```

2. Install

(for both client and server node)
```bash
cd ~/RDMAbox/kernel/setup
./compile.sh
```

3. Run remote paging example

Assume that IP address is 100.10.10.0(client)/100.10.10.1(peer1)/100.10.10.2(peer2)/100.10.10.3(peer3) and disk partion is /dev/sda3

(Modify portal list for Daemon nodes: repeat below for node 2 and 3 with their own address information)

vi ~/RDMAbox/kernel/setup/daemon_portal.list
```bash
3 -> number of daemon nodes
100.10.10.1:9999 -> IPaddr and Port
100.10.10.2:9999
100.10.10.3:9999
```

(Run Daemon nodes at each node with their own address)

```bash
cd ~/RDMAbox/kernel/setup/
~/ib_setup.sh 100.10.10.1

cd ~/RDMAbox/kernel/daemon
./daemon -a 100.10.10.1 -p 9999 -i "../setup/portal.list"
```

(Modify portal list for client node)

vi ~/RDMAbox/kernel/setup/bd_portal.list
```bash
100.10.10.0 -> my(i.e. client) address
3 -> number of daemon nodes
100.10.10.1:9999 -> IPaddr and Port
100.10.10.2:9999
100.10.10.3:9999
```

(Run Client node)
```bash
cd ~/RDMAbox/kernel/setup/
~/ib_setup.sh 100.10.10.0
sudo swapoff /dev/sda3
sudo ~/bd_setup.sh 0
```

(Check)

sudo swapon -s (on client node)

## Setting parameters

RDMAbox/kernel/bd/rdmabox.h

- SERVER_SELECT_NUM [number]

  This number should be equal to or less than the number of peer nodes.(e.g. if 3 peers, this should be <=3)

- NUM_REPLICA [number]

  Number of replicated copies on peer nodes.
```bash
#define SERVER_SELECT_NUM 2
#define NUM_REPLICA 1
```

RDMAbox/kernel/bd/rpg_drv.h

- SWAPSPACE_SIZE_G [number]

  Set total size of swapspace for remote paging system

- BACKUP_DISK [string]

  Set disk partition path is diskbackup is used. (By default, remote paging system does not use diskbackup. Use replication)

```bash
#define SWAPSPACE_SIZE_G        40
#define BACKUP_DISK     "/dev/sda4"
```

## Supported Platforms for paging system example

Tested environment:

OS : Ubuntu 14.04(kernel 3.13.0)

RDMA NIC driver: MLNX_OFED 3.1 and Inbox driver

Hardware : Infiniband, Mellanox ConnectX-3/4, disk partition(Only if diskbackup option is enabled.)


## Installation and Dependencies for Experiments in Paper (Userspace Network File System example)

We use "inbox RDMA driver" for userspace RDMA.

1. Fuse driver install

Use install scripts under RDMAbox/userspace/libfuse
```bash
For fuse permission, 
sudo vi /etc/fuse.conf
add below in the fuse.conf
user_allow_other
```

2. RDMA inbox driver setting
```bash
sudo vi /etc/security/limits.conf 
add below in limits.conf for both client and server(daemon) side
Note that empty space must be a tab(not space)
*	hard	memlock		unlimited
*	soft	memlock		unlimited
```

Reboot the system. In next booting, check with ulimit -a
```bash
max locked memory       (kbytes, -l) unlimited
max memory size         (kbytes, -m) unlimited
```

3. Run user space network file system example

#build rdmabox userspace modules
```bash
RDMAbox/userspace/setup/compile.sh
```

#edit daemonportal
```bash
vi RDMAbox/userspace/rdmabox/daemon/daemonportal
Add daemon address. First line is the number of total daemon servers.
1
100.100.100.91:9999
```

#run daemon
```bash
RDMAbox/userspace/rdmabox/daemon/daemon -a 100.100.100.92 -p 9999 -i "rdmabox/userspace/rdmabox/daemon/daemonportal"
```

#run rdmabox client
```bash
RDMAbox/userspace/ramdisk/ramdisk/run.sh 100.100.100.91
```

## RDMAbox API part
Coming soon...
![](asset/arch.png)

## Status
The code is provided as is, without warranty or support. If you use our code, please cite:
```
@inproceedings{bae2021rdmabox,
  title={RDMAbox : Optimizing RDMA for Memory Intensive Workloads},
  author={Bae, Juhyun and Liu, Ling and Wu, Yanzhao and Su, Gong and Iyengar, Arun},
  booktitle={IEEE International Conference on Collaboration and Internet Computing},
  year={2021},
  organization={IEEE}
}
```
