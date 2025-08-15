

对于rdma编程，目前主流实现是利用rdma_cm来建立连接，然后利用verbs来传输数据。   
rdma_cm和ibverbs分别会创建一个fd，这两个fd的分工不同。rdma_cm fd主要用于通知建连相关的事件，verbs fd则主要通知有新的cqe发生。   
当直接对rdma_cm fd进行poll/epoll监听时，此时只能监听到POLLIN事件，这意味着有rdma_cm事件发生。   
当直接对verbs fd进行poll/epoll监听时，同样只能监听到POLLIN事件，这意味着有新的cqe。   
 

RDMA技术的实现需要使用两种不同的接口：RDMA Connection Manager（rdmacm）接口和verbs接口。
RDMA  ConnectionManager（rdmacm）接口主要用于管理RDMA通信的连接过程，这个接口提供了连接、断开连接、接受连接请求、错误处理、打印调试信息等功能。rdmacm接口使用TCP/IP协议进行连接管理，它提供了一种简单的方式来管理RDMA传输和通信。   

verbs接口则是对RDMA传输和通信进行操作的主要API，它提供了访问设备、内存和网络协议的抽象层。verbs接口是一种底层的编程接口，通过verbs接口可以控制RDMA网络适配器中的硬件资源、创建和管理RDMA操作队列等功能，同时verbs接口也提供了一些原子操作、随机访问、远程直接内存访问等工具，从而实现了高效的、无锁的、直接内存访问。   

总之，rdmacm接口和verbs接口都是RDMA技术的实现接口，其中rdmacm接口主要用于连接管理，并提供了简单易用的接口，verbs接口则是整个RDMA传输和通信的主要API，提供了底层、高效的编程接口。   


# 1、IB_VERBS

接口以ibv_xx（用户态）或者ib_xx（内核态）作为前缀，是最基础的编程接口，使用IB_VERBS就足够编写RDMA应用了。   

比如：  
ibv_create_qp() 用于创建QP    
ibv_post_send() 用于下发Send WR  
ibv_poll_cq() 用于从CQ中轮询CQE   
#  2、RDMA_CM

以rdma_为前缀，主要分为两个功能：

1）CMA（Connection Management Abstraction）---- 建连连接管理

在Socket和Verbs API基础上实现的，用于CM建链并交换信息的一组接口。CM建链是在Socket基础上封装为QP实现，从用户的角度来看，是在通过QP交换之后数据交换所需要的QPN，Key等信息。

比如：

rdma_listen()用于监听链路上的CM建链请求。
rdma_connect()用于确认CM连接。

2）CM VERBS                                                        ----收发数据   

RDMA_CM也可以用于数据交换 (收发数据），相当于在verbs API上又封装了一套数据交换接口。   
比如：   
rdma_post_read()可以直接下发RDMA READ操作的WR，而不像ibv_post_send()，需要在参数中指定操作类型为READ。
rdma_post_ud_send()可以直接传入远端QPN，指向远端的AH，本地缓冲区指针等信息触发一次UD SEND操作。  
上述接口虽然方便，但是需要配合CMA管理的链路使用，不能配合Verbs API使用。  
Verbs API除了IB_VERBS和RDMA_CM之外，还有MAD（Management Datagram）接口等。   


# rdma_cm

RDMA的队列分为几种：发送队列Send Queue (SQ)，接收队列Receive Queue(RQ)，和完成队列Completion Queue (CQ)。其中SQ和RQ统称工作队列Work Queue (WQ)，也称为Queue Pair (QP)。此外，RDMA提供了两个接口，ibv_post_send和ibv_post_recv，由用户程序调用，分别用于发送消息和接收消息：   

1） 用户程序调用ibv_post_send把发送请求Send Request (SR)插入SQ，成为发送队列的一个新的元素Send Queue Element (SQE)；
用户程序调用ibv_post_recv把接收请求Receive Request (RR)插入RQ，成为接收队列的一个新元素Receive Queue Element (RQE)。
SQE和RQE也统称工作队列元素Work Queue Element (WQE)。   

当SQ里有消息发送完成，或RQ有接收到新消息，RDMA会在CQ里放入一个新的完成队列元素Completion Queue Element (CQE)，用以通知用户程序。
***用户程序有两种同步的方式来查询CQ***：   

2） 用户程序调用ibv_cq_poll来轮询CQ，一旦有新的CQE就可以及时得到通知，但是这种轮询方式很消耗CPU资源；
用户程序在创建CQ的时候，指定一个完成事件通道ibv_comp_channel，然后调用ibv_get_cq_event接口等待该完成事件通道来通知有新的CQE，如果没有新的CQE，则调用ibv_get_cq_event时发生阻塞，这种方法比轮询要节省CPU资源，但是阻塞会降低程序性能。    
关于RDMA的CQE，有个需要注意的地方：对于RDMA的Send和Receive这种双边操作，发送端在发送完成后能收到CQE，接收端在接收完成后也能收到CQE；对于RDMA的Read和Write这种单边操作，比如节点A从节点B读数据，或节点A往节点B写数据，只有发起Read和Write操作的一端，即节点A在操作结束后能收到CQE，另一端节点B完全不会感知到节点A发起的Read或Write操作，节点B也不会收到CQE。  


##  RDMA完成队列CQ读取CQE的同步和异步方法

一） RDMA轮询方式读取CQE    
RDMA轮询方式读取CQ非常简单，就是不停调用ibv_poll_cq来读取CQ里的CQE。这种方式能够最快获得新的CQE，直接用户程序轮询CQ，而且也不需要内核参与，但是缺点也很明显，用户程序轮询消耗大量CPU资源。   


```
loop {
    // 尝试读取一个CQE
    poll_result = ibv_poll_cq(cq, 1, &mut cqe);
    if poll_result != 0 {
        // 处理CQE
    }
}
```
二) RDMA完成事件通道方式读取CQE（ibv_get_cq_event阻塞cpu）   

RDMA用完成事件通道读取CQE的方式如下：
+ 1 用户程序通过调用ibv_create_comp_channel创建完成事件通道；
+ 2 接着在调用ibv_create_cq创建CQ时关联该完成事件通道；
+ 3 再通过调用ibv_req_notify_cq来告诉CQ当有新的CQE产生时从完成事件通道来通知用户程序；
+ 4 然后通过调用ibv_get_cq_event查询该完成事件通道，没有新的CQE时阻塞，有新的CQE时返回；
+ 5
接下来用户程序从ibv_get_cq_event返回之后，还要再调用ibv_poll_cq从CQ里读取新的CQE，此时调用ibv_poll_cq一次就好，不需要轮询。
RDMA用完成事件通道读取CQE的代码示例如下：

```
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

用RDMA完成事件通道的方式来读取CQE，本质是RDMA通过内核来通知用户程序CQ里有新的CQE。事件队列是通过一个设备文件，/dev/infiniband/uverbs0（如果有多个RDMA网卡，则每个网卡对应一个设备文件，序号从0开始递增），来让内核通过该设备文件通知用户程序有事件发生。用户程序调用ibv_create_comp_channel创建完成事件通道，其实就是打开上述设备文件；用户程序调用ibv_get_cq_event查询该完成事件通道，其实就是读取打开的设备文件。但是这个设备文件只用于做事件通知，通知用户程序有新的CQE可读，但并不能通过该设备文件读取CQE，用户程序还要是调用ibv_poll_cq来从CQ读取CQE。

用完成事件通道的方式来读取CQE，比轮询的方法要节省CPU资源，但是调用***ibv_get_cq_event***读取完成事件通道会发生阻塞，进而影响用户程序性能。

（三） 基于epoll异步读取CQE    

上面提到用RDMA完成事件通道的方式来读取CQE，本质是用户程序通过事件队列打开/dev/infiniband/uverbs0设备文件，并读取内核产生的关于新CQE的事件通知。从完成事件通道ibv_comp_channel的定义可以看出，里面包含了一个Linux文件描述符，指向打开的设备文件。
于是可以借助epoll机制来检查该设备文件是否有新的事件产生，避免用户程序调用ibv_get_cq_event读取完成事件通道时（即读取该设备文件时）发生阻塞。
+ 3.1 首先，用fcntl来修改完成事件通道里设备文件描述符的IO方式为非阻塞   

 

+ 3.2 接着，创建epoll实例，并把要检查的事件注册给epoll实例：   

+ 3.3 循环调用epoll_wait检查设备文件是否有新数据可读，有新数据可读说明有新的CQE产生，再调用ibv_poll_cq来读取CQE：
 ```
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
 上面代码有个要注意的地方，因为epoll是用边沿触发，所以每次有新CQE产生时，都要调用ibv_poll_cq把CQ队列读空。考虑如下场景，同时有多个新的CQE产生，但是epoll边沿触发只通知一次，如果用户程序收到通知后没有读空CQ，那epoll也不会再产生新的通知，除非再有新的CQE产生，epoll才会再次通知用户程序。  
 
 
 #  QP 状态 RESET->INIT->RTR->RTS（ibv_modify_qp）
 
+ 1  状态：RESET -> INIT -> RTR -> RTS  
+ 2  要严格按照顺序进行转换   
+ 3  QP 刚创建时状态为 RESET  
+ 4  INIT 之后就可以调用 ibv_post_recv 提交一个 receive buffer 了   
+ 5  当 QP 进入 RTR(ready to receive)状态以后，便开始进行接收处理  
+ 6  RTR 之后便可以转为 RTS(ready to send)，RTS 状态下可以调用 ibv_post_send  