## 使用DPDK的简单web server

一个极高性能的简单web server，使用DPDK框架，可以用于极高并发但处理逻辑比较简单的环境，比如高考查分/物联网应用。

由于不处理/保存客户端TCP连接的状态信息，因此本服务器支持的并发数是无限的，仅仅受限于带宽。

以高考查分为例，假定一个省有100万(1M)个考生，每个考生的信息有200字节，全部数据是200MB字节。
这些信息读入内存用hash查找，可以在O(1)时间查到数据(用数据库太慢了)。每个考生的查询过程，可以精简到大约
15个数据包交互，全部考生查1次成绩大约需要15M个数据包，本程序的目标是每秒钟处理超过10M个数据包，也就是
说如果带宽足够，可以在1.5秒钟内让一个省考生查询1次成绩。实际受限于带宽，不能这么快。这些数据包预计有
2GB左右，如果是1Gbps的网络，大约20秒可以完成。

本服务器只能处理极其简单的请求: 仅仅处理用户发来的第一个TCP包中的请求（最好使用GET请求），
HTTP应答也最好不超过TCP MSS长度（也许将来可以实现IP包分片功能）。

已实现功能：
* 响应ARP
* 响应ICMP echo
* 响应TCP SYN
* 响应HTTP GET

我的环境：(Ubuntu 17.10)

```
安装Ubuntu artful(17.10)

apt-get install gcc git make libnuma-dev

ln -s /usr/bin/python3 /usr/bin/python

cd /usr/src
wget https://fast.dpdk.org/rel/dpdk-17.11.tar.xz
xzcat dpdk-17.11.tar.xz | tar xvf -

#以下dpdk环境准备，选择一种即可

#2.1 dpdk环境准备，如果网卡是vmxnet3或intel常见网卡，可以用usertools/dpdk-setup.py
cd dpdk-17.11
usertools/dpdk-setup.py
select 14 编译
select 17 加载模块
select 21 输入64
select 22 查看网卡
select 23 输入空余的网卡名字，绑定网卡

#2.2 如果是不常用网卡，可能需要手工操作
cd dpdk-17.11
export RTE_SDK=$PWD
export RTE_TARGET=x86_64-native-linuxapp-gcc
make install T=${RTE_TARGET}
#必要时修改 x86_64-native-linuxapp-gcc/config 启用某些网卡驱动
modprobe uio
insmod ./x86_64-native-linuxapp-gcc/kmod/igb_uio.ko
mkdir /mnt/huge
mount -t hugetlbfs nodev /mnt/huge
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
./usertools/dpdk-devbind.py --bind igb_uio eth1


cd /usr/src/
git clone https://github.com/bg6cq/dpdk-simple-web.git
cd dpdk-simple-web
source env_vars

make

#测试运行, 其中192.168.1.2是网卡的IP地址
build/printreq -c1 -n1 -- 192.168.1.2 80

```

从其他机器访问 http://192.168.1.2 能看到显示HTTP请求的信息

下面是从其他机器进行的测试(受限于测试机性能，并不能测试出本程序的性能)：

```
ab -n 100000 -c 1000 http://222.195.81.233/
This is ApacheBench, Version 2.3 <$Revision: 1796539 $>
Copyright 1996 Adam Twiss, Zeus Technology Ltd, http://www.zeustech.net/
Licensed to The Apache Software Foundation, http://www.apache.org/

Benchmarking 222.195.81.233 (be patient)
Completed 10000 requests
Completed 20000 requests
Completed 30000 requests
Completed 40000 requests
Completed 50000 requests
Completed 60000 requests
Completed 70000 requests
Completed 80000 requests
Completed 90000 requests
Completed 100000 requests
Finished 100000 requests


Server Software:        dpdk-simple-web-server
Server Hostname:        222.195.81.233
Server Port:            80

Document Path:          /
Document Length:        123 bytes

Concurrency Level:      1000
Time taken for tests:   0.975 seconds
Complete requests:      100000
Failed requests:        0
Total transferred:      31900000 bytes
HTML transferred:       12300000 bytes
Requests per second:    102565.26 [#/sec] (mean)
Time per request:       9.750 [ms] (mean)
Time per request:       0.010 [ms] (mean, across all concurrent requests)
Transfer rate:          31951.48 [Kbytes/sec] received

Connection Times (ms)
              min  mean[+/-sd] median   max
Connect:        2    4   0.5      4       7
Processing:     3    5   0.8      5       9
Waiting:        2    4   0.7      4       8
Total:          7   10   0.7      9      13
WARNING: The median and mean for the total time are not within a normal deviation
        These results are probably not that reliable.

Percentage of the requests served within a certain time (ms)
  50%      9
  66%     10
  75%     10
  80%     10
  90%     11
  95%     11
  98%     11
  99%     12
 100%     13 (longest request)
```
