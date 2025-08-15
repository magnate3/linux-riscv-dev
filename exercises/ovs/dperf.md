
# make

```
dperf]# make install RTE_TARGET=arm64-armv8a-linuxapp-gcc  RTE_SDK=/root/dpdk-19.11/ -j 64
```
# server

```
[root@centos7 dperf]# cat test/http/server-cps.conf
mode        server
cpu         0
duration    10m
port         0000:05:00.0   10.10.103.251  10.10.103.81 
#client      6.6.241.3       254
#client      6.6.242.2       254
client      10.10.103.82      1
client      10.10.103.81      1
server      10.10.103.251      1
listen      80              1
[root@centos7 dperf]# 
```
0000:05:00.0: pci地址

```
./build/dperf -c test/http/server-cps.conf
```

```
[root@centos7 dpdk-19.11]# ./usertools/dpdk-devbind.py  -s

Network devices using DPDK-compatible driver
============================================
0000:05:00.0 'Hi1822 Family (2*100GE) 0200' drv=vfio-pci unused=hinic

```

# client1
```
[root@bogon ~]# ping  10.10.103.251
PING 10.10.103.251 (10.10.103.251) 56(84) bytes of data.
64 bytes from 10.10.103.251: icmp_seq=1 ttl=64 time=0.147 ms
64 bytes from 10.10.103.251: icmp_seq=2 ttl=64 time=0.147 ms
64 bytes from 10.10.103.251: icmp_seq=3 ttl=64 time=0.152 ms
^C
--- 10.10.103.251 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2049ms
rtt min/avg/max/mdev = 0.147/0.148/0.152/0.014 ms
[root@bogon ~]# curl http://10.10.103.251/
hello dperf!
[root@bogon ~]# curl http://10.10.103.251/
hello dperf!
[root@bogon ~]# 
```

```
5: enahisic2i3: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 48:57:02:64:ea:1e brd ff:ff:ff:ff:ff:ff
    inet 192.168.11.81/24 scope global enahisic2i3
       valid_lft forever preferred_lft forever
    inet 192.168.10.81/24 scope global enahisic2i3
       valid_lft forever preferred_lft forever
    inet 192.168.1.81/24 scope global enahisic2i3
       valid_lft forever preferred_lft forever
    inet 10.10.103.81/24 scope global enahisic2i3
       valid_lft forever preferred_lft forever
```

# client2

```
5: enahisic2i3: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 48:57:02:64:e7:ae brd ff:ff:ff:ff:ff:ff
    inet 10.10.103.82/24 scope global enahisic2i3
       valid_lft forever preferred_lft forever
    inet6 fe80::4a57:2ff:fe64:e7ae/64 scope link 
```

```
root@ubuntu:~# curl http://10.10.103.251/
hello dperf!
root@ubuntu:~# curl http://10.10.103.251/
hello dperf!
root@ubuntu:~# curl http://10.10.103.251/
hello dperf!
```