

# config


+ client  

```
static uint32_t client_ip_addr = RTE_IPV4(10,10,103,81);
struct rte_ether_addr cli_mac=
    {{0x48, 0x57, 0x02, 0x64, 0xea, 0x1e}};
```

```
ip a add 2008:db8:0:0::10.10.103.81/96 dev enahisic2i3
[root@bogon ~]# iperf -V -c 2008:db8:0:0::10.10.103.82 -p 8000
------------------------------------------------------------
Client connecting to 2008:db8:0:0::10.10.103.82, TCP port 8000
TCP window size: 1.84 MByte (default)
------------------------------------------------------------
[  3] local 2008:db8::a0a:6751 port 40584 connected with 2008:db8::a0a:6752 port 8000
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0-10.0 sec  7.17 GBytes  6.15 Gbits/sec
[root@bogon ~]# 
```
nat46测试    
```
[root@bogon ~]# iperf -c 10.10.103.251 -p 8800
------------------------------------------------------------
Client connecting to 10.10.103.251, TCP port 8800
TCP window size: 45.0 KByte (default)
------------------------------------------------------------
[  3] local 10.10.103.81 port 47192 connected with 10.10.103.251 port 8800
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0-10.1 sec  66.9 KBytes  54.5 Kbits/sec
[root@bogon ~]# 
```

+ dpdk

const char *ip6str = "2008:db8::a0a:6751";   
10.10.103.251


+ gw 



const char *gwip6str= "2008:db8::a0a:6752";   

```
ip a add 2008:db8:0:0::10.10.103.82/96 dev enahisic2i3
root@ubuntu:~# iperf -V -s  -p 8000
------------------------------------------------------------
Server listening on TCP port 8000
TCP window size:  128 KByte (default)
------------------------------------------------------------
[  4] local ::ffff:10.10.103.82 port 8000 connected with ::ffff:10.10.103.81 port 48114
[ ID] Interval       Transfer     Bandwidth
[  4]  0.0- 9.5 sec  5.28 GBytes  4.78 Gbits/sec
[  4] local 2008:db8::a0a:6752 port 8000 connected with 2008:db8::a0a:6751 port 40584

```