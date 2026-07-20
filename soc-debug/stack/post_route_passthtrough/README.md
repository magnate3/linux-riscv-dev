
# 固定 10.11.11.81 的mac

```
 char mac[ETH_ALEN] = {0x48,0x57,0x02,0x64,0xea,0x1b};
```

```
2: enahisic2i0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 48:57:02:64:ea:1b brd ff:ff:ff:ff:ff:ff
    inet 10.11.11.81/24 brd 10.11.11.255 scope global noprefixroute enahisic2i0
       valid_lft forever preferred_lft forever
    inet6 fe80::997e:ea4a:6f5d:f076/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
```

#   10.11.11.81  ping  10.11.11.251

***执行 ping***

```
rtt min/avg/max/mdev = 41.880/41.895/41.907/0.167 ms
[root@bogon ~]# ping 10.11.11.251
PING 10.11.11.251 (10.11.11.251) 56(84) bytes of data.
64 bytes from 10.11.11.251: icmp_seq=1 ttl=64 time=41.9 ms
64 bytes from 10.11.11.251: icmp_seq=2 ttl=64 time=41.8 ms
64 bytes from 10.11.11.251: icmp_seq=3 ttl=64 time=41.8 ms
64 bytes from 10.11.11.251: icmp_seq=4 ttl=64 time=41.8 ms
^C
--- 10.11.11.251 ping statistics ---
4 packets transmitted, 4 received, 0% packet loss, time 3005ms
rtt min/avg/max/mdev = 41.867/41.886/41.902/0.205 ms
[root@bogon ~]# 
```

***查看日志***

```
[1325421.763691] dst_ip: 10.11.11.81
[1325421.766993] POSTROUTING dev_queue_xmit returned 0
[1325422.737767] local out Got ICMP Reply packet and print it. 
[1325422.743402] src_ip: 10.11.11.251 
[1325422.746877] dst_ip: 10.11.11.81
[1325422.750179] ****************SOCK_RAW *************
[1325422.755120] postroute Got ICMP  packet and print it. 
[1325422.760321] src_ip: 10.11.11.251 
[1325422.763793] dst_ip: 10.11.11.81
[1325422.767095] POSTROUTING dev_queue_xmit returned 0
[root@centos7 post_route_passthtrough]# dmesg | tail -n 20
[1325430.759185] postroute Got ICMP  packet and print it. 
[1325430.764385] src_ip: 10.11.11.251 
[1325430.767859] dst_ip: 10.11.11.81
[1325430.771160] POSTROUTING dev_queue_xmit returned 0
[1325431.742746] local out Got ICMP Reply packet and print it. 
[1325431.748383] src_ip: 10.11.11.251 
[1325431.751855] dst_ip: 10.11.11.81
[1325431.755156] ****************SOCK_RAW *************
[1325431.760100] postroute Got ICMP  packet and print it. 
[1325431.765300] src_ip: 10.11.11.251 
[1325431.768773] dst_ip: 10.11.11.81
[1325431.772073] POSTROUTING dev_queue_xmit returned 0
[1325432.742849] local out Got ICMP Reply packet and print it. 
[1325432.748486] src_ip: 10.11.11.251 
[1325432.751957] dst_ip: 10.11.11.81
[1325432.755258] ****************SOCK_RAW *************
[1325432.760203] postroute Got ICMP  packet and print it. 
[1325432.765402] src_ip: 10.11.11.251 
[1325432.768875] dst_ip: 10.11.11.81
[1325432.772176] POSTROUTING dev_queue_xmit returned 0
```

# 跨网段ping

```
[1507020.910203] local in Got ICMP Request packet and print it.   --- local in 有ip头
[1507020.915924] src_ip: 192.168.116.19 
[1507020.919572] dst_ip: 10.10.16.251
[1507020.922966] local out Got ICMP Reply packet and print it.   --- local out 有ip头
[1507020.928601] src_ip: 10.10.16.251 
[1507020.932072] dst_ip: 192.168.116.19
[1507020.935631] ****************SOCK_RAW *************
[1507020.940574] postroute Got ICMP  packet and print it.    --- postroute 有ip头
[1507020.945774] src_ip: 10.10.16.251 
[1507020.949246] dst_ip: 192.168.116.19
[1507020.952807] POSTROUTING dev_queue_xmit returned 0
[1507021.796217] local out Got ICMP Reply packet and print it. 
[1507021.801853] src_ip: 0.0.0.0 
[1507021.804894] dst_ip: 255.255.255.255
```

# references

[linuxIpIpTunnel](https://github.com/tanmayM/linuxIpIpTunnel/tree/f314c8512b988b160cf9d429013d344848e8e63a)  

[hwaddr-cache](https://github.com/OSLL/hwaddr-cache)