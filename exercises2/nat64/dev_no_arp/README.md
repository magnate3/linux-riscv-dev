


```
11: virt_net: <NOARP,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UNKNOWN group default qlen 1000
    link/ether 88:88:88:88:88:88 brd ff:ff:ff:ff:ff:ff
    inet6 fe80::6b52:aed:4fa5:bc8d/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
```

```
[root@centos7 dev_no_arp]# ip l set virt_net up
[root@centos7 dev_no_arp]# ip a  add  10.10.107.251/24 dev virt_net
[root@centos7 dev_no_arp]# ip a sh virt_net
11: virt_net: <NOARP,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UNKNOWN group default qlen 1000
    link/ether 88:88:88:88:88:88 brd ff:ff:ff:ff:ff:ff
    inet 10.10.107.251/24 scope global virt_net
       valid_lft forever preferred_lft forever
[root@centos7 dev_no_arp]# 
[root@centos7 dev_no_arp]# ping 10.10.107.252
PING 10.10.107.252 (10.10.107.252) 56(84) bytes of data.
64 bytes from 10.10.107.252: icmp_seq=1 ttl=64 time=0.024 ms
64 bytes from 10.10.107.252: icmp_seq=2 ttl=64 time=0.011 ms
^C
```

目的mac 和源mac一样，没有arp请求    
```
[root@centos7 ~]# tcpdump -i  virt_net -eennvvv
tcpdump: listening on virt_net, link-type EN10MB (Ethernet), capture size 262144 bytes
23:35:47.289500 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 49122, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.107.251 > 10.10.107.252: ICMP echo request, id 3915, seq 1, length 64
23:35:47.289505 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 49122, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.107.252 > 10.10.107.251: ICMP echo reply, id 3915, seq 1, length 64 (wrong icmp cksum 2f64 (->3764)!)
23:35:48.290755 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 49213, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.107.251 > 10.10.107.252: ICMP echo request, id 3915, seq 2, length 64
23:35:48.290760 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 49213, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.107.252 > 10.10.107.251: ICMP echo reply, id 3915, seq 2, length 64 (wrong icmp cksum 405e (->485e)!)
^C
4 packets captured
4 packets received by filter
0 packets dropped by kernel
[root@centos7 ~]# 
```