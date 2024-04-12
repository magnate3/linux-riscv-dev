

```
root@ubuntux86:# ip l set up dev sn0
root@ubuntux86:# ip l set up dev sn1
root@ubuntux86:# ip a add 1.1.0.1/24 dev sn0
root@ubuntux86:# ip a add 1.1.1.2/24 dev sn1
root@ubuntux86:# ip r | grep 1.1
1.1.0.0/24 dev sn0 proto kernel scope link src 1.1.0.1 
1.1.1.0/24 dev sn1 proto kernel scope link src 1.1.1.2 
10.11.11.0/24 dev enx00e04c3662aa proto kernel scope link src 10.11.11.82 metric 100 
10.11.12.0/24 dev enp0s31f6 proto kernel scope link src 10.11.12.82 metric 101 
192.168.5.0/24 dev enp0s31f6 proto kernel scope link src 192.168.5.82 metric 101 
```

```
root@ubuntux86:# ping 1.1.0.2 -c 1
PING 1.1.0.2 (1.1.0.2) 56(84) bytes of data.
64 bytes from 1.1.0.2: icmp_seq=1 ttl=64 time=0.107 ms

--- 1.1.0.2 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.107/0.107/0.107/0.000 ms
root@ubuntux86:# ip  a sh sn0
11: sn0: <BROADCAST,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 1000
    link/ether 00:53:4e:55:4c:30 brd ff:ff:ff:ff:ff:ff
    inet 1.1.0.1/24 scope global sn0
       valid_lft forever preferred_lft forever
    inet6 fe80::6ad6:ea43:c599:d220/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
root@ubuntux86:# ip  a sh sn1
12: sn1: <BROADCAST,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 1000
    link/ether 00:53:4e:55:4c:31 brd ff:ff:ff:ff:ff:ff
    inet6 fe80::34a0:95be:a52b:e959/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
root@ubuntux86:# 
```