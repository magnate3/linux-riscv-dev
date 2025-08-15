

# ipv6   

```
[root@centos7 nat64_test]# insmod  nat64_device.ko 
[root@centos7 nat64_test]# ip a add 2001:db8::a0a:6751/96 dev nat64
[root@centos7 nat64_test]# ip l set nat64 up
[root@centos7 nat64_test]# ping6 2001:db8::a0a:6752 
PING 2001:db8::a0a:6752(2001:db8::a0a:6752) 56 data bytes
168 bytes from 2001:db8::a0a:6752: icmp_seq=1 ttl=64 time=0.046 ms
168 bytes from 2001:db8::a0a:6752: icmp_seq=2 ttl=64 time=0.018 ms
^C
--- 2001:db8::a0a:6752 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1001ms
rtt min/avg/max/mdev = 0.018/0.032/0.046/0.014 ms
[root@centos7 nat64_test]# 
```