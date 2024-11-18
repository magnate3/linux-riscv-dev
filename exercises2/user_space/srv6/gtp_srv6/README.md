
[How To run SRv6 Mobile Uplane POC](https://github.com/ebiken/p4srv6/blob/c5049a80ba366f0cacf20b8bfb88b21540150383/archive/demo/srv6/demo1-SRv6MobileUplane-dropin.md) 

# libgtpnl
```
$ git clone git://git.osmocom.org/libgtpnl.git
$ cd libgtpnl
 autoreconf -fi
 ./configure  --prefix=/opt/gtp
 make -j8
 root@ubuntux86:# ls /opt/gtp
bin  include  lib
root@ubuntux86:# 
```
```
export PATH=$PATH:/opt/gtp/bin/
```

# p4 switch

```
root@ubuntux86:# p4c --target bmv2 --arch v1model  ./archive/p4src/switch.p4
./archive/p4src/switch.p4(154): [--Wwarn=unused] warning: Table local_mac is not used; removing
    table local_mac {
          ^^^^^^^^^
root@ubuntux86:# find ./ -name ns-hosts-srv6-demo1.sh
./archive/demo/srv6/ns-hosts-srv6-demo1.sh
root@ubuntux86:# ./archive/demo/srv6/ns-hosts-srv6-demo1.sh -c
```



```
simple_switch_CLI  < fw.txt 
```


```
root@ubuntux86:# ip netns exec host1 ping 172.20.0.2
PING 172.20.0.2 (172.20.0.2) 56(84) bytes of data.
64 bytes from 172.20.0.2: icmp_seq=1 ttl=64 time=5.84 ms
64 bytes from 172.20.0.2: icmp_seq=2 ttl=64 time=4.45 ms
^C
--- 172.20.0.2 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1002ms
rtt min/avg/max/mdev = 4.450/5.145/5.841/0.695 ms
root@ubuntux86:# ip netns exec host1 ping6 fd01::2
PING fd01::2(fd01::2) 56 data bytes
64 bytes from fd01::2: icmp_seq=1 ttl=64 time=6.64 ms
64 bytes from fd01::2: icmp_seq=2 ttl=64 time=5.99 ms
^C
--- fd01::2 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1002ms
rtt min/avg/max/mdev = 5.992/6.315/6.638/0.323 ms
root@ubuntux86:# 
```

# gtp   


## shell1
```
root@ubuntux86:# ip netns exec host1 ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet 172.99.0.1/32 scope global lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
6: veth1@if5: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1000
    link/ether ba:c3:2c:2d:0f:13 brd ff:ff:ff:ff:ff:ff link-netnsid 0
    inet 172.20.0.1/24 scope global veth1
       valid_lft forever preferred_lft forever
    inet6 fd01::1/64 scope global 
       valid_lft forever preferred_lft forever
    inet6 fe80::b8c3:2cff:fe2d:f13/64 scope link 
       valid_lft forever preferred_lft forever
root@ubuntux86:# ip netns exec host1 gtp-link add gtp1 ip
WARNING: attaching dummy socket descriptors. Keep this process running for testing purposes.

```

## shell2

```
root@ubuntux86:# export PATH=$PATH:/opt/gtp/bin/
root@ubuntux86:# ip netns exec host1 gtp-tunnel add gtp1 v1 200 100 172.99.0.2 172.20.0.2
root@ubuntux86:# 

```

## shell3


```
root@ubuntux86:# export PATH=$PATH:/opt/gtp/bin/
root@ubuntux86:# ip netns exec host2 gtp-link add gtp2  ip 
WARNING: attaching dummy socket descriptors. Keep this process running for testing purposes.



```


## shell2

```
root@ubuntux86:# ip netns exec host2 gtp-tunnel add gtp2 v1 100 200 172.99.0.1 172.20.0.1
root@ubuntux86:#  ip netns exec host2 ip route add 172.99.0.1/32 dev gtp2
root@ubuntux86:# 

```


```
simple_switch_CLI  <  srv6.txt 
```


```

root@ubuntux86:# ip netns exec host1 ping 172.99.0.2
PING 172.99.0.2 (172.99.0.2) 56(84) bytes of data.
^C
--- 172.99.0.2 ping statistics ---
89 packets transmitted, 0 received, 100% packet loss, time 90063ms

root@ubuntux86:# 
```


```
root@ubuntux86:# ip netns exec host1 tcpdump  -i veth1
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on veth1, link-type EN10MB (Ethernet), capture size 262144 bytes
^C10:28:59.977533 IP 172.20.0.1.2152 > 172.20.0.2.2152: UDP, length 92
10:29:00.979479 IP 172.20.0.1.2152 > 172.20.0.2.2152: UDP, length 92
10:29:01.993847 IP 172.20.0.1.2152 > 172.20.0.2.2152: UDP, length 92
10:29:03.017757 IP 172.20.0.1.2152 > 172.20.0.2.2152: UDP, length 92
10:29:04.041582 IP 172.20.0.1.2152 > 172.20.0.2.2152: UDP, length 92
```