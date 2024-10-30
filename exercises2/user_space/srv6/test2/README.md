
# 参考项目
[SRV6-sample](https://github.com/torukita/SRV6-sample/tree/4a42aa15b7ccd030ddc908c0b6cf396d66f61a9c)

# create network topo
```
cd namespace
sudo ./srv6-AB.sh
```

# test
![images](test2.png)

```
root@ubuntux86:# ip netns exec host1 ping fc00:000b::10
PING fc00:000b::10(fc00:b::10) 56 data bytes
64 bytes from fc00:b::10: icmp_seq=1 ttl=63 time=0.280 ms
64 bytes from fc00:b::10: icmp_seq=2 ttl=63 time=0.126 ms
64 bytes from fc00:b::10: icmp_seq=3 ttl=63 time=0.135 ms
^C
--- fc00:000b::10 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2041ms
rtt min/avg/max/mdev = 0.126/0.180/0.280/0.070 ms
```

# 退出


输入exit   
```
[SRv6(0.0.4)]root@ubuntux86:# exit
exit
-----
Cleaned Virtual Network Topology successfully
-----
ip netns del host1
ip netns del routerA
ip netns del routerB
ip netns del host2
[SRv6(0.0.4)]root@ubuntux86:# 
```

```
[SRv6(0.0.4)]root@ubuntux86:# ip netns exec host1 ping fc00:000b::10
Cannot open network namespace "host1": No such file or directory
[SRv6(0.0.4)]root@ubuntux86:# 
```