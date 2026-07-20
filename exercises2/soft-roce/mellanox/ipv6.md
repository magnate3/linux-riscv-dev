

# node1

```
4: ens6: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 08:c0:eb:da:e2:ca brd ff:ff:ff:ff:ff:ff
    inet 172.17.24.56/24 brd 172.17.242.255 scope global ens6
       valid_lft forever preferred_lft forever
    inet6 fe80::ac0:ebff:feda:e2ca/64 scope link 
       valid_lft forever preferred_lft forever
8: ib0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 2044 qdisc mq state UP group default qlen 256
    link/infiniband 20:00:10:29:fe:80:00:00:00:00:00:00:08:c0:eb:03:00:b6:a0:b4 brd 00:ff:ff:ff:ff:12:40:1b:ff:ff:00:00:00:00:00:00:ff:ff:ff:ff
    inet 172.162.42.56/24 brd 172.16.242.255 scope global ib0
       valid_lft forever preferred_lft forever
    inet6 fe80::ac0:eb03:b6:a0b4/64 scope link 
       valid_lft forever preferred_lft forever
```

# node2

```
5: ens14f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 08:c0:eb:40:91:7e brd ff:ff:ff:ff:ff:ff
    inet 192.168.24.47/24 brd 192.168.242.255 scope global ens14f0
       valid_lft forever preferred_lft forever
    inet6 fe80::ac0:ebff:fe40:917e/64 scope link 
       valid_lft forever preferred_lft forever
8: ib0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 2044 qdisc mq state UP group default qlen 256
    link/infiniband 20:00:10:29:fe:80:00:00:00:00:00:00:08:c0:eb:03:00:ea:50:e6 brd 00:ff:ff:ff:ff:12:40:1b:ff:ff:00:00:00:00:00:00:ff:ff:ff:ff
    inet 172.162.42.47/24 brd 172.16.242.255 scope global ib0
       valid_lft forever preferred_lft forever
    inet6 fe80::ac0:eb03:ea:50e6/64 scope link 
       valid_lft forever preferred_lft forever
```
## ping6 fe80::ac0:eb03:b6:a0b4%ib0
```
root@ubuntu:/home/ubuntu# ping6 fe80::ac0:eb03:b6:a0b4%ib0
PING fe80::ac0:eb03:b6:a0b4%ib0(fe80::ac0:eb03:b6:a0b4%ib0) 56 data bytes
64 bytes from fe80::ac0:eb03:b6:a0b4%ib0: icmp_seq=1 ttl=64 time=1009 ms
64 bytes from fe80::ac0:eb03:b6:a0b4%ib0: icmp_seq=2 ttl=64 time=0.460 ms
64 bytes from fe80::ac0:eb03:b6:a0b4%ib0: icmp_seq=3 ttl=64 time=0.065 ms
64 bytes from fe80::ac0:eb03:b6:a0b4%ib0: icmp_seq=4 ttl=64 time=0.041 ms
64 bytes from fe80::ac0:eb03:b6:a0b4%ib0: icmp_seq=5 ttl=64 time=0.039 ms
^C
--- fe80::ac0:eb03:b6:a0b4%ib0 ping statistics ---
5 packets transmitted, 5 received, 0% packet loss, time 4081ms
rtt min/avg/max/mdev = 0.039/202.111/1009.952/403.920 ms, pipe 2
root@ubuntu:/home/ubuntu# ping fe80::ac0:ebff:feda:e2ca%ens14f0
PING fe80::ac0:ebff:feda:e2ca%ens14f0(fe80::ac0:ebff:feda:e2ca%ens14f0) 56 data bytes
64 bytes from fe80::ac0:ebff:feda:e2ca%ens14f0: icmp_seq=1 ttl=64 time=0.258 ms
64 bytes from fe80::ac0:ebff:feda:e2ca%ens14f0: icmp_seq=2 ttl=64 time=0.062 ms
64 bytes from fe80::ac0:ebff:feda:e2ca%ens14f0: icmp_seq=3 ttl=64 time=0.043 ms
^C
--- fe80::ac0:ebff:feda:e2ca%ens14f0 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2055ms
rtt min/avg/max/mdev = 0.043/0.121/0.258/0.097 ms
```

## ping6 fe80::ac0:ebff:feda:e2ca%ens14f0
```
root@ubuntu:/home/ubuntu# ping6 fe80::ac0:ebff:feda:e2ca%ens14f0
PING fe80::ac0:ebff:feda:e2ca%ens14f0(fe80::ac0:ebff:feda:e2ca%ens14f0) 56 data bytes
64 bytes from fe80::ac0:ebff:feda:e2ca%ens14f0: icmp_seq=1 ttl=64 time=0.077 ms
64 bytes from fe80::ac0:ebff:feda:e2ca%ens14f0: icmp_seq=2 ttl=64 time=0.068 ms
^C
--- fe80::ac0:ebff:feda:e2ca%ens14f0 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1023ms
rtt min/avg/max/mdev = 0.068/0.072/0.077/0.009 ms
root@ubuntu:/home/ubuntu# 
```