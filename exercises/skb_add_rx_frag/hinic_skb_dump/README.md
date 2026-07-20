#  NETIF_F_SG

## ip a
```
[root@centos7 linux-4.14.115]# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: enp125s0f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b0:08:75:5f:b8:5b brd ff:ff:ff:ff:ff:ff
    inet 10.10.16.251/24 brd 10.10.16.255 scope global noprefixroute enp125s0f0
       valid_lft forever preferred_lft forever
    inet6 fe80::a82e:8486:712:201a/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
3: enp125s0f1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether b0:08:75:5f:b8:5c brd ff:ff:ff:ff:ff:ff
4: enp125s0f2: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether b0:08:75:5f:b8:5d brd ff:ff:ff:ff:ff:ff
5: enp125s0f3: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b0:08:75:5f:b8:5e brd ff:ff:ff:ff:ff:ff
8: virbr0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default qlen 1000
    link/ether 52:54:00:af:8e:96 brd ff:ff:ff:ff:ff:ff
    inet 192.168.122.1/24 brd 192.168.122.255 scope global virbr0
       valid_lft forever preferred_lft forever
9: virbr0-nic: <BROADCAST,MULTICAST> mtu 1500 qdisc pfifo_fast master virbr0 state DOWN group default qlen 1000
    link/ether 52:54:00:af:8e:96 brd ff:ff:ff:ff:ff:ff
10: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default 
    link/ether 02:42:78:3e:53:65 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 scope global docker0
       valid_lft forever preferred_lft forever
19: enp5s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 44:a1:91:a4:9c:0b brd ff:ff:ff:ff:ff:ff
20: enp6s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 44:a1:91:a4:9c:0c brd ff:ff:ff:ff:ff:ff
    inet 192.168.10.251/24 scope global enp6s0
       valid_lft forever preferred_lft forever
```

## enp125s0f0

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_add_rx_frag/hinic_skb_dump/enp125s0f0.png)

### hook

```

static struct nf_hook_ops test_hookops = {
    .pf = NFPROTO_IPV4,
    .priority = NF_IP_PRI_MANGLE,
    //.hooknum = NF_INET_PRE_ROUTING,
    //.hooknum = NF_INET_POST_ROUTING,
    //.hooknum = NF_INET_LOCAL_IN,
    .hooknum = NF_INET_LOCAL_OUT,
    .hook = test_hookfn,
#if LINUX_VERSION_CODE < KERNEL_VERSION(4,4,0)
    .owner = THIS_MODULE,
#endif
};
```

```
[root@centos7 linux-4.14.115]# ethtool -K  enp125s0f0  tx-scatter-gather on
Actual changes:
scatter-gather: on
        tx-scatter-gather: on
tcp-segmentation-offload: on
        tx-tcp-segmentation: on
        tx-tcp6-segmentation: on
generic-segmentation-offload: on
[root@centos7 linux-4.14.115]# ethtool -K  enp125s0f0  tx-scatter-gather off
Actual changes:
scatter-gather: off
        tx-scatter-gather: off
tcp-segmentation-offload: off
        tx-tcp-segmentation: off [requested on]
        tx-tcp6-segmentation: off [requested on]
generic-segmentation-offload: off [requested on]
[root@centos7 linux-4.14.115]# 
```
***tx-scatter-gather off 关闭与否， ping -c 1  -s 6500 10.10.16.251 只有fraglist***



# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/hinic/new_id 
```
[root@centos7 igb-uio]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/hinic/new_id 
[root@centos7 igb-uio]# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: enp125s0f0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b0:08:75:5f:b8:5b brd ff:ff:ff:ff:ff:ff
    inet 10.10.16.251/24 brd 10.10.16.255 scope global noprefixroute enp125s0f0
       valid_lft forever preferred_lft forever
    inet6 fe80::a82e:8486:712:201a/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
3: enp125s0f1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether b0:08:75:5f:b8:5c brd ff:ff:ff:ff:ff:ff
4: enp125s0f2: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether b0:08:75:5f:b8:5d brd ff:ff:ff:ff:ff:ff
5: enp125s0f3: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether b0:08:75:5f:b8:5e brd ff:ff:ff:ff:ff:ff
8: virbr0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default qlen 1000
    link/ether 52:54:00:af:8e:96 brd ff:ff:ff:ff:ff:ff
    inet 192.168.122.1/24 brd 192.168.122.255 scope global virbr0
       valid_lft forever preferred_lft forever
9: virbr0-nic: <BROADCAST,MULTICAST> mtu 1500 qdisc pfifo_fast master virbr0 state DOWN group default qlen 1000
    link/ether 52:54:00:af:8e:96 brd ff:ff:ff:ff:ff:ff
10: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default 
    link/ether 02:42:78:3e:53:65 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 scope global docker0
       valid_lft forever preferred_lft forever
11: enp5s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 44:a1:91:a4:9c:0b brd ff:ff:ff:ff:ff:ff
12: enp6s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 44:a1:91:a4:9c:0c brd ff:ff:ff:ff:ff:ff
[root@centos7 igb-uio]# ip  a add 192.168.10.251/24 dev  enp6s0
```

# ping -c 1  -s 6500  192.168.10.81  

```
[root@centos7 hinic]# ip a add 192.168.10.251/24 dev  enp6s0
[root@centos7 hinic]# ping -c 1  -s 6500  192.168.10.81
PING 192.168.10.81 (192.168.10.81) 6500(6528) bytes of data.
6508 bytes from 192.168.10.81: icmp_seq=1 ttl=64 time=550 ms

--- 192.168.10.81 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 550.582/550.582/550.582/0.000 ms
[root@centos7 hinic]# 
```
 
## hooknum = NF_INET_LOCAL_OUT

```
static struct nf_hook_ops test_hookops = {
    .pf = NFPROTO_IPV4,
    .priority = NF_IP_PRI_MANGLE,
    //.hooknum = NF_INET_PRE_ROUTING,
     //.hooknum = NF_INET_POST_ROUTING,
   .hooknum = NF_INET_LOCAL_OUT,
    .hook = test_hookfn,
#if LINUX_VERSION_CODE < KERNEL_VERSION(4,4,0)
    .owner = THIS_MODULE,
#endif
};
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_add_rx_frag/hinic_skb_dump/ping16.81.png)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_add_rx_frag/hinic_skb_dump/ping10.81.png)

两个网卡都没有frag_info 有frag_list


# ping -c 1  -s 6500  192.168.10.251  and ping -c 1  -s 6500 10.10.16.251


## hooknum = NF_INET_LOCAL_IN

```
static struct nf_hook_ops test_hookops = {
    .pf = NFPROTO_IPV4,
    .priority = NF_IP_PRI_MANGLE,
    //.hooknum = NF_INET_PRE_ROUTING,
     //.hooknum = NF_INET_POST_ROUTING,
   .hooknum = NF_INET_LOCAL_IN,
    .hook = test_hookfn,
#if LINUX_VERSION_CODE < KERNEL_VERSION(4,4,0)
    .owner = THIS_MODULE,
#endif
};
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_add_rx_frag/hinic_skb_dump/ping16.251.png)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_add_rx_frag/hinic_skb_dump/ping10.251.png)

#  enp6s0

## dev->features&NETIF_F_SG
dev->features&NETIF_F_SG: 1 






