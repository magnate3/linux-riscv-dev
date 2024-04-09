
#arp
IFF_NOARP    ARPHRD_NONE   
```
static void nat64_setup(struct net_device *dev)
{
        struct nat64_if_info *nif = (struct nat64_if_info *)netdev_priv(dev);

        /* Point-to-Point interface */
        dev->netdev_ops = &nat64_netdev_ops;
        dev->hard_header_len = 0;
        dev->addr_len = 0;
        dev->mtu = 1500;
        dev->needed_headroom = sizeof(struct ip6) - sizeof(struct ip4);

        /* Zero header length */
        dev->type = ARPHRD_NONE;
        dev->flags = IFF_POINTOPOINT | IFF_NOARP | IFF_MULTICAST;
        dev->tx_queue_len = 500;  /* We prefer our own queue length */

        /* Setup private data */
        memset(nif, 0x0, sizeof(nif[0]));
        nif->dev = dev;
}
```


```
[root@centos7 src]# insmod tayga.ko ipv6_addr=2001:200:0:ff99::251  ipv4_addr=192.168.255.1  prefix=2001:200:0:ff99::/96 dynamic_pool=192.168.0.0/16
[root@centos7 src]# 
```


```
[root@centos7 src]# insmod tayga.ko ipv6_addr=2001:200:0:ff99::10.10.104.251  ipv4_addr=10.10.104.251  prefix=2001:200:0:ff99::/96 dynamic_pool=10.10.104.0/24
[root@centos7 src]# 
```

```
[root@centos7 src]# ip link set nat64 up
[root@centos7 src]# ip a add 10.10.104.251/24 dev nat64
         ip a add 2001:200:0:ff99::10.10.104.251/96 dev nat64
[root@centos7 src]# ip -6 route add 2001:200:0:ff99::/96 dev nat64
[root@centos7 src]# 
```

```
iptables -t nat  -A POSTROUTING -s 10.10.105.20/24 -d 10.10.103.81 -j SNAT --to-source  10.10.103.251
```

+ 4 rp_filter

```
sysctl -w net.ipv4.conf.all.rp_filter=0
sysctl -w net.ipv4.conf.nat64.rp_filter=0

```