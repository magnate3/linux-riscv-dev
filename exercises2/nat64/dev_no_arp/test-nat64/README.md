

# 实现
```
        len = skb->len;
        if (len < sizeof(struct icmphdr) + sizeof(struct iphdr)) {
                pr_info("snull: Hmm... packet too short (%i octets)\n", len);
                return -1;
        }
        if(ETH_P_IP != ntohs(skb->protocol))
        {
                pr_info("not ip hdr \n");
                return -1;
        }
        ih = (struct iphdr *)(skb->data);
```
直接传输ip报文，没有ether 头    


setup没有采用ether_setup    
```

static void nat64_setup(struct net_device *dev)
{
        struct nat64_if_info *nif = (struct nat64_if_info *)netdev_priv(dev);

        /* Point-to-Point interface */
        dev->netdev_ops = &nat64_netdev_ops;
        dev->hard_header_len = 0;
        dev->addr_len = 0;
        dev->mtu = 1500;
        //dev->needed_headroom = sizeof(struct ip6) - sizeof(struct ip4);

        /* Zero header length */
        dev->type = ARPHRD_NONE;
        dev->flags = IFF_POINTOPOINT | IFF_NOARP | IFF_MULTICAST;
        dev->tx_queue_len = 500;  /* We prefer our own queue length */

        /* Setup private data */
        memset(nif, 0x0, sizeof(nif[0]));
        nif->dev = dev;
}

```



# run

```
[root@centos7 dev_no_arp]# insmod  net_device.ko 
[root@centos7 dev_no_arp]#  ip a  add  10.10.107.251/24 dev nat64
[root@centos7 dev_no_arp]# ip l set nat64 up
[root@centos7 dev_no_arp]# ping 10.10.107.252
PING 10.10.107.252 (10.10.107.252) 56(84) bytes of data.
64 bytes from 10.10.107.252: icmp_seq=1 ttl=64 time=7.78 ms
64 bytes from 10.10.107.252: icmp_seq=2 ttl=64 time=7.83 ms
64 bytes from 10.10.107.252: icmp_seq=3 ttl=64 time=7.84 ms
64 bytes from 10.10.107.252: icmp_seq=4 ttl=64 time=7.85 ms
```


```
[root@centos7 ~]# tcpdump -i nat64 -eennvv
tcpdump: listening on nat64, link-type RAW (Raw IP), capture size 262144 bytes
02:23:14.038146 ip: (tos 0x0, ttl 64, id 27001, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.107.251 > 10.10.107.252: ICMP echo request, id 9902, seq 14, length 64
02:23:14.038153 ip: (tos 0x0, ttl 64, id 27001, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.107.252 > 10.10.107.251: ICMP echo request, id 9902, seq 14, length 64
02:23:14.046329 ip: (tos 0x0, ttl 64, id 27002, offset 0, flags [none], proto ICMP (1), length 84)
    10.10.107.251 > 10.10.107.252: ICMP echo reply, id 9902, seq 14, length 64
02:23:14.046330 ip: (tos 0x0, ttl 64, id 27002, offset 0, flags [none], proto ICMP (1), length 84)
    10.10.107.252 > 10.10.107.251: ICMP echo reply, id 9902, seq 14, length 64
02:23:15.039565 ip: (tos 0x0, ttl 64, id 27003, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.107.251 > 10.10.107.252: ICMP echo request, id 9902, seq 15, length 64
02:23:15.039568 ip: (tos 0x0, ttl 64, id 27003, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.107.252 > 10.10.107.251: ICMP echo request, id 9902, seq 15, length 64
02:23:15.047735 ip: (tos 0x0, ttl 64, id 27004, offset 0, flags [none], proto ICMP (1), length 84)
    10.10.107.251 > 10.10.107.252: ICMP echo reply, id 9902, seq 15, length 64
02:23:15.047736 ip: (tos 0x0, ttl 64, id 27004, offset 0, flags [none], proto ICMP (1), length 84)
    10.10.107.252 > 10.10.107.251: ICMP echo reply, id 9902, seq 15, length 64
^C
8 packets captured
8 packets received by filter
0 packets dropped by kernel
[root@centos7 ~]# 
```