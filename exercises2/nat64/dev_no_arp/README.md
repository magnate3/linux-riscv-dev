


```
21: virt_net: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UNKNOWN group default qlen 1000
    link/ether 88:88:88:88:88:88 peer ff:ff:ff:ff:ff:ff
    inet6 fe80::6b52:aed:4fa5:bc8d/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
```

```
[root@centos7 dev_no_arp]# insmod  net_device2.ko 
[root@centos7 dev_no_arp]#  ip a  add  10.10.108.251/24 dev virt_net
[root@centos7 dev_no_arp]# ip l set virt_net  up
[root@centos7 dev_no_arp]# ping 10.10.108.252
PING 10.10.108.252 (10.10.108.252) 56(84) bytes of data.
64 bytes from 10.10.108.252: icmp_seq=1 ttl=64 time=0.036 ms
64 bytes from 10.10.108.252: icmp_seq=2 ttl=64 time=0.015 ms
^C
--- 10.10.108.252 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1001ms
rtt min/avg/max/mdev = 0.015/0.025/0.036/0.011 ms
[root@centos7 dev_no_arp]# 
```

目的mac 和源mac一样，没有arp请求    
```
[root@centos7 ~]# tcpdump -i virt_net icmp -env
tcpdump: listening on virt_net, link-type EN10MB (Ethernet), capture size 262144 bytes
02:32:06.488860 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 61690, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.108.251 > 10.10.108.252: ICMP echo request, id 10927, seq 1, length 64
02:32:06.488869 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 61690, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.108.252 > 10.10.108.251: ICMP echo reply, id 10927, seq 1, length 64 (wrong icmp cksum 80d7 (->88d7)!)
02:32:07.489909 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 61694, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.108.251 > 10.10.108.252: ICMP echo request, id 10927, seq 2, length 64
02:32:07.489915 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 61694, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.108.252 > 10.10.108.251: ICMP echo reply, id 10927, seq 2, length 64 (wrong icmp cksum 56d2 (->5ed2)!)
02:32:08.490942 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 61786, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.108.251 > 10.10.108.252: ICMP echo request, id 10927, seq 3, length 64
02:32:08.490951 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 61786, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.108.252 > 10.10.108.251: ICMP echo reply, id 10927, seq 3, length 64 (wrong icmp cksum 51cd (->59cd)!)
02:32:09.491995 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 61849, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.108.251 > 10.10.108.252: ICMP echo request, id 10927, seq 4, length 64
02:32:09.492002 88:88:88:88:88:88 > 88:88:88:88:88:88, ethertype IPv4 (0x0800), length 98: (tos 0x0, ttl 64, id 61849, offset 0, flags [DF], proto ICMP (1), length 84)
    10.10.108.252 > 10.10.108.251: ICMP echo reply, id 10927, seq 4, length 64 (wrong icmp cksum 31c8 (->39c8)!)
^C
8 packets captured
8 packets received by filter
0 packets dropped by kernel
[root@centos7 ~]#
```

# test2


```
[root@centos7 dev_no_arp]# ip a  add  10.10.107.251/24 dev virt_net
[root@centos7 dev_no_arp]# ip l set virt_net up
[root@centos7 dev_no_arp]# ip a sh virt_net
11: virt_net: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UNKNOWN group default qlen 1000
    link/none 
    inet 10.10.107.251/24 scope global virt_net
       valid_lft forever preferred_lft forever
    inet6 fe80::b75f:17b7:a413:8a11/64 scope link flags 800 
       valid_lft forever preferred_lft forever
[root@centos7 dev_no_arp]# ping 10.10.107.252
PING 10.10.107.252 (10.10.107.252) 56(84) bytes of data.
64 bytes from 10.10.107.252: icmp_seq=1 ttl=64 time=0.035 ms
64 bytes from 10.10.107.252: icmp_seq=2 ttl=64 time=0.016 ms
^C
--- 10.10.107.252 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1001ms
rtt min/avg/max/mdev = 0.016/0.025/0.035/0.010 ms
[root@centos7 dev_no_arp]# 
```

但是tcpdump    

```
[root@centos7 ~]# tcpdump -i virt_net -eennvvv
tcpdump: listening on virt_net, link-type RAW (Raw IP), capture size 262144 bytes
02:14:25.742167 ip: unknown ip 0
02:14:25.742174 ip: unknown ip 0
02:14:26.744044 ip: unknown ip 0
02:14:26.744047 ip: unknown ip 0
^C
4 packets captured
4 packets received by filter
0 packets dropped by kernel
[root@centos7 ~]# 
```
这是因为virt_net->type = ARPHRD_NONE     
```
        virt_net->netdev_ops= &net_ops;
        virt_net->flags = IFF_NOARP;
        virt_net->dev_addr[0] = 0x88;
        virt_net->dev_addr[1] = 0x88;
        virt_net->dev_addr[2] = 0x88;
        virt_net->dev_addr[3] = 0x88;
        virt_net->dev_addr[4] = 0x88;
        virt_net->dev_addr[5] = 0x88;

        virt_net->hard_header_len = 0;
        virt_net->mtu = 1500;
        virt_net->addr_len = ETH_ALEN;
        virt_net->type = ARPHRD_NONE;
        virt_net->needed_headroom = ETH_HLEN;
```

```
static inline bool dev_is_mac_header_xmit(const struct net_device *dev)
{
        switch (dev->type) {
        case ARPHRD_TUNNEL:
        case ARPHRD_TUNNEL6:
        case ARPHRD_SIT:
        case ARPHRD_IPGRE:
        case ARPHRD_VOID:
        case ARPHRD_NONE:
        case ARPHRD_RAWIP:
                return false;
        default:
                return true;
        }

```


> ##  eth_type_trans

+ 1 分配一个新的skb并且将帧拷贝到skb的数据区；   
+ 2 对skb的dev、protocol、pkt_type字段初始化；   
+ 3 让skb的mac_header执行帧的开头(skb_reset_mac_header)，data指向高层协议数据包的开头skb->data (skb_pull(skb, ETH_HLEN))。      

```
__be16 eth_type_trans(struct sk_buff *skb, struct net_device *dev)
{
	struct ethhdr *eth;
	unsigned char *rawp;
	
	// 输入数据包中，该字段表示数据包是被谁收到的
	skb->dev = dev;
	// 让skb->mac_header指向skb->data位置，即指向帧的开头
	skb_reset_mac_header(skb);
	// skb->data指针前移，剥掉以太网帧首部的14个字节，这样skb->data将指向上层协议报文的开头
	skb_pull(skb, ETH_HLEN);
	eth = eth_hdr(skb);

	// 根据mac地址类型确定输入数据包的类型，即skb->pkt_type字段
	if (is_multicast_ether_addr(eth->h_dest)) {
		if (!compare_ether_addr(eth->h_dest, dev->broadcast))
			skb->pkt_type = PACKET_BROADCAST;
		else
			skb->pkt_type = PACKET_MULTICAST;
	}
	/*
	 *      This ALLMULTI check should be redundant by 1.4
	 *      so don't forget to remove it.
	 *
	 *      Seems, you forgot to remove it. All silly devices
	 *      seems to set IFF_PROMISC.
	 */
	else if (1 /*dev->flags&IFF_PROMISC */ ) {
		if (unlikely(compare_ether_addr(eth->h_dest, dev->dev_addr)))
			skb->pkt_type = PACKET_OTHERHOST;
	}

	// 如果帧首部的协议字段值超过1536，那么h_proto字段的含义就是上层协议号，这是标准的以太网帧
	if (ntohs(eth->h_proto) >= 1536)
		return eth->h_proto;

	// 如果协议字段小于1536，那么h_proto代表的就是报文的长度，从rawp开始继续解析报头
	rawp = skb->data;
	/*
	 *      This is a magic hack to spot IPX packets. Older Novell breaks
	 *      the protocol design and runs IPX over 802.3 without an 802.2 LLC
	 *      layer. We look for FFFF which isn't a used 802.2 SSAP/DSAP. This
	 *      won't work for fault tolerant netware but does for the rest.
	 */
	// 如果帧数据的前两个字节为0xFFFF，那么这是一个802.3以太网数据帧
	if (*(unsigned short *)rawp == 0xFFFF)
		return htons(ETH_P_802_3);

	/*
	 *      Real 802.2 LLC
	 */
	// 其余情况说明这是一个802.2以太网数据帧
	return htons(ETH_P_802_2);
}
```

eth_type_trans()函数还有一个重要任务是从L2的角度确定输入数据包的类型，将确认结果记录到skb->pakcet_type，该字段高层协议会进行判断，进而决定如何处理该数据包。

内核从L2角度总共定义了如下几种数据包类型：
```
/* Packet types */
#define PACKET_HOST		0			/* To us		*/
#define PACKET_BROADCAST	1		/* To all		*/
#define PACKET_MULTICAST	2		/* To group		*/
#define PACKET_OTHERHOST	3		/* To someone else 	*/
#define PACKET_OUTGOING		4		/* Outgoing of any type */
/* These ones are invisible by user level */
#define PACKET_LOOPBACK		5		/* MC/BRD frame looped back */
#define PACKET_FASTROUTE	6		/* Fastrouted frame	*/
```

> ## recv_packet


```
static void gsm_mux_rx_netchar(struct gsm_dlci *dlci,
				const unsigned char *in_buf, int size)
{
	struct net_device *net = dlci->net;
	struct sk_buff *skb;
	struct gsm_mux_net *mux_net = netdev_priv(net);
	muxnet_get(mux_net);

	/* Allocate an sk_buff */
	skb = dev_alloc_skb(size + NET_IP_ALIGN);
	if (!skb) {
		/* We got no receive buffer. */
		net->stats.rx_dropped++;
		muxnet_put(mux_net);
		return;
	}
	skb_reserve(skb, NET_IP_ALIGN);
	skb_put_data(skb, in_buf, size);

	skb->dev = net;
	skb->protocol = htons(ETH_P_IP);

	/* Ship it off to the kernel */
	netif_rx(skb);

	/* update out statistics */
	net->stats.rx_packets++;
	net->stats.rx_bytes += size;
	muxnet_put(mux_net);
	return;
}


```