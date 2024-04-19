
# run

```
[root@centos7 nat64_udp_frag]# insmod nat64_device.ko 
[root@centos7 nat64_udp_frag]# ip a add 2001:db8::a0a:6751/96 dev nat64
[root@centos7 nat64_udp_frag]# ip l set nat64 up
[root@centos7 nat64_udp_frag]# ./udp_cli 
client send susscessfully 
[root@centos7 nat64_udp_frag]# ping6  2001:db8::a0a:6752
PING 2001:db8::a0a:6752(2001:db8::a0a:6752) 56 data bytes
64 bytes from 2001:db8::a0a:6752: icmp_seq=1 ttl=64 time=0.047 ms
64 bytes from 2001:db8::a0a:6752: icmp_seq=2 ttl=64 time=0.017 ms
^C
--- 2001:db8::a0a:6752 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1054ms
rtt min/avg/max/mdev = 0.017/0.032/0.047/0.015 ms
[root@centos7 nat64_udp_frag]# 
```

#  dev->needed_headroom for skb_push
```
static void nat64_setup(struct net_device *dev)
{
        struct nat64_if_info *nif = (struct nat64_if_info *)netdev_priv(dev);

        /* Point-to-Point interface */
        dev->netdev_ops = &nat64_netdev_ops;
        dev->hard_header_len = 0;
        dev->addr_len = 0;
        dev->mtu = 1500;
        dev->needed_headroom = sizeof(struct ip6) - sizeof(struct ip4) + sizeof(ETH_HLEN);

        /* Zero header length */
        dev->type = ARPHRD_NONE;
        dev->flags = IFF_POINTOPOINT | IFF_NOARP | IFF_MULTICAST;
        dev->tx_queue_len = 500;  /* We prefer our own queue length */

        /* Setup private data */
        memset(nif, 0x0, sizeof(nif[0]));
        nif->dev = dev;
}
```



# xlate_header_6to4  xlate_payload_6to4
```
static void xlate_6to4_data(struct pkt *p)
{
	struct {
		struct ip4 ip4;
	} __attribute__ ((__packed__)) header;
	struct sk_buff *skb = p->skb;

	if (map_ip6_to_ip4(&header.ip4.dest, &p->ip6->dest, 0)) {
		host_send_icmp6_error(1, 0, 0, p);
		kfree_skb(skb);
		return;
	}

	if (map_ip6_to_ip4(&header.ip4.src, &p->ip6->src, 1)) {
		host_send_icmp6_error(1, 5, 0, p);
		kfree_skb(skb);
		return;
	}

	if (sizeof(struct ip6) + p->header_len + p->data_len > gcfg.mtu) {
		host_send_icmp6_error(2, 0, gcfg.mtu, p);
		kfree_skb(skb);
		return;
	}

	xlate_header_6to4(p, &header.ip4, p->data_len);
	--header.ip4.ttl;

	if (xlate_payload_6to4(p, &header.ip4) < 0) {
		kfree_skb(skb);
		return;
	}

	header.ip4.cksum = htons(swap_u16(ip_checksum(&header.ip4, sizeof(header.ip4))));

	skb_pull(skb, (unsigned)(p->data - sizeof(header) - skb->data));
	skb->protocol = htons(ETH_P_IP);
	skb_reset_network_header(skb);
	memcpy(ip_hdr(skb), &header, sizeof(header));
	skb->dev = p->dev;

	p->dev->stats.rx_bytes += skb->len;
	p->dev->stats.rx_packets++;
	netif_rx(skb);
}
```