/*
 * Driver for NAT64 Virtual Network Interface.
 *
 * Author: Jianying Liu <rssnsj@gmail.com>
 * Date: 2014/01/21
 *
 * This source code is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 */

#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/etherdevice.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/poll.h>
#include <linux/sched.h>
#include <linux/netdevice.h>
#include <linux/if.h>
#include <linux/if_ether.h>
#include <linux/if_arp.h>
#include <net/sock.h>
#include <linux/ip.h>
#include <linux/icmp.h>
#include <linux/netdev_features.h>

#include "tayga.h"

struct nat64_if_info {
	struct net_device *dev;
};

static int virt_rs_packet(struct sk_buff *skb, struct net_device *dev)
{

	unsigned char *type;
	struct iphdr *ih;
	__be32 *saddr, *daddr, tmp;
	struct sk_buff *rx_skb;
	int ret;
        int len;
#if 0
	unsigned char tmp_dev_addr[ETH_ALEN];
	struct ethhdr *ethhdr;
	len = skb->len;
	if (len < sizeof(struct ethhdr) + sizeof(struct iphdr)) {
		pr_info("snull: Hmm... packet too short (%i octets)\n", len);
		return -1;
	}
	//对调ethhdr结构体 "源/目的"MAC地址*/
	ethhdr = (struct ethhdr *)skb->data;
        if(0x0800 != ntohs(ethhdr->h_proto)){
		pr_info("not ip hdr \n");
		return -1;
	}
	memcpy(tmp_dev_addr, ethhdr->h_dest, ETH_ALEN);
	memcpy(ethhdr->h_dest, ethhdr->h_source, ETH_ALEN);
	memcpy(ethhdr->h_source, tmp_dev_addr, ETH_ALEN);
	//对调iphdr结构体"源/目的" IP地址
	ih = (struct iphdr *)(skb->data + sizeof(struct ethhdr));
#else
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
#endif
        if(IPPROTO_ICMP != ih->protocol)
        {
		pr_info("not icmp hdr \n");
		return -1;
	}
	saddr = &ih->saddr;
	daddr = &ih->daddr;
	tmp = *saddr;
	*saddr = *daddr;
	*daddr = tmp;
        
        ih->check=0;
        ih->check = ip_fast_csum((unsigned char *)ih,ih->ihl);
	
	//之前是发送ping包0x08,需要改为0x00,表示接收ping包
	type = skb->data + sizeof(struct ethhdr) + sizeof(struct iphdr);
	*type = 0; 
 
	rx_skb = dev_alloc_skb(skb->len + 2);
	skb_reserve(rx_skb, 2);
	
	memcpy(skb_put(rx_skb, skb->len), skb->data, skb->len);
	rx_skb->dev = dev;	
	rx_skb->ip_summed = CHECKSUM_UNNECESSARY;
	//rx_skb->protocol = eth_type_trans(rx_skb, dev);
        rx_skb->protocol = htons(ETH_P_IP);
	ret=netif_rx(rx_skb);
	
	dev->stats.rx_packets++;        
	dev->stats.rx_bytes += skb->len;
	pr_info("rx_packets=%ld rx_bytes=%ld ret=%d\n",dev->stats.rx_packets,dev->stats.rx_bytes,ret);
	return NETDEV_TX_OK;
}
#if 0
static void host_handle_icmp6(struct pkt *p)
{
}
#endif
static inline u16 swap_u16(u16 val)
{
	return (val << 8) | (val >> 8);
}

static inline u32 swap_u32(u32 val)
{
	val = ((val << 8) & 0xff00ff00) | ((val >> 8) & 0xff00ff);
	return (val << 16) | (val >> 16);
}

static u16 ip_checksum(void *d, int c)
{
	u32 sum = 0xffff;
	u16 *p = d;

	while (c > 1) {
		sum += swap_u16(ntohs(*p++));
		c -= 2;
	}

	if (c)
		sum += swap_u16(*((u8 *)p) << 8);

	while (sum > 0xffff)
		sum = (sum & 0xffff) + (sum >> 16);

	return ~sum;
}

static inline u16 ones_add(u16 a, u16 b)
{
	u32 sum = (u16)~a + (u16)~b;

	return ~((sum & 0xffff) + (sum >> 16));
}

static u16 ip6_checksum(struct ip6 *ip6, u32 data_len, u8 proto)
{
	u32 sum = 0;
	u16 *p;
	int i;

	for (i = 0, p = ip6->src.s6_addr16; i < 16; ++i)
		sum += swap_u16(ntohs(*p++));
	sum += swap_u32(data_len) >> 16;
	sum += swap_u32(data_len) & 0xffff;
	sum += swap_u16(proto);

	while (sum > 0xffff)
		sum = (sum & 0xffff) + (sum >> 16);

	return ~sum;
}
static void host_send_icmp6(u8 tc, struct in6_addr *src,
		struct in6_addr *dest, struct icmp *icmp,
		u8 *data, int data_len, struct net_device *dev)
{
    	struct {
		struct ip6 ip6;
		struct icmp icmp;
	} __attribute__ ((__packed__)) header;
	struct sk_buff *skb;

	header.ip6.ver_tc_fl = htonl((0x6 << 28) | (tc << 20));
	header.ip6.payload_length = htons(sizeof(header.icmp) + data_len);
	header.ip6.next_header = 58;
	header.ip6.hop_limit = 64;
	header.ip6.src = *src;
	header.ip6.dest = *dest;
	header.icmp = *icmp;
	header.icmp.cksum = 0;
	header.icmp.cksum = htons(swap_u16(ones_add(ip_checksum(data, data_len),
			ip_checksum(&header.icmp, sizeof(header.icmp)))));
	header.icmp.cksum = htons(swap_u16(ones_add(swap_u16(ntohs(header.icmp.cksum)),
			ip6_checksum(&header.ip6, data_len + sizeof(header.icmp), 58))));

	skb = netdev_alloc_skb(dev, sizeof(header) + data_len);
	if (!skb) {
		dev->stats.rx_dropped++;
		return;
	}
	memcpy(skb_put(skb, sizeof(header)), &header, sizeof(header));
	memcpy(skb_put(skb, data_len), data, data_len);
	skb->protocol = htons(ETH_P_IPV6);
	
	dev->stats.rx_bytes += skb->len;
	dev->stats.rx_packets++;
	netif_rx(skb);
}
void handle_ip6(struct sk_buff *skb)
{
     struct ip6 *ip6;
     struct icmp *icmp;
     int len;
     u8 *data;
     len = skb->len;
     if (len < sizeof(struct icmp) + sizeof(struct ip6)) {
		pr_info("snull: Hmm... packet too short (%i octets)\n", len);
		return ;
      }
      if(ETH_P_IPV6 != ntohs(skb->protocol))
      {
		pr_info("not ip6 hdr \n");
		goto err1;
      }
      ip6 = (struct ip6 *)(skb->data);
      if(IPPROTO_ICMP6 != ip6->next_header)
      {
		pr_info("not icmp6 hdr \n");
		goto err1;
      }
      icmp = (struct icmp *)(ip6 + 1); 
      data =  skb->data + sizeof(struct ip6) + sizeof(struct icmp);
      len  = len - sizeof(struct ip6) - sizeof(struct icmp);
      switch (icmp->type) {
      case 128:
		icmp->type = 129;
		host_send_icmp6((ntohl(ip6->ver_tc_fl) >> 20) & 0xff,
				&ip6->dest, &ip6->src,
				icmp, data,len , skb->dev);
		break;
       default:
           dev_kfree_skb(skb);
           return; 
       }
      dev_kfree_skb(skb);
      return ;
err1:
      dev_kfree_skb(skb);
}
static int nat64_start_xmit(struct sk_buff *skb, struct net_device *dev)
{
	netif_stop_queue(dev);
#if 0
       	skb_orphan(skb);
	skb_dst_drop(skb);
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
	nf_reset(skb);
#else
	nf_reset_ct(skb);
#endif

	if (skb_linearize(skb) < 0)
		return NETDEV_TX_OK;
#endif
        if(ETH_P_IPV6 == ntohs(skb->protocol))
        {
             handle_ip6(skb);
        }
        else
        {
	     virt_rs_packet(skb,dev);
	     dev_kfree_skb(skb);
	     dev->stats.tx_packets++; 
	     dev->stats.tx_bytes+=skb->len; 
        }
	pr_info("tx_packets=%ld tx_bytes=%ld\n",dev->stats.tx_packets,dev->stats.tx_bytes);
	netif_wake_queue(dev); 
	return NETDEV_TX_OK;
}
static int nat64_start(struct net_device *dev)
{
	netif_tx_start_all_queues(dev);
	return 0;
}

static int nat64_stop(struct net_device *dev)
{
	netif_tx_stop_all_queues(dev);
	return 0;
}

static const struct net_device_ops nat64_netdev_ops = {
	.ndo_open		= nat64_start,
	.ndo_stop		= nat64_stop,
	.ndo_start_xmit	= nat64_start_xmit,
};

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

/* Handle of the NAT64 virtual interface */
static struct net_device *nat64_netdev;
struct proc_dir_entry *g_tayga_proc_dir;

int  __init nat64_module_init(void)
{
	struct net_device *dev;
	int err = -1;


	if (!(dev = alloc_netdev_mqs(sizeof(struct nat64_if_info), "nat64",
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3, 17, 0)
		NET_NAME_UNKNOWN,
#endif
		nat64_setup, 8, 8))) {
		printk(KERN_ERR "tayga: alloc_netdev() failed.\n");
		err = -ENOMEM;
		goto err3;
	}

	if ((err = register_netdev(dev)) < 0)
		goto err4;

	nat64_netdev = dev;
	netif_carrier_on(dev);

	return 0;

err4:
	free_netdev(dev);
err3:
	return err;
}

void __exit nat64_module_exit(void)
{
	unregister_netdev(nat64_netdev);
	free_netdev(nat64_netdev);
}

module_init(nat64_module_init);
module_exit(nat64_module_exit);

MODULE_DESCRIPTION("Kernel NAT64 Transition Module, ported from \"tayga\"");
MODULE_AUTHOR("Jianying Liu <rssnsj@gmail.com>");
MODULE_LICENSE("GPL");

