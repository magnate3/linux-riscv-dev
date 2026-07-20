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
	virt_rs_packet(skb,dev);
	dev_kfree_skb(skb);
	dev->stats.tx_packets++; 
	dev->stats.tx_bytes+=skb->len; 
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

