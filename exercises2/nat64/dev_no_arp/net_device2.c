// https://blog.csdn.net/baidu_19348579/article/details/126283743

#include <linux/module.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h>
#include <linux/skbuff.h>
#include <linux/ip.h>
#include <linux/netdev_features.h>
#include <linux/netdevice.h>
#include <linux/if.h>
#include <linux/if_ether.h>
#include <linux/if_arp.h>
#include <linux/version.h>
#include <linux/poll.h>
#include <net/dst.h>
 
static struct net_device *virt_net;

#define TEST_NO_L2 1
   
static void virt_rs_packet(struct sk_buff *skb, struct net_device *dev)
{
	unsigned char *type;
	struct iphdr *ih;
	__be32 *saddr, *daddr, tmp;
	unsigned char tmp_dev_addr[ETH_ALEN];
	struct sk_buff *rx_skb;
	int ret;
        int len;
	struct ethhdr *ethhdr;
	len = skb->len;
	if (len < sizeof(struct ethhdr) + sizeof(struct iphdr)) {
		pr_info("snull: Hmm... packet too short (%i octets)\n", len);
		return ;
	}
	//对调ethhdr结构体 "源/目的"MAC地址*/
	ethhdr = (struct ethhdr *)skb->data;
        if(0x0800 != ntohs(ethhdr->h_proto)){
		pr_info("not ip hdr \n");
		return ;
	}
	memcpy(tmp_dev_addr, ethhdr->h_dest, ETH_ALEN);
	memcpy(ethhdr->h_dest, ethhdr->h_source, ETH_ALEN);
	memcpy(ethhdr->h_source, tmp_dev_addr, ETH_ALEN);
	//对调iphdr结构体"源/目的" IP地址
	ih = (struct iphdr *)(skb->data + sizeof(struct ethhdr));
        if(IPPROTO_ICMP != ih->protocol)
        {
		pr_info("not icmp hdr %x \n", ntohs(ih->protocol));
		return ;
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
	rx_skb->protocol = eth_type_trans(rx_skb, dev);
        rx_skb->protocol = htons(ETH_P_IP);
	ret=netif_rx(rx_skb);
	
	dev->stats.rx_packets++;        
	dev->stats.rx_bytes += skb->len;
	pr_info("rx_packets=%ld rx_bytes=%ld ret=%d\n",dev->stats.rx_packets,dev->stats.rx_bytes,ret);
}
 
static int virt_send_packet(struct sk_buff *skb, struct net_device *dev)
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
 
static int set_mac_address(struct net_device *dev,void *p)
{
	struct sockaddr *addr = p;
	pr_info("set_mac_address\n");
	if (netif_running(dev))
	{
		return -EBUSY;
	}
	memcpy(dev->dev_addr, addr->sa_data, dev->addr_len);
	return 0;
}
//void virt_tx_timeout(struct net_device *net,unsigned int txqueue)
void virt_tx_timeout(struct net_device *net)
{
	pr_info("virt_tx_timeout\n");
}
 
static int virt_start(struct net_device *dev)
{
	netif_tx_start_all_queues(dev);
	return 0;
}

static int virt_stop(struct net_device *dev)
{
	netif_tx_stop_all_queues(dev);
	return 0;
}
#if TEST_NO_L2
static const struct net_device_ops net_ops = {
	.ndo_open		= virt_start,
	.ndo_stop		= virt_stop,
	.ndo_set_mac_address    =set_mac_address,
	.ndo_start_xmit	= virt_send_packet,
};
#else
static const struct net_device_ops net_ops =
{
	.ndo_start_xmit		= virt_send_packet,
	.ndo_set_mac_address    =set_mac_address,
	.ndo_tx_timeout		= virt_tx_timeout,
};
#endif 
static int virt_net_init(void){
	virt_net= alloc_netdev(sizeof(struct net_device), "virt_net", NET_NAME_UNKNOWN,ether_setup);
#if 1 
	virt_net->netdev_ops= &net_ops;
	virt_net->flags = IFF_NOARP;
	virt_net->dev_addr[0] = 0x88;
	virt_net->dev_addr[1] = 0x88;
	virt_net->dev_addr[2] = 0x88;
	virt_net->dev_addr[3] = 0x88;
	virt_net->dev_addr[4] = 0x88;
	virt_net->dev_addr[5] = 0x88;
#else
	virt_net->dev_addr[0] = 0x88;
	virt_net->dev_addr[1] = 0x88;
	virt_net->dev_addr[2] = 0x88;
	virt_net->dev_addr[3] = 0x88;
	virt_net->dev_addr[4] = 0x88;
	virt_net->dev_addr[5] = 0x88;
	virt_net->netdev_ops= &net_ops;
        virt_net->hard_header_len = 0;
        virt_net->addr_len = ETH_ALEN;
        /* Zero header length */
        virt_net->type = ARPHRD_NONE;
        virt_net->flags = IFF_POINTOPOINT | IFF_NOARP | IFF_MULTICAST;
        virt_net->needed_headroom = ETH_HLEN;
#endif 
	register_netdev(virt_net);
	return 0;
}
 
static void virt_net_exit(void)
{
    unregister_netdev(virt_net);
    free_netdev(virt_net);   
}
 
module_init(virt_net_init);
module_exit(virt_net_exit);
 
MODULE_LICENSE("GPL");


