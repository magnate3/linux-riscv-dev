#include <linux/module.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h>


#include <asm/io.h>
#include <asm/irq.h>

static struct net_device * test_ndev;
static int ntest_open(struct net_device *dev)
{
    printk("ntest_open\n");
    return 0;
}
static int ntest_stop(struct net_device *dev)
{
    printk("ntest_stop\n");
    return 0;
}

static void emulator_rx_data(struct sk_buff *skb,
						   struct net_device *dev)
{
    unsigned char *type;
    struct iphdr *ih;
    __be32 *saddr, *daddr,tmp;
    unsigned char tmp_dev_addr[ETH_ALEN];
    struct ethhdr *ethhdr;
    struct sk_buff *rx_skb;
    
//    printk("emulator_rx_data\n");
    ethhdr =(struct ethhdr*)skb->data;
    memcpy(tmp_dev_addr,ethhdr->h_dest,ETH_ALEN);
    memcpy(ethhdr->h_dest,ethhdr->h_source,ETH_ALEN);
    memcpy(ethhdr->h_source,tmp_dev_addr,ETH_ALEN);
//    printk("memcpy(h_source\n");

    ih=(struct iphdr*)(skb->data+sizeof(struct ethhdr));
    saddr =&(ih->saddr);
    daddr =&(ih->daddr);

    tmp=*saddr;
    *saddr=*daddr;
    *saddr=tmp;

    type = skb->data + sizeof(ethhdr)+sizeof(struct iphdr);
    *type=0;
    ih->check=0;
//    printk("ip_fast_csum\n");
//    ih->check=ip_fast_csum((unsigned char*)ih,ih->ihl);
//    printk("dev_alloc_skb\n");
    rx_skb=dev_alloc_skb(skb->len+2);
//    printk("skb_reserve\n");
    skb_reserve(skb,2);
//    printk("skb_put\n");
    memcpy(skb_put(rx_skb,skb->len),skb->data,skb->len);    
    rx_skb->dev=dev;
    rx_skb->protocol=eth_type_trans(rx_skb,dev);
    rx_skb->ip_summed=CHECKSUM_UNNECESSARY;
    dev->stats.rx_packets++;
    dev->stats.rx_bytes += skb->len;
//    printk("netif_rx(rx_skb);\n");
    netif_rx(rx_skb);
}
void print_skb(struct sk_buff* skb)
{
    if (skb_is_nonlinear(skb)) {
        printk("is nonlinear");
    } else {
         printk("is linear");
    }
    printk("sk_buff: len:%d  skb->data_len:%d  truesize:%d head:%0X  data:%0X tail:%d end:%d"
    ,skb->len,skb->data_len,skb->truesize,(skb->head),(skb->data),(skb->tail),(skb->end));
   struct skb_shared_info *sp = skb_shinfo(skb);
   int i;
   u32 byte_count = 0;
   for (i = 0; i < sp->nr_frags; i++) 
   {
      skb_frag_t *frag = &sp->frags[i];
      byte_count = skb_frag_size(frag);
      pr_err(" fp->size %u ", be32_to_cpu(byte_count));
   }
}
void skb_test(struct sk_buff* skb, struct net_device *dev)
{
      
   printk("***********************  skb test begin \n ");
   printk("sizeof(ethhdr): %d, sizeof(struct iphdr)； %d \n",  sizeof(struct ethhdr), sizeof(struct iphdr));
   print_skb(skb);
   int headerlen = skb_headroom(skb);
   unsigned int size = skb_end_offset(skb) + skb->data_len;
   // ‘sk_buff_data_t’ and ‘unsigned char *’
   printk("***********************  sk_buff: len:%d, size %d,  head - end %d \n", skb->len, size,  skb->end);
   struct sk_buff *skb2 = __alloc_skb(size, GFP_ATOMIC,SKB_ALLOC_RX /* skb_alloc_rx_flag(skb)*/, NUMA_NO_NODE);
  print_skb(skb2);
  /* Set the data pointer */
  skb_reserve(skb2, headerlen);
  /* Set the tail pointer and length */
  skb_put(skb2, skb->len);
  print_skb(skb2);
  dev_kfree_skb_any(skb2);
   printk("***********************  skb test netdev_alloc_skb\n ");
   // netdev_alloc_skb  also call   skb = __alloc_skb(size, gfp_mask, SKB_ALLOC_RX, NUMA_NO_NODE);
   struct sk_buff *skb3 = netdev_alloc_skb(dev, size);
   print_skb(skb3);
   dev_kfree_skb_any(skb3);
   printk("***********************  skb test  sizeof(struct ethhdr) + sizeof(struct iphdr) + 64\n ");
   struct sk_buff *skb4 = __alloc_skb( sizeof(struct ethhdr) + sizeof(struct iphdr) + 64, GFP_ATOMIC,SKB_ALLOC_RX /* skb_alloc_rx_flag(skb)*/, NUMA_NO_NODE);
  print_skb(skb4);
  dev_kfree_skb_any(skb4);
   printk("***********************  skb test end\n ");
}
static netdev_tx_t	ntest_start_xmit(struct sk_buff *skb,
						   struct net_device *dev)
{
    static int count;
    printk("ntest_start_xmit %d\n",count++);
    //printk("ntest_start_xmit skb->len:   %d  skb->data_len  %d \n", skb->len, skb->data_len);
    skb_test(skb, dev); 
    netif_stop_queue(dev);
    dev->stats.tx_packets++;
    dev->stats.tx_bytes += skb->len;
    emulator_rx_data(skb,dev);
    dev_kfree_skb(skb);
    netif_wake_queue(dev);
    return NETDEV_TX_OK;
}

static void ntest_timeout(struct net_device *dev)
{
    printk("ntest_timeout\n");
}
static int ntest_set_mac_address(struct net_device *dev,
                                  void *addr_p)
{
	struct sockaddr *addr = addr_p;
    printk("ntest_set_mac_address"); 
	if (netif_running(dev))
		return -EBUSY;
	if (!is_valid_ether_addr(addr->sa_data))
		return -EADDRNOTAVAIL;
	memcpy(dev->dev_addr, addr->sa_data, ETH_ALEN);
	return 0;
}

static const struct net_device_ops test_netdev_ops = {
	.ndo_open		= ntest_open,
	.ndo_stop		= ntest_stop,
	.ndo_start_xmit		= ntest_start_xmit,
	.ndo_tx_timeout		= ntest_timeout,
//	.ndo_set_multicast_list	= test_hash_table,
//	.ndo_do_ioctl		= test_ioctl,
	.ndo_change_mtu		= eth_change_mtu,
//	.ndo_set_features	= test_set_features,
	.ndo_validate_addr	= eth_validate_addr,
	.ndo_set_mac_address	= ntest_set_mac_address,
//#ifdef CONFIG_NET_POLL_CONTROLLER
#if 0
	.ndo_poll_controller	= dm9000_poll_controller,
#endif
};


static int virt_net_init(void)
{
    printk("virt_net_init\n"); 
    test_ndev=alloc_etherdev(0);    
	test_ndev->netdev_ops	= &test_netdev_ops;
    test_ndev->dev_addr[0]=0x00;
    test_ndev->dev_addr[1]=0x00;
    test_ndev->dev_addr[2]=0xaa;
    test_ndev->dev_addr[3]=0xbb;
    test_ndev->dev_addr[4]=0xcc;
    test_ndev->dev_addr[5]=0xdd;    
    
    test_ndev->flags      |= IFF_NOARP;
    test_ndev->features |= NETIF_F_HW_CSUM;
    //test_ndev->features   |= NETIF_F_NO_CSUM;
    
    register_netdev(test_ndev);
    return 0;
}
static void virt_net_exit(void)
{
    unregister_netdev(test_ndev);
    free_netdev(test_ndev);
}
module_init (virt_net_init);
module_exit (virt_net_exit);
MODULE_LICENSE ("GPL");
MODULE_AUTHOR ("king");
MODULE_DESCRIPTION ("Observing net");
