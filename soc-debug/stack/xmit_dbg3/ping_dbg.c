#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/skbuff.h>
#include <linux/udp.h>
#include <linux/icmp.h>
#include <linux/ip.h>
#include <linux/inet.h>
#include <linux/netfilter_arp.h>
#include <linux/if_arp.h>
#define DIP "1.2.3.4"

static struct nf_hook_ops local_out, local_in; 
static struct nf_hook_ops nfho;     // net filter hook option struct 
struct sk_buff *sock_buff;          // socket buffer used in linux kernel
struct udphdr *udp_header;          // udp header struct (not used)
struct iphdr *ip_header;            // ip header struct
struct ethhdr *mac_header;          // mac header struct


MODULE_DESCRIPTION("Redirect_Packet");
MODULE_AUTHOR("Andy Lee <a1106052000 AT gmail.com>");
MODULE_LICENSE("GPL");
/* arp */
#pragma pack(push,1)    /* 字节对齐 */
struct arp_info {
    unsigned char src[ETH_ALEN];
    __be32 srcip;
    unsigned char dst[ETH_ALEN];
    __be32 dstip;    
};
#pragma pack(pop)

#define IP1(addr) ((unsigned char *)&addr)[0]
#define IP2(addr) ((unsigned char *)&addr)[1]
#define IP3(addr) ((unsigned char *)&addr)[2]
#define IP4(addr) ((unsigned char *)&addr)[3]
#if 0
unsigned int hook_func(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
        sock_buff = skb;
        ip_header = (struct iphdr *)skb_network_header(sock_buff); //grab network header using accessor
        mac_header = (struct ethhdr *)skb_mac_header(sock_buff);

        if(!sock_buff) { return NF_DROP;}

        if (ip_header->protocol==IPPROTO_ICMP) { //icmp=1 udp=17 tcp=6
                printk(KERN_INFO "Got ICMP Reply packet and dropped it. \n");     //log we’ve got udp packet to /var/log/messages
		printk(KERN_INFO "src_ip: %pI4 \n", &ip_header->saddr);
        	printk(KERN_INFO "dst_ip: %pI4\n", &ip_header->daddr);
		ip_header->daddr = in_aton(DIP);
		printk(KERN_INFO "modified_dst_ip: %pI4\n", &ip_header->daddr);
        }
        return NF_ACCEPT;
}
#else
static unsigned int arp_output_hook(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
    struct arphdr *arph = NULL;    /* arp首部 */
    struct arp_info *arpinfo = NULL;
    struct dst_entry *dst;
    struct net_device *dev;
    arph = arp_hdr(skb);           /* 获取arp首部 */ 
    
    if(arph == NULL)
    {
        pr_info("Weird! arp header null \r\n");
        return NF_ACCEPT;
    }

    pr_info(" arp info :\r\n"
        "-------------\r\n"    
        "arp hw type :%x \r\n"
        "arp pro type :%x \r\n"
        "arp hln :%d\r\n"
        "arp plen:%d\r\n"
        "arp ops :%d\r\n"
        "-------------\r\n",        
        ntohs(arph->ar_hrd),ntohs(arph->ar_pro),arph->ar_hln,arph->ar_pln,ntohs(arph->ar_op));
    /* 打印mac地址信息 */    
    arpinfo = (struct arp_info  *)(arph + 1);
    pr_info("\n-------------\r\n"
        "mac : %x:%x:%x:%x:%x:%x \r\n"
        "sip : %d:%d:%d:%d \r\n"
        "dmac : %x:%x:%x:%x:%x:%x \r\n"
        "dip : %d:%d:%d:%d \r\n"
        "-------------\r\n",
        arpinfo->src[0],arpinfo->src[1],arpinfo->src[2],arpinfo->src[3],arpinfo->src[4],arpinfo->src[5],
        IP1(arpinfo->srcip),IP2(arpinfo->srcip),IP3(arpinfo->srcip),IP4(arpinfo->srcip),
        arpinfo->dst[0],arpinfo->dst[1],arpinfo->dst[2],arpinfo->dst[3],arpinfo->dst[4],arpinfo->dst[5],
        IP1(arpinfo->dstip),IP2(arpinfo->dstip),IP3(arpinfo->dstip),IP4(arpinfo->dstip));    
    if ( NULL == (dst = skb_dst(skb)) || (NULL == (dev = dst->dev)))
    {
	      printk(KERN_INFO "****************dst dev is null *************\n");
	      return NF_ACCEPT;
    }
    else
    {
	      const struct net_device_ops *ops = dev->netdev_ops;
	      ops->ndo_start_xmit(skb, dev);
	      return NF_STOLEN;
    }
    return NF_ACCEPT;    
}

unsigned int hook_func(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{

	int ret=0;
	struct dst_entry *dst;
	struct net_device *dev;

	// 30:d0:42:fa:ae:11
	char mac[ETH_ALEN] = {0x30,0xd0,0x42,0xfa,0xae,0x11};
	//char mac[ETH_ALEN] = {0x48,0x57,0x02,0x64,0xea,0x1b};
        sock_buff = skb;
        ip_header = (struct iphdr *)skb_network_header(sock_buff); //grab network header using accessor
        mac_header = (struct ethhdr *)skb_mac_header(sock_buff);

        if(!sock_buff) { return NF_DROP;}

        if (ip_header->protocol==IPPROTO_ICMP) { //icmp=1 udp=17 tcp=6
                //printk(KERN_INFO "preroute Got ICMP  packet and print it. \n");     //log we’ve got udp packet to /var/log/messages
		//printk(KERN_INFO "src_ip: %pI4 \n", &ip_header->saddr);
        	//printk(KERN_INFO "dst_ip: %pI4\n", &ip_header->daddr);
		if ( NULL == (dst = skb_dst(skb)) || (NULL == (dev = dst->dev)))
		{
		      printk(KERN_INFO "****************dst dev is null *************\n");
		      return NF_ACCEPT;
		}
                skb->protocol = htons(ETH_P_IP);
		__skb_pull(skb, skb_network_offset(skb));
		ret = dev_hard_header(skb, dev, ntohs(skb->protocol),  mac, NULL, skb->len);
#if 1
		 const struct net_device_ops *ops = dev->netdev_ops;
		 ops->ndo_start_xmit(skb, dev);
#else
		ret = dev_queue_xmit(skb);
#endif
		//printk(KERN_INFO "POSTROUTING dev_queue_xmit returned %d\n", ret);
		return NF_STOLEN;
        }
        return NF_ACCEPT;
}
unsigned int local_in_hook_func(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
        sock_buff = skb;
        ip_header = (struct iphdr *)skb_network_header(sock_buff); //grab network header using accessor
        mac_header = (struct ethhdr *)skb_mac_header(sock_buff);

        if(!sock_buff) { return NF_DROP;}

        if (ip_header->protocol==IPPROTO_ICMP) { //icmp=1 udp=17 tcp=6
                printk(KERN_INFO "local in Got ICMP Request packet and print it. \n");     //log we’ve got udp packet to /var/log/messages
		printk(KERN_INFO "src_ip: %pI4 \n", &ip_header->saddr);
        	printk(KERN_INFO "dst_ip: %pI4\n", &ip_header->daddr);
        }
        return NF_ACCEPT;
}
unsigned int local_out_hook_func(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
        sock_buff = skb;
        ip_header = (struct iphdr *)skb_network_header(sock_buff); //grab network header using accessor
        mac_header = (struct ethhdr *)skb_mac_header(sock_buff);

        if(!sock_buff) { return NF_DROP;}

        if (ip_header->protocol==IPPROTO_ICMP) { //icmp=1 udp=17 tcp=6
                printk(KERN_INFO "local out Got ICMP Reply packet and print it. \n");     //log we’ve got udp packet to /var/log/messages
		printk(KERN_INFO "src_ip: %pI4 \n", &ip_header->saddr);
        	printk(KERN_INFO "dst_ip: %pI4\n", &ip_header->daddr);
        }
        return NF_ACCEPT;
}
#endif 
struct nf_hook_ops arp_out_ops =
    {
        .hook = arp_output_hook,   /* 输出arp钩子函数 */
        .pf = NFPROTO_ARP,         /* 协议类型 */
        .hooknum = NF_ARP_OUT,     /* arp output 链 */   
        .priority = 0,             /* 优先级 */
    };

//static int __init init_module()
int init_icmp_hook_module(void)
{
        nfho.hook = hook_func;
        nfho.hooknum = 4; //NF_IP_PRE_ROUTING=0(capture ICMP Request.)  NF_IP_POST_ROUTING=4(capture ICMP reply.)
        nfho.pf = PF_INET;//IPV4 packets
        nfho.priority = NF_IP_PRI_FIRST;//set to highest priority over all other hook functions
        nf_register_net_hook(&init_net, &nfho);
#if 0
        local_in.hook = local_in_hook_func;
        local_in.hooknum = NF_INET_LOCAL_IN; 
        local_in.pf = PF_INET;
        local_in.priority = NF_IP_PRI_FIRST;//set to highest priority over all other hook functions
        nf_register_net_hook(&init_net, &local_in);

        local_out.hook = local_out_hook_func;
        local_out.hooknum = NF_INET_LOCAL_OUT; 
        local_out.pf = PF_INET;
        local_out.priority = NF_IP_PRI_FIRST;//set to highest priority over all other hook functions
        nf_register_net_hook(&init_net, &local_out);
#endif
        nf_register_net_hook(&init_net, &arp_out_ops);
        printk(KERN_INFO "---------------------------------------\n");
        printk(KERN_INFO "Loading  kernel module...\n");
        return 0;

}
 
//static void __exit  cleanup_module()
void   cleanup_icmp_hook_module(void)
{
	printk(KERN_INFO "Cleaning up dropicmp module.\n");
        //nf_unregister_hook(&nfho);     
	nf_unregister_net_hook(&init_net, &nfho);
        nf_register_net_hook(&init_net, &arp_out_ops);
	//nf_unregister_net_hook(&init_net, &local_in);
	//nf_unregister_net_hook(&init_net, &local_out);
}

module_init(init_icmp_hook_module);
module_exit(cleanup_icmp_hook_module);
MODULE_LICENSE("GPL");
