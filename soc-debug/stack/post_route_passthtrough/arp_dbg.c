
#include <linux/init.h>
#include <linux/module.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/if_arp.h>
#include <linux/skbuff.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/netfilter_arp.h>

#define LOG(fmt,arg...) printk("[%s %d] "fmt,__FUNCTION__,__LINE__,##arg)

/* arp内容 */
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

static unsigned int arp_input_hook(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
    struct ethhdr * ethh = NULL;
    /* 获取L2层首部 */
    ethh = eth_hdr(skb);
    if(ethh == NULL)
    {
        return NF_ACCEPT;    
    }    
    /* 打印网络层协议类型 */
    //LOG(" L3 type :%x \r\n",ethh->h_proto);
    return NF_ACCEPT;    
}
static unsigned int arp_output_hook(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
    struct arphdr *arph = NULL;    /* arp首部 */
    struct arp_info *arpinfo = NULL;
    arph = arp_hdr(skb);           /* 获取arp首部 */ 
    
    if(arph == NULL)
    {
        LOG("Weird! arp header null \r\n");
        return NF_ACCEPT;
    }
    /* 打印arp首部信息 */
    LOG(" arp info :\r\n"
        "-------------\r\n"    
        "arp hw type :%x \r\n"
        "arp pro type :%x \r\n"
        "arp hln :%d\r\n"
        "arp plen:%d\r\n"
        "arp ops :%d\r\n"
        "-------------\r\n",        
        ntohs(arph->ar_hrd),ntohs(arph->ar_pro),arph->ar_hln,arph->ar_pln,ntohs(arph->ar_op));
    /* 打印mac地址信息 */    
    arpinfo = (unsigned char *)(arph + 1);
    LOG("\n-------------\r\n"
        "mac : %x:%x:%x:%x:%x:%x \r\n"
        "sip : %d:%d:%d:%d \r\n"
        "dmac : %x:%x:%x:%x:%x:%x \r\n"
        "dip : %d:%d:%d:%d \r\n"
        "-------------\r\n",
        arpinfo->src[0],arpinfo->src[1],arpinfo->src[2],arpinfo->src[3],arpinfo->src[4],arpinfo->src[5],
        IP1(arpinfo->srcip),IP2(arpinfo->srcip),IP3(arpinfo->srcip),IP4(arpinfo->srcip),
        arpinfo->dst[0],arpinfo->dst[1],arpinfo->dst[2],arpinfo->dst[3],arpinfo->dst[4],arpinfo->dst[5],
        IP1(arpinfo->dstip),IP2(arpinfo->dstip),IP3(arpinfo->dstip),IP4(arpinfo->dstip));    
    return NF_ACCEPT;    
}


struct nf_hook_ops arp_in_ops =
    {
        .hook = arp_input_hook,    /* 输入arp钩子函数*/
        .pf = NFPROTO_ARP,         /* 协议类型 */   
        .hooknum = NF_ARP_IN,      /* arp input 链*/  
        .priority = 0,             /* 优先级 */   
    };
struct nf_hook_ops arp_out_ops =
    {
        .hook = arp_output_hook,   /* 输出arp钩子函数 */
        .pf = NFPROTO_ARP,         /* 协议类型 */
        .hooknum = NF_ARP_OUT,     /* arp output 链 */   
        .priority = 0,             /* 优先级 */
    };

static int __init arp_hook_init(void)
{
    nf_register_net_hook(&init_net, &arp_in_ops);
    nf_register_net_hook(&init_net, &arp_out_ops);
    return 0;
}

static void __exit arp_hook_exit(void)
{   
    nf_unregister_net_hook(&init_net, &arp_in_ops);
    nf_unregister_net_hook(&init_net, &arp_out_ops);
    return ;
}

module_init(arp_hook_init)
module_exit(arp_hook_exit)
MODULE_LICENSE("GPL");
 
