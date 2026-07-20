#include <linux/kmod.h>
#include <linux/module.h>
#include <linux/version.h>
#include <linux/skbuff.h>
#include <net/ipv6.h>
#include <net/ip6_checksum.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/netfilter_ipv6.h>
#include <net/netfilter/nf_conntrack.h>
#define DRV_VERSION "1.0.0"
#define DRV_DESC    "ping6 debug"
/* refer to static int icmpv6_rcv(struct sk_buff *skb)
 * void ip_vs_nat_icmp_v6(struct sk_buff *skb, struct ip_vs_protocol *pp
 * frag_mt6(const struct sk_buff *skb, struct xt_action_param *par)
   */
static unsigned int icmp6_test_nf_hook(void *priv,
        struct sk_buff *skb,
        const struct nf_hook_state *state)
{
    //struct net_device *dev = skb->dev;
    //struct inet6_dev *idev = __in6_dev_get(dev);
    unsigned int ptr;
    const struct frag_hdr *fh;
    struct ipv6hdr *iph      = ipv6_hdr(skb);
#if 1
    unsigned int icmp_offset = 0;
    unsigned int offs        = 0; /* header offset*/
    //int protocol;
    struct icmp6hdr *icmph6;
    unsigned short fragoffs;
    unsigned int dataoff,datalen;
    const char *dptr;

#endif
    u8 nexthdr;
    int err;
    const struct in6_addr *saddr, *daddr;
    u_int16_t payload_len = ntohs(iph->payload_len);
    u_int16_t icmp_data_len = 0, i = 0,j=0;
    //char * data = NULL;
    saddr = &iph->saddr;
    daddr = &iph->daddr;
    nexthdr = iph->nexthdr;
    //ipv6_skip_exthdr(skb, skb_network_offset(skb) + sizeof(_ip6h), &nexthdr, &frag_off);
    pr_info(" saddr=%pI6c daddr=%pI6c proto=%hhu", saddr, daddr, nexthdr);
    if(NEXTHDR_FRAGMENT == nexthdr){
	  //goto find_frag;
	  fh = (struct frag_hdr *)skb_transport_header(skb);
          pr_info("INFO %04X ", fh->frag_off);
          pr_info("OFFSET %04X ", ntohs(fh->frag_off) & ~0x7);
          pr_info("RES %02X %04X", fh->reserved, ntohs(fh->frag_off) & 0x6);
          pr_info("MF %04X ", fh->frag_off & htons(IP6_MF));
          pr_info("ID %u %08X\n", ntohl(fh->identification), ntohl(fh->identification));
    }
#if 1
    err = ipv6_find_hdr(skb, &icmp_offset, IPPROTO_ICMPV6, &fragoffs, NULL);
    if (err < 0) {
	  goto find_frag;
    }
    icmph6 = (struct icmp6hdr *)(skb_network_header(skb) + icmp_offset);
    if (icmph6->icmp6_type == ICMPV6_ECHO_REQUEST){
         pr_info("ipv6 find icmp6 echo request hdr,cksum %u, cksum to host oreder: %u, id %u,seq %u \n", icmph6->icmp6_cksum,ntohs(icmph6->icmp6_cksum), ntohs(icmph6->icmp6_identifier)\
			 ,ntohs(icmph6->icmp6_sequence));
        icmp_data_len = payload_len - sizeof(struct icmp6hdr);
#if 1
	dataoff = icmp_offset + sizeof(struct icmp6hdr);
	skb_linearize(skb);
	dptr = skb->data + dataoff;
	datalen = skb->len - dataoff;
#endif
#if 0
        data =  (char*)(icmph6 +1);
	j = 0;
        for(i=0; i < icmp_data_len; ++i)
        {
             if('a' == data[i]){
                 ++ j;
             }
        }

        pr_info("ipv6  icmp6 echo request data, number of a: %u, expected number of a: %u \n",j, icmp_data_len);
#endif
	j = 0;
        for(i=0; i < datalen; ++i)
        {
             if('a' == dptr[i]){
                 ++ j;
             }
        }
        pr_info("ipv6  icmp6 echo request data, number of a: %u, expected number of a: %u \n",j, datalen);
        if (skb_checksum_validate(skb, IPPROTO_ICMPV6, ip6_compute_pseudo)) {
            pr_info("ICMPv6 checksum failed [%pI6c > %pI6c]\n", saddr, daddr);
            			                  goto out;
        }
    } 
    return NF_ACCEPT;
#endif
find_frag:
#if 1
    err = ipv6_find_hdr(skb, &ptr, NEXTHDR_FRAGMENT, &fragoffs,NULL);
    if (err < 0) {
        goto out;
    }
    pr_info("ipv6 find frag hdr \n");
#if 0
    fh = skb_header_pointer(skb, ptr, sizeof(_frag), &_frag); 
    if (fh == NULL) {
        return NF_ACCEPT;
    }
    pr_debug("INFO %04X ", fh->frag_off);
    pr_debug("OFFSET %04X ", ntohs(fh->frag_off) & ~0x7);
    pr_debug("RES %02X %04X", fh->reserved, ntohs(fh->frag_off) & 0x6);
    pr_debug("MF %04X ", fh->frag_off & htons(IP6_MF));
    pr_debug("ID %u %08X\n", ntohl(fh->identification), ntohl(fh->identification));
#endif
 #endif
out:
    return NF_ACCEPT;

}


static struct nf_hook_ops icmp6_test_nf_hook_ops[] =
{
    {
        .hook = icmp6_test_nf_hook,
        .pf = NFPROTO_IPV6,
        .hooknum = NF_INET_LOCAL_IN,
        //.hooknum = NF_INET_PRE_ROUTING,   
        .priority = NF_IP_PRI_LAST,            // ---优先级
        //.priority = NF_IP_PRI_FIRST,            // ---优先级
    },

};

/**
 * module init
 */
static int __init icmp6_test_init(void)
{
    int ret = 0;

    //need_conntrack();

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4,13,0)
    ret = nf_register_net_hooks(&init_net, icmp6_test_nf_hook_ops, ARRAY_SIZE(icmp6_test_nf_hook_ops));
#else
    ret = nf_register_hooks(icmp6_test_nf_hook_ops, ARRAY_SIZE(icmp6_test_nf_hook_ops));
#endif
    if (ret != 0) {
        printk("nf_register_hook failed: %d\n", ret);
        return ret;
    }
    printk("icmp6 test init OK\n");

    return 0;
}

/**
 * module uninit
 */
static void __exit icmp6_test_fini(void)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4,13,0)
    nf_unregister_net_hooks(&init_net, icmp6_test_nf_hook_ops, ARRAY_SIZE(icmp6_test_nf_hook_ops));
#else
    nf_unregister_hooks(icmp6_test_nf_hook_ops, ARRAY_SIZE(icmp6_test_nf_hook_ops));
#endif

    printk("icmp6 test exit OK\n");
}

module_init(icmp6_test_init);
module_exit(icmp6_test_fini);

MODULE_DESCRIPTION(DRV_DESC);
MODULE_VERSION(DRV_VERSION);
MODULE_AUTHOR("isshe <i.sshe@foxmail.com>");
MODULE_LICENSE("GPL v2");
