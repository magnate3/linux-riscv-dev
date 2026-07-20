/*
 * =====================================================================================
 *
 *       Filename:  skb_linearize.c
 *
 *    Description:  测试skb_linearize是如何作用的
 *
 *        Version:  1.0
 *        Created:  2018年08月29日 21时40分02秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  NYB (), niuyabeng@126.com
 *   Organization:  
 *
 * =====================================================================================
 */


#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/module.h>
#include <linux/moduleparam.h>

#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/skbuff.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>

#include <net/tcp.h>
#include <net/udp.h>
#include <net/sock.h>
#include <net/netfilter/nf_nat.h>
#include <net/netfilter/nf_nat_helper.h>
#include <net/netfilter/nf_conntrack.h>
#include <net/netfilter/nf_conntrack_core.h>
#include <net/netfilter/nf_conntrack_helper.h>
#include <net/netfilter/nf_conntrack_expect.h>
#include <linux/netfilter/nf_conntrack_common.h>

void print_skb(struct sk_buff* skb)
{
    int index =0;
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
      pr_err("fragment fp->size %u , %u", be32_to_cpu(byte_count), byte_count);
   }
   struct sk_buff *list;
   index = 0;
   for (list = skb_shinfo(skb)->frag_list; list; list = list->next)
   {
      pr_err("frag list %d", index++);
   }
}

unsigned int test_hookfn(unsigned int hooknum,
        struct sk_buff *skb, 
        const struct net_device *in, 
        const struct net_device *out,
        int (*okfn)(struct sk_buff *))
{
#if 0
    /* 如果syn置位，count++ */
    struct iphdr *iph = ip_hdr(skb);
    if(iph->protocol == IPPROTO_TCP) {
        struct tcphdr *tcph = tcp_hdr(skb);
        if(ntohs(tcph->dest) == 1995 && tcph->syn) {
            count++;
            printk("syn count to %u is %lu\n", ntohs(tcph->dest), count);
        }
    }
#endif
    //printk("==== in log_tcpsyn_count_hookfn =====\n");
    char rpl_c[9] = "niuyaben";
    char rpl_s[10] = "huan_huan";

    enum ip_conntrack_info ctinfo;
    struct nf_conn *ct = nf_ct_get(skb, &ctinfo);
    if(ct) {
        //printk("nf_ct_get: ok, ctinfo = %d\n", ctinfo);
    }
    else {
        //printk("nf_ct_get: NULL\n");
        return NF_ACCEPT;
    }
#if 1
   struct iphdr *iph = ip_hdr(skb);
   if(iph->protocol == IPPROTO_ICMP) {
   printk("****************** skb_linearize test begin *********************\n");
       print_skb(skb);
       if (skb_is_nonlinear(skb)) {
        printk("ping is nonlinear");
        } else {
         printk("ping is linear");
        } 
        struct sk_buff *skb2 = skb_copy(skb, GFP_ATOMIC);
        printk("after skb_copy , print skb2\n");
        print_skb(skb2);
        //kfree_skb(skb2);
        if(0 != skb_linearize(skb)) {
         printk("Canot skb_linearize\n");
        }
        else {
         printk("after skb_linearize, print skb\n");
         print_skb(skb);
           }
        struct iphdr *iph = ip_hdr(skb);
        if ( (iph->frag_off & htons(IP_MF)) || (iph->frag_off & IP_MF) )
        {
            printk("****************** have more fragment *********************\n");
        }
        printk("****************** skb_linearize test end *********************\n");
        kfree_skb(skb2);
    }
#endif
#if 0
    if(0 != skb_linearize(skb)) {
        printk("Canot skb_linearize\n");
        //return NF_ACCEPT;
    }
    else {
        //iph = ip_hdr(skb);
        //tcph = (struct tcphdr *)(skb_network_header(skb) + ip_hdrlen(skb))
        struct iphdr *iph = ip_hdr(skb);
        print_skb(skb);
        if(iph->protocol == IPPROTO_TCP) {
            struct tcphdr *tcph = tcp_hdr(skb);
            unsigned char *tcp_payload = skb->data + iph->ihl*4 + tcph->doff*4;
            int len_tcp_payload = ntohs(iph->tot_len) - iph->ihl*4 - tcph->doff*4;
            if(len_tcp_payload > 0) {
#if 0
                printk("len_tcp_payload = %d, tcp_payload = %.*s\n", 
                        len_tcp_payload, len_tcp_payload, tcp_payload);
                if(ntohs(tcph->dest) == 8080) {
                    nf_nat_mangle_tcp_packet(skb, ct, ctinfo, ip_hdrlen(skb), 
                            0, 1, rpl_c, 8);
                }
                else if(ntohs(tcph->source) == 8080) {
                    nf_nat_mangle_tcp_packet(skb, ct, ctinfo, ip_hdrlen(skb), 
                            0, 1, rpl_s, 9);
                }
#endif
            }
        }
    }
#endif

    return NF_ACCEPT;
}


static struct nf_hook_ops test_hookops = {
    .pf = NFPROTO_IPV4,
    .priority = NF_IP_PRI_MANGLE,
    //.hooknum = NF_INET_PRE_ROUTING,
    .hooknum = NF_INET_LOCAL_IN,
    .hook = test_hookfn,
#if LINUX_VERSION_CODE < KERNEL_VERSION(4,4,0)
    .owner = THIS_MODULE,
#endif
};

static __init int test_init(void)
{
    /*
     * register钩子函数
     */
    printk("===== test_init =====\n");
    return nf_register_net_hook(&init_net,&test_hookops);
}

static __exit void test_exit(void)
{
    /*
     * un-register钩子函数
     */
    printk("===== test_exit =====\n");
    nf_unregister_net_hook(&init_net,&test_hookops);
}


MODULE_LICENSE("GPL");
module_init(test_init);
module_exit(test_exit);
