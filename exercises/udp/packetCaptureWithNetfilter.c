/***********************************************************************
* File: packetCaptureWithNetfilter.c
* Copy Right: http://www.cnblogs.com/piky/articles/1587767.html
* Author: Unknow
* Create Date: Unknow
* Abstract Description:
*             To catch the packet from the network with netfilter.
*
*------------------------Revision History------------------------
* No.    Date        Revised By   Description
* 1      2011/7/28   Sam          +print the payload.(unsuccessful)
* 2      2011/7/30   Sam          +get the packet from the specific ip.
* 3      2011/8/10   Sam          correct the checksum and
*                                 +print the payload.
***********************************************************************/





#ifndef __KERNEL__
    #define __KERNEL__
#endif
#ifndef MODULE
    #define MODULE
#endif


#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/netfilter.h>
#include <linux/skbuff.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/netdevice.h>
#include <linux/if_arp.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <net/tcp.h>
#include <net/udp.h>
#include <linux/netfilter_ipv4.h>

#define SIP     "10.66.16.141"

#define NIPQUAD(addr) \
        ((unsigned char *)&addr)[0], \
        ((unsigned char *)&addr)[1], \
        ((unsigned char *)&addr)[2], \
        ((unsigned char *)&addr)[3]

#define HIPQUAD(addr) \
        ((unsigned char *)&addr)[3], \
        ((unsigned char *)&addr)[2], \
        ((unsigned char *)&addr)[1], \
        ((unsigned char *)&addr)[0]
static struct nf_hook_ops nfho;

unsigned int hook_func(unsigned int hooknum,
                       struct sk_buff *skb,
                       const struct net_device *in,
                       const struct net_device *out,
                       int (*okfn)(struct sk_buff *))
{
        if(skb) {
                struct sk_buff *sb = skb;
                struct tcphdr *tcph = NULL;
                struct udphdr *udph = NULL;
                struct iphdr *iph = NULL;
                u8 *payload;    // The pointer for the tcp payload.
                char sourceAddr[20];
                char myAddr[20];

                iph = ip_hdr(sb);
                if(iph) {
                        /* NIPQUAD() was defined in the linux/kernel.h.
                         * Display an IP address in readable format.*/            
                        /* These two sprintf are used to get the packet
                         * from the specific ip.*/
                        sprintf(myAddr, SIP);
                        sprintf(sourceAddr, "%u.%u.%u.%u", NIPQUAD(iph->saddr));

                        if (!(strcmp(sourceAddr, myAddr))) {
                                printk("IP:[%u.%u.%u.%u]-->[%u.%u.%u.%u];\n",
                                        NIPQUAD(iph->saddr), NIPQUAD(iph->daddr));
                                printk("IP (version %u, ihl %u, tos 0x%x, ttl %u, id %u, length %u, ",
                                        iph->version, iph->ihl, iph->tos, iph->ttl,
                                        ntohs(iph->id), ntohs(iph->tot_len));
                                /* 此处读取udp或tcp报头时, 不能用udp_hdr或tcp_hdr.
                                 * 因为对于skbuff结构中的指向各层协议头部的指针, 只
                                 * 有当到达该层时才对他们赋值.而netfilter处于网络层.
                                 * 所以不能直接访问skbuff中的传输层协议头指针，而必
                                 * 须用skb->data+iph->ihl*4来得到指向传输层头部的指
                                 * 针。 */
                                switch (iph->protocol) {
                                case IPPROTO_UDP:
                                        /*get the udp information*/
                                        udph = (struct udphdr *)(sb->data + iph->ihl*4);
                                        printk("UDP: [%u]-->[%u];\n", ntohs(udph->source), ntohs(udph->dest));    
                                        payload = (char *)udph + (char)sizeof(struct udphdr);
                                        /* 此处不能用"printk("payload: %20s\n", payload);"
                                         * 否则会出现乱码并且"hello world"打印不出来.
                                         * 不过用下面的方法打印出来的是"hello world"
                                         * 前面有一些乱码. */
                                        printk("\n%s\n", payload);
                                        break;
                                case IPPROTO_TCP:
                                        /*get the tcp header*/
                                        tcph = sb->data + iph->ihl*4;
                                        //payload = (char *)((__u32 *)tcph+tcph->doff);
                                        printk("TCP: [%u]-->[%u];\n", ntohs(tcph->source), ntohs(tcph->dest));
                                        break;
                                default:
                                        printk("unkown protocol!\n");
                                        break;
                                }
                        }
                } else
                        printk("iph is null\n");
        } else
                printk("skb is null,hooknum:%d\n", hooknum);

        return NF_ACCEPT;         
}

int init_module()
{
        nfho.hook = hook_func;        
        nfho.hooknum  = NF_INET_PRE_ROUTING;
        nfho.pf = PF_INET;
        nfho.priority = NF_IP_PRI_FIRST;

        nf_register_net_hook(&init_net,&nfho);

        printk("init module----------------ok\n");

        return 0;
}

void cleanup_module()
{
        nf_unregister_net_hook(&init_net,&nfho);
        printk("exit module----------------ok\n");
}

MODULE_LICENSE("GPL");
