#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/skbuff.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/netdevice.h>

MODULE_LICENSE("GPL");

/* This is the structure we shall use to register our function */
static struct nf_hook_ops nfho;

/*
 * This module is to drop tcp packet whose dest port is 23
 * telnet uses port 23
 */
static __be16 deny_port = htons(23);   /* port 23 */


/* This is the hook function itself */
unsigned int hook_func(unsigned int hooknum,
		struct sk_buff *skb,
		const struct net_device *in,
		const struct net_device *out,
		int (*okfn)(struct sk_buff *))
{
	struct sk_buff *sb = skb;
	struct iphdr *iph;
	struct tcphdr *tcph;

	if(!sb) 
		return NF_ACCEPT;
	
	iph = ip_hdr(sb);
	if(!iph) return NF_ACCEPT;

	/*Make sure this is a TCP packet first*/
	if(iph->protocol != IPPROTO_TCP) {
		return NF_ACCEPT;
	}
	tcph = (struct tcphdr *)(sb->data + iph->ihl * 4);
	//tcph = tcp_hdr(sb);
	//pr_warning("%d.%d.%d.%d:%u/t%d.%d.%d.%d:%u/n",NIPQUAD(iph->saddr),ntohs(tcph->source),NIPQUAD(iph->daddr),ntohs(tcph->dest));
	//if(tcph->dest == *(__be16 *)deny_port) {
	if(tcph->dest == deny_port) {
		pr_warning("Dropped packet to port %d/n", ntohs(tcph->dest));
		return NF_DROP;
	}

	return NF_ACCEPT;

}

/* Initialisation routine */
int init_module()
{
	/* Fill in our hook structure */
	nfho.hook     = hook_func;         /* Handler function */
	nfho.hooknum  = NF_INET_PRE_ROUTING; /* First hook for IPv4 */
	nfho.pf       = PF_INET;
	nfho.priority = NF_IP_PRI_FIRST;   /* Make our function first */

	nf_register_hook(&nfho);

	pr_info("filterPort install into kernel!/n");
	return 0;
}

/* Cleanup routine */
void cleanup_module()
{
	nf_unregister_hook(&nfho);
	pr_info("filterPort removed from kernel!/n");
}
