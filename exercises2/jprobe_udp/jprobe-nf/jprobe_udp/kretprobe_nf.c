/*
 *
 * kretprobe to trace netfilter skb processing
 * The objective is to find where a packet gets dropped
 *
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>

#include <linux/skbuff.h>
#include <linux/inet.h>
#include <net/ip.h>
#include <linux/netfilter/x_tables.h>
#include <linux/netfilter_ipv6/ip6_tables.h>	/* ipt_do_table() */

#define NAME_LEN 50

#define SERVADDR "2001:db8::a0a:6752"

#define PORT  5080
#define NIP6(addr) \
    ntohs((addr).s6_addr16[0]), \
    ntohs((addr).s6_addr16[1]), \
    ntohs((addr).s6_addr16[2]), \
    ntohs((addr).s6_addr16[3]), \
    ntohs((addr).s6_addr16[4]), \
    ntohs((addr).s6_addr16[5]), \
    ntohs((addr).s6_addr16[6]), \
    ntohs((addr).s6_addr16[7])
    
#define NIP6_FMT "%04x:%04x:%04x:%04x:%04x:%04x:%04x:%04x"
#define IN6_ARE_ADDR_EQUAL(a,b)  (__extension__  \
	({ __const struct in6_addr *__a = (__const struct in6_addr *) (a); \
		__const struct in6_addr *__b = (__const struct in6_addr *) (b); \
		__a->s6_addr32[0] == __b->s6_addr32[0]  \
		&& __a->s6_addr32[1] == __b->s6_addr32[1]  \
		&& __a->s6_addr32[2] == __b->s6_addr32[2]  \
		&& __a->s6_addr32[3] == __b->s6_addr32[3]; }))
static struct in6_addr  server_ip ;
static char func_name[NAME_LEN] = "ip6t_do_table";

/* Per-instance private data struct */
struct steph {
	struct sk_buff *skb;
	struct nf_hook_state *state;
	struct xt_table *table;
};

#if 0
static inline unsigned long pt_regs_read_reg(const struct pt_regs *regs, int r)
{
        return (r == 31) ? 0 : regs->regs[r];
}
#endif
static inline unsigned long regs_get_kernel_argument(struct pt_regs *regs,
                                                     unsigned int n)
{
#define NR_REG_ARGUMENTS 8
        if (n < NR_REG_ARGUMENTS)
                return pt_regs_read_reg(regs, n);
        return 0;
}
/*
linux 6.0
*extern unsigned int ip6t_do_table(void *priv, struct sk_buff *skb,
*                                  const struct nf_hook_state *state);
*extern unsigned int ipt_do_table(void *priv,
*                                 struct sk_buff *skb,
*                                 const struct nf_hook_state *state);
*
* linux 4.14.0-115
* extern unsigned int ipt_do_table(struct sk_buff *skb,
*                                  const struct nf_hook_state *state,
*                                                                   struct xt_table *table);
*/
/*
 * Grabbing the registers/arguments before we move on into the function
 */
static int entry_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	struct steph *data;

	data = (struct steph *)ri->data;

	data->skb = (struct sk_buff *) regs_get_kernel_argument(regs, 0);
	if (IS_ERR_OR_NULL(data->skb)) {
		pr_err("%s found NULL skb pointer", func_name);
		return 1;
	}

	data->state = (struct nf_hook_state *) regs_get_kernel_argument(regs, 1);
	if (!data->state) {
		pr_err("%s found NULL nf_hook_state pointer", func_name);
		return 1;
	}

	data->table = (struct xt_table *) regs_get_kernel_argument(regs, 2);
	if (!data->table) {
		pr_err("%s found NULL xt_table pointer", func_name);
		return 1;
	}

	return 0;
}

static void print_skb_ipv6(struct sk_buff* skb)
{
    struct ipv6hdr* ip6h;
    struct tcphdr* th;
    struct in6_addr ip6_sip, ip6_dip;
    int    sport, dport;

    
    
    ip6h = ipv6_hdr(skb);
    
    ip6_sip = ip6h->saddr;
    ip6_dip = ip6h->daddr;
    
    if (ip6h->nexthdr != NEXTHDR_TCP && ip6h->nexthdr != NEXTHDR_ICMP && ip6h->nexthdr != NEXTHDR_UDP\
            && ip6h->nexthdr != 44 ) {
        return;
    }
    if (ip6h->nexthdr == NEXTHDR_ICMP) {
        //printk(" Source: "NIP6_FMT"  Dest: " NIP6_FMT"  icmp ------>\n",
        //        NIP6(ip6_sip), NIP6(ip6_dip));
        return;
    }
    else if (ip6h->nexthdr == NEXTHDR_UDP && IN6_ARE_ADDR_EQUAL(&server_ip, &ip6_dip)) {
         //struct udphdr * uh  = (struct udphdr *)skb_transport_header(skb);
         //struct udphdr * uh  = udp_hdr(skb);
         struct udphdr * uh  = (struct udphdr *)(ip6h +1);

         sport = ntohs(uh->source);
         dport = ntohs(uh->dest);
         printk("udp Source: "NIP6_FMT" sport:%d --->Dest:  "NIP6_FMT" dport:%d    ---> \n", NIP6(ip6_sip), sport, NIP6(ip6_dip), dport);
        return;
    }
    //else if (ip6h->nexthdr == 44 && IN6_ARE_ADDR_EQUAL(&server_ip, &ip6_dip)) {
    else if (ip6h->nexthdr == 44)  {
        //struct frag_hdr *fhdr = (struct frag_hdr *)skb_transport_header(skb);
        struct frag_hdr *fhdr = (struct frag_hdr *)(ip6h +1);
        //printk("frag: "NIP6_FMT" --->Dest:  "NIP6_FMT" ,next proto : %u   \n ", NIP6(ip6_sip),  NIP6(ip6_dip),fhdr->nexthdr);
        if (skb->len - skb_network_offset(skb) < IPV6_MIN_MTU && fhdr->frag_off & htons(IP6_MF))
        {
#if 0
             if (!atomic_fetch_inc(&one))
             {
                 skb_des_func = skb->destructor;
                 skb->destructor =  dump_stack_skb;
             }
#endif
             printk("frag(id %u | offset %u ): "NIP6_FMT" --->Dest:  "NIP6_FMT" will be droped \n" ,ntohl(fhdr->identification),\
                     ntohs(fhdr->frag_off)&IP6_OFFSET,    NIP6(ip6_sip),  NIP6(ip6_dip));
        }
        return;
    }
    else if (ip6h->nexthdr == NEXTHDR_TCP && IN6_ARE_ADDR_EQUAL(&server_ip, &ip6_dip)) {
    
    //skb->transport_header = skb->network_header + sizeof(*ip6h);
    // skb->transport_header = skb->network_header + sizeof(*hdr);
    //th = tcp_hdr(skb);
    th = (struct tcphdr*)(ip6h +1);
    sport = ntohs(th->source);
    dport = ntohs(th->dest);
    //printk("tcp  Source: "NIP6_FMT" sport:%d --->Dest:  "NIP6_FMT" dport:%d    ---> \n", NIP6(ip6_sip), sport, NIP6(ip6_dip), dport);
        return;
    }        
    return;    
}

/*
 * The packet and netfilter verdict inspection
 */
static int ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	unsigned int verdict;
	struct steph *data;
	struct nf_hook_state *state;
	struct xt_table *table;
	//struct iphdr *iph;
	//struct tcphdr *tcph;
	//struct udphdr *udph;
	unsigned int proto;
	const char *devin, *devout;
	int devidxin, devidxout;
        //__u16 src, dst;

	verdict = regs_return_value(regs);

	/* We don't care about accepted packets so exit quickly */
	//if (verdict == NF_ACCEPT)
	//	return 0;

	/* Initialize the devices & indexes */
	devin = NULL;
	devidxin = 0;
	devout = NULL;
	devidxout = 0;

	data = (struct steph *)ri->data;
	if (!data) {
		pr_err("%s: NULL private data", __func__);
		return 1;
	}
        print_skb_ipv6(data->skb);
	state = data->state;
	if (state) {
		if (state->in) {
			devin = state->in->name;
			devidxin = state->in->ifindex;
		}

		if (state->out) {
			devout = state->out->name;
			devidxout = state->out->ifindex;
		}
	}

	table = data->table;

	/* Now, we replay ipt_do_table() */
	// int replay;
	// replay = replay_ipt_do_table(data->skb, data->state, table);

        pr_info("%s(%s) - devin=%s/%d, devout=%s/%d,  proto=%d, "
		"verdict=0x%x\n", func_name, table->name, devin,
					devidxin, devout, devidxout,
					proto, verdict); 

	return 0;
}

static struct kretprobe my_kretprobe = {
	.entry_handler		= entry_handler,
	.handler		= ret_handler,
	/* Necessary for the proper kzalloc() size to include data[] */
	.data_size		= sizeof(struct steph),
	/* Probe up to 20 instances concurrently. */
	.maxactive		= 20,
};

static int __init kretprobe_init(void)
{
	int ret;

        in6_pton(SERVADDR, strlen(SERVADDR), (void *)&server_ip, '\0', NULL);
	my_kretprobe.kp.symbol_name = func_name;

	ret = register_kretprobe(&my_kretprobe);
	if (ret < 0) {
		pr_err("register_kretprobe failed, returned %d\n", ret);
		return -1;
	}
	pr_info("Planted return probe at %s: 0x%lx\n",
			my_kretprobe.kp.symbol_name, (unsigned long) my_kretprobe.kp.addr);
	return 0;
}

static void __exit kretprobe_exit(void)
{
	unregister_kretprobe(&my_kretprobe);
	pr_info("kretprobe at 0x%lx unregistered\n", (unsigned long) my_kretprobe.kp.addr);

	/* nmissed > 0 suggests that maxactive was set too low. */
	pr_info("Missed probing %d instances of %s\n",
		my_kretprobe.nmissed, my_kretprobe.kp.symbol_name);
}

module_init(kretprobe_init)
module_exit(kretprobe_exit)
MODULE_LICENSE("GPL");
