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
#include <net/ipv6.h>
#include <linux/ip.h>
#include <linux/icmp.h>
#include <linux/netdev_features.h>
#include <linux/ipv6.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/netdevice.h>
#include <linux/inet.h>
#include <linux/netfilter.h>
#include <linux/net.h>

#define SERVADDR "2001:db8::a0a:6751"

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
    
struct ip6_frag {
	u8 next_header;
	u8 reserved;
	u16 offset_flags; /* 15-3: frag offset, 2-0: flags */
	u32 ident;
} __attribute__ ((__packed__));

#define IP6_F_MF	0x0001
#define IP6_F_MASK	0xfff8
#define SKB_MARK 0x8888
//static char func_name[KSYM_NAME_LEN] = "nf_ct_frag6_gather";
static char func_name[KSYM_NAME_LEN] = "ipv6_defrag";
//static char func_name[KSYM_NAME_LEN] = "ipv6_rcv";
//static char func_name[KSYM_NAME_LEN] = "ipv6_frag_rcv";
static struct in6_addr  server_ip ;
void (*skb_des_func)(struct sk_buff *skb) = NULL;
static atomic_t one = ATOMIC_INIT(0) ;
static int __udp6_lib_rcv_wrapper(struct sk_buff * skb, struct udp_table * udptable,
                           int proto) {
    int    sport, dport;
    struct udphdr * uh  = udp_hdr(skb);
    dport = ntohs(uh->dest);
    if (PORT == dport) {
        sport = ntohs(uh->source);
        printk("udp recv: sport:%d ---> dport:%d    ---> \n",  sport, dport);
    }
    jprobe_return();
    return 0;
}
static void dump_stack_skb(struct sk_buff *skb)
{
        if(NULL != skb_des_func)
        {
            skb_des_func(skb);
        }
	return ;
}
static void kfree_skb_wrap(struct sk_buff *skb)
{
    //skb probably is NULL
    if(NULL != skb && SKB_MARK == (SKB_MARK & skb->mark))
    {
         //will coredump
         printk("Caller is %s\n",__func__ );
         dump_stack();
    }
    jprobe_return();
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
             skb->mark |= SKB_MARK;
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
#if 0
//nf_hook_entry_hookfn(const struct nf_hook_entry *entry, struct sk_buff *skb,
//                     struct nf_hook_state *state)
//{
//        return entry->hook(entry->priv, skb, state);
//}
//
//static unsigned int nf_iterate(struct sk_buff *skb,
/* Returns 1 if okfn() needs to be executed by the caller,
 *  * -EPERM for NF_DROP, 0 otherwise.  Caller must hold rcu_read_lock. */
int test_nf_hook_slow(struct sk_buff *skb, struct nf_hook_state *state,
                 const struct nf_hook_entries *e, unsigned int s)
{
#if 0
        unsigned int verdict;
        int ret;

        for (; s < e->num_hook_entries; s++) {
                verdict = nf_hook_entry_hookfn(&e->hooks[s], skb, state);
                switch (verdict & NF_VERDICT_MASK) {
                case NF_ACCEPT:
                        break;
                case NF_DROP:
                        kfree_skb(skb);
                        ret = NF_DROP_GETERR(verdict);
                        if (ret == 0)
                                ret = -EPERM;
                        return ret;
                case NF_QUEUE:
                        ret = nf_queue(skb, state, e, s, verdict);
                        if (ret == 1)
                                continue;
                        return ret;
                default:
                        /* Implicit handling for NF_STOLEN, as well as any other
 *                          * non conventional verdicts.
 *                                                   */
                        return 0;
                }
        }

        return 1;
#else
        unsigned int verdict;
        for (; s < e->num_hook_entries; s++) {
             //verdict = nf_hook_entry_hookfn(&e->hooks[s], skb, state);
             const struct nf_hook_entry *entry = &e->hooks[s];
             pr_info("hook func %p \n", entry->hook);
        }
        return 1;
#endif
}
  
static inline int test_nf_hook(u_int8_t pf, unsigned int hook, struct net *net,
                          struct sock *sk, struct sk_buff *skb,
                          struct net_device *indev, struct net_device *outdev,
                          int (*okfn)(struct net *, struct sock *, struct sk_buff *))
{
        struct nf_hook_entries *hook_head;
        int ret = 1;

#ifdef HAVE_JUMP_LABEL
        if (__builtin_constant_p(pf) &&
            __builtin_constant_p(hook) &&
            !static_key_false(&nf_hooks_needed[pf][hook]))
                return 1;
#endif

        rcu_read_lock();
        hook_head = rcu_dereference(net->nf.hooks[pf][hook]);
        if (hook_head) {
                struct nf_hook_state state;

                nf_hook_state_init(&state, hook, pf, indev, outdev,
                                   sk, net, okfn);

                ret = test_nf_hook_slow(skb, &state, hook_head, 0);
        }
        rcu_read_unlock();
        return ret;
}
#endif
static int ipv6_rcv_hook(struct sk_buff *skb, struct net_device *dev, struct packet_type *pt, struct net_device *orig_dev)
{
    
#if 0
    struct net *net = dev_net(skb->dev);
    test_nf_hook(NFPROTO_IPV6, NF_INET_PRE_ROUTING,
                       net, NULL, skb, dev, NULL,
                       NULL);
                       //(nf_hookfn*)kallsyms_lookup_name("ip6_rcv_finish"));
#endif
    print_skb_ipv6(skb);
    jprobe_return();
    return 0;
}
static  int ip6_rcv_finish_wrap(struct net *net, struct sock *sk, struct sk_buff *skb)
{
    print_skb_ipv6(skb);
    jprobe_return();
    return 0;
}
static unsigned int ipv6_defrag_wrap(void *priv,
                                struct sk_buff *skb,
                                const struct nf_hook_state *state)
{
    print_skb_ipv6(skb);
    jprobe_return();
    return 0;
}

static int nf_ct_frag6_queue_wrap(struct frag_queue *fq, struct sk_buff *skb,
                             const struct frag_hdr *fhdr, int nhoff)
{
    unsigned int payload_len;
    int offset, end;
    payload_len = ntohs(ipv6_hdr(skb)->payload_len);

    offset = ntohs(fhdr->frag_off) & ~0x7;
    end = offset + (payload_len -
    ((u8 *)(fhdr + 1) - (u8 *)(ipv6_hdr(skb) + 1)));
    if (!(fhdr->frag_off & htons(IP6_MF))) {
         print_skb_ipv6(skb);
    }
    jprobe_return();
    return 0;
}
static int ipv6_rcv_wrap(struct sk_buff *skb)
{
    print_skb_ipv6(skb);
    jprobe_return();
    return 0;
}
static int ipv6_frag_rcv_wrap(struct sk_buff *skb)
{
    print_skb_ipv6(skb);
    jprobe_return();
    return 0;
}
static struct jprobe ipv6_recv_probe;

static int probe_netif_receive_skb_fun(struct sk_buff *skb)
{
    __be16 type;
    type = skb->protocol;

    if (type == htons(ETH_P_IPV6)) {
        struct ipv6hdr *hdr;
        
        hdr = (struct ipv6hdr *)(skb->data);
        if (hdr->nexthdr == NEXTHDR_ICMP ) {
            //printk(" dev source: "NIP6_FMT"  Dest: " NIP6_FMT"  dev_netif_receive icmp------>\n",
            //    NIP6(hdr->saddr), NIP6(hdr->daddr));
        }else if(hdr->nexthdr == NEXTHDR_UDP) {
            printk(" dev source: "NIP6_FMT"  Dest: " NIP6_FMT"  dev_netif_receive udp------>\n",
                NIP6(hdr->saddr), NIP6(hdr->daddr));
            
        } else if(hdr->nexthdr == NEXTHDR_TCP) {
            //printk(" dev source: "NIP6_FMT"  Dest: " NIP6_FMT"  dev_netif_receive tcp------>\n",
            //    NIP6(hdr->saddr), NIP6(hdr->daddr));
            
        }else {
            //printk("dev source: "NIP6_FMT"  Dest: " NIP6_FMT"  dev_netif_receive ipporttype:0x%x---0x%x--->\n",
            //       NIP6(hdr->saddr), NIP6(hdr->daddr), hdr->nexthdr , ntohs(hdr->nexthdr));
            
        }
    }
    
    jprobe_return();
    return 0;
    
}

static struct jprobe  probe_netif_receive_skb;
static struct jprobe  udp4_lib_recv_probe;
static struct jprobe  nf_hook_slow_probe;
static struct kretprobe my_kretprobe;
/* per-instance private data */
struct my_data {
	//ktime_t entry_stamp;
};

/* Here we use the entry_handler to timestamp function entry */
static int entry_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
#if 0
	struct my_data *data;

	if (!current->mm)
		return 1;	/* Skip kernel threads */

	data = (struct my_data *)ri->data;
	data->entry_stamp = ktime_get();
#endif
	return 0;
}
NOKPROBE_SYMBOL(entry_handler);

/*
 *  * Return-probe handler: Log the return value and duration. Duration may turn
 *   * out to be zero consistently, depending upon the granularity of time
 *    * accounting on the platform.
 *     */
static int ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	//unsigned long retval = regs_return_value(regs);
	int retval = regs_return_value(regs);
#if 0
	struct my_data *data = (struct my_data *)ri->data;
	s64 delta;
	ktime_t now;

	now = ktime_get();
	delta = ktime_to_ns(ktime_sub(now, data->entry_stamp));
	pr_info("%s returned %lu and took %lld ns to execute\n",
			func_name, retval, (long long)delta);
#endif
	pr_info("%s returned -EINPROGRESS == retval ? %d \n", func_name, -EINPROGRESS == retval);
	//pr_info("%s returned %d, %d \n", func_name, retval, NET_RX_DROP);
	//pr_info("%s returned %lu, %d \n", func_name, retval, NET_RX_DROP);
	return 0;
}
NOKPROBE_SYMBOL(ret_handler);

static struct kretprobe my_kretprobe = {
	.handler		= ret_handler,
	.entry_handler		= entry_handler,
	.data_size		= sizeof(struct my_data),
	/* Probe up to 20 instances concurrently. */
	.maxactive		= 128,
};

int  __init  kp_init(void)
{
    int    retval;

    //inet_pton(AF_INET6, SERVADDR, &server_ip);
    in6_pton(SERVADDR, strlen(SERVADDR), (void *)&server_ip, '\0', NULL);
    printk(" server ip6: "NIP6_FMT" \n", NIP6(server_ip));
    nf_hook_slow_probe.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("kfree_skb");
    nf_hook_slow_probe.entry = (kprobe_opcode_t*)kfree_skb_wrap;
    ////nf_hook_slow_probe.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("nf_hook_slow");
    ////nf_hook_slow_probe.entry = (kprobe_opcode_t*)ipv6_defrag_wrap;
    retval = register_jprobe(&nf_hook_slow_probe);
    if (retval < 0) {
        pr_err("register_jprobe nf hook slow  failed, returned %d\n", retval);
        return retval; 
    }    
#if 0
    ipv6_recv_probe.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("ipv6_defrag");
    ipv6_recv_probe.entry = (kprobe_opcode_t*)ipv6_defrag_wrap;
#elif 0
    ipv6_recv_probe.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("nf_ct_frag6_queue");
    ipv6_recv_probe.entry = (kprobe_opcode_t*)nf_ct_frag6_queue_wrap;
#elif 0
    ipv6_recv_probe.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("ipv6_frag_rcv");
    ipv6_recv_probe.entry = (kprobe_opcode_t*)ipv6_frag_rcv_wrap;
#elif 1
    ipv6_recv_probe.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("ipv6_rcv");
    ipv6_recv_probe.entry = (kprobe_opcode_t*)ipv6_rcv_hook;
#elif 0
    ipv6_recv_probe.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("ip6_rcv_finish");
    ipv6_recv_probe.entry = (kprobe_opcode_t*)ip6_rcv_finish_wrap;
#endif
    retval = register_jprobe(&ipv6_recv_probe);
    if (retval < 0) {
	pr_err("register_jretprobe failed, returned %d\n", retval);
        goto err4;
    }    
    pr_notice("init register_jprobe %d\n", retval);


#if 0
    // use napi_gro_receive 
    probe_netif_receive_skb.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("netif_receive_skb_internal");
#else
    probe_netif_receive_skb.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("netif_receive_skb");
#endif
    probe_netif_receive_skb.entry = (kprobe_opcode_t*)probe_netif_receive_skb_fun;
    retval = register_jprobe(&probe_netif_receive_skb);
    if (retval < 0) {
	pr_err("register_jretprobe failed, returned %d\n", retval);
	goto err3;
    }    

    udp4_lib_recv_probe.kp.addr = (kprobe_opcode_t*)kallsyms_lookup_name("__udp6_lib_rcv");
    udp4_lib_recv_probe.entry = (kprobe_opcode_t*)__udp6_lib_rcv_wrapper;
    retval = register_jprobe(&udp4_lib_recv_probe);
    if (retval < 0) {
	pr_err("register_jretprobe failed, returned %d\n", retval);
	goto err2;
    }    
#if 0
    my_kretprobe.kp.symbol_name = func_name;
    retval = register_kretprobe(&my_kretprobe);
    if (retval < 0) {
	pr_err("register_kretprobe failed, returned %d\n", retval);
        goto err1;
    }    
#endif
    pr_notice("init probe_netif_receive_skb register_jprobe %d\n", retval);

    return 0;
err1:
    unregister_jprobe(&udp4_lib_recv_probe);
err2:
    unregister_jprobe(&probe_netif_receive_skb);
err3:
    unregister_jprobe(&ipv6_recv_probe);
err4:
    unregister_jprobe(&nf_hook_slow_probe);
    return retval;
}

void  __exit   kp_exit(void)
{
    unregister_jprobe(&nf_hook_slow_probe);
    unregister_jprobe(&ipv6_recv_probe);
    unregister_jprobe(&probe_netif_receive_skb);
    unregister_jprobe(&udp4_lib_recv_probe);
    unregister_kretprobe(&my_kretprobe);
    pr_notice("module removed\n ");
}

module_init(kp_init);
module_exit(kp_exit);
 
MODULE_AUTHOR("Okash Khawaja");
MODULE_DESCRIPTION("Experimental System Call Tracer");
MODULE_LICENSE("GPL");
