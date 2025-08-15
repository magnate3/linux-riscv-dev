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
#include <net/ip.h>
#include <linux/netfilter/x_tables.h>
#include <linux/netfilter_ipv4/ip_tables.h>	/* ipt_do_table() */

#define NAME_LEN 50

static char func_name[NAME_LEN] = "ipt_do_table";

/* Per-instance private data struct */
struct steph {
	struct sk_buff *skb;
	struct nf_hook_state *state;
	struct xt_table *table;
};

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

/*
 * COMMENTING THIS OUT
 * Highly non-portable across kernel versions so, just to play around
 * Helper functions to replay the packet admission control

static inline struct ipt_entry *
get_entry(const void *base, unsigned int offset)
{
	return (struct ipt_entry *)(base + offset);
}

static inline
struct ipt_entry *ipt_next_entry(const struct ipt_entry *entry)
{
	return (void *)entry + entry->next_offset;
}

static inline bool
ip_packet_match(const struct iphdr *ip,
		const char *indev,
		const char *outdev,
		const struct ipt_ip *ipinfo,
		int isfrag)
{
	unsigned long ret;

	if (NF_INVF(ipinfo, IPT_INV_SRCIP,
		    (ip->saddr & ipinfo->smsk.s_addr) != ipinfo->src.s_addr) ||
	    NF_INVF(ipinfo, IPT_INV_DSTIP,
		    (ip->daddr & ipinfo->dmsk.s_addr) != ipinfo->dst.s_addr))
		return false;

	ret = ifname_compare_aligned(indev, ipinfo->iniface, ipinfo->iniface_mask);

	if (NF_INVF(ipinfo, IPT_INV_VIA_IN, ret != 0))
		return false;

	ret = ifname_compare_aligned(outdev, ipinfo->outiface, ipinfo->outiface_mask);

	if (NF_INVF(ipinfo, IPT_INV_VIA_OUT, ret != 0))
		return false;

	if (ipinfo->proto &&
	    NF_INVF(ipinfo, IPT_INV_PROTO, ip->protocol != ipinfo->proto))
		return false;

	if (NF_INVF(ipinfo, IPT_INV_FRAG,
		    (ipinfo->flags & IPT_F_FRAG) && !isfrag))
		return false;

	return true;
}

static inline const struct xt_entry_target *
ipt_get_target_c(const struct ipt_entry *e)
{
        return ipt_get_target((struct ipt_entry *)e);
}

unsigned int
replay_ipt_do_table(struct sk_buff *skb,
	     const struct nf_hook_state *state,
	     struct xt_table *table)
{
	unsigned int hook = state->hook;
	static const char nulldevname[IFNAMSIZ] __attribute__((aligned(sizeof(long))));
	const struct iphdr *ip;
	unsigned int verdict = NF_DROP;
	const char *indev, *outdev;
	const void *table_base;
	struct ipt_entry *e, **jumpstack;
	unsigned int stackidx, cpu;
	const struct xt_table_info *private;
	struct xt_action_param acpar;
	unsigned int addend;

	stackidx = 0;
	ip = ip_hdr(skb);
	indev = state->in ? state->in->name : nulldevname;
	outdev = state->out ? state->out->name : nulldevname;
	acpar.fragoff = ntohs(ip->frag_off) & IP_OFFSET;
	acpar.thoff   = ip_hdrlen(skb);
	acpar.hotdrop = false;
	acpar.state   = state;

	WARN_ON(!(table->valid_hooks & (1 << hook)));
	local_bh_disable();
	addend = xt_write_recseq_begin();
	private = READ_ONCE(table->private);
	cpu        = smp_processor_id();
	table_base = private->entries;
	jumpstack  = (struct ipt_entry **)private->jumpstack[cpu];

	if (static_key_false(&xt_tee_enabled))
		jumpstack += private->stacksize * __this_cpu_read(nf_skb_duplicated);

	e = get_entry(table_base, private->hook_entry[hook]);

	do {
		const struct xt_entry_target *t;
		const struct xt_entry_match *ematch;
		struct xt_counters *counter;

		WARN_ON(!e);
		if (!ip_packet_match(ip, indev, outdev,
		    &e->ip, acpar.fragoff)) {
 no_match:
			e = ipt_next_entry(e);
			continue;
		}

		xt_ematch_foreach(ematch, e) {
			acpar.match     = ematch->u.kernel.match;
			acpar.matchinfo = ematch->data;

			// NMS
			if (acpar.match)
				pr_info("STEPH: match %s\n", acpar.match->name
								? acpar.match->name : "(null)");

			if (!acpar.match->match(skb, &acpar))
				goto no_match;
		}

		counter = xt_get_this_cpu_counter(&e->counters);
		ADD_COUNTER(*counter, skb->len, 1);

		t = ipt_get_target_c(e);
		WARN_ON(!t->u.kernel.target);

		// NMS
		pr_info("STEPH: target %s\n", t->u.kernel.target->name
							? t->u.kernel.target->name : "(null)");

		if (!t->u.kernel.target->target) {
			int v;

			v = ((struct xt_standard_target *)t)->verdict;
			if (v < 0) {
				if (v != XT_RETURN) {
					verdict = (unsigned int)(-v) - 1;
					break;
				}
				if (stackidx == 0) {
					e = get_entry(table_base,
					    private->underflow[hook]);
				} else {
					e = jumpstack[--stackidx];
					e = ipt_next_entry(e);
				}
				continue;
			}
			if (table_base + v != ipt_next_entry(e) &&
			    !(e->ip.flags & IPT_F_GOTO)) {
				if (unlikely(stackidx >= private->stacksize)) {
					verdict = NF_DROP;
					break;
				}
				jumpstack[stackidx++] = e;
			}

			e = get_entry(table_base, v);
			continue;
		}

		acpar.target   = t->u.kernel.target;
		acpar.targinfo = t->data;

		verdict = t->u.kernel.target->target(skb, &acpar);
		if (verdict == XT_CONTINUE) {
			ip = ip_hdr(skb);
			e = ipt_next_entry(e);
		} else {
			break;
		}
	} while (!acpar.hotdrop);

	xt_write_recseq_end(addend);
	local_bh_enable();

	if (acpar.hotdrop)
		return NF_DROP;
	else return verdict;
}

*
*/

/*
 * The packet and netfilter verdict inspection
 */
static int ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	unsigned int verdict;
	struct steph *data;
	struct nf_hook_state *state;
	struct xt_table *table;
	struct iphdr *iph;
	struct tcphdr *tcph;
	struct udphdr *udph;
	__u32 saddr, daddr;
	__u16 src, dst;
	unsigned int proto;
	const char *devin, *devout;
	int devidxin, devidxout;

	verdict = regs_return_value(regs);

	/* We don't care about accepted packets so exit quickly */
	if (verdict == NF_ACCEPT)
		return 0;

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

	iph = ip_hdr(data->skb);
	if (!iph) {
		pr_err("%s: failed to find the iphdr structure", __func__);
		return 1;
	}

	saddr = ntohl(iph->saddr);
	daddr = ntohl(iph->daddr);

	switch(iph->protocol) {
		case IPPROTO_TCP:
		       	tcph = tcp_hdr(data->skb);

			if (!tcph) {
				pr_err("%s: failed to find the tcphdr structure", __func__);
				return 1;
			}

			proto = 0x000000ff & IPPROTO_TCP;
			src = ntohs(tcph->source);
			dst = ntohs(tcph->dest);

			break;
		case IPPROTO_UDP:
			udph = udp_hdr(data->skb);

			if (!udph) {
				pr_err("%s: failed to find the udph structure", __func__);
				return 1;
			}

			proto = 0x000000ff & IPPROTO_UDP;
			src = ntohs(udph->source);
			dst = ntohs(udph->dest);

			break;
		default:
			pr_warn("%s: unsupported L4 protocol; only TCP and UDP are supported", func_name);
			return 0;
	}

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

	pr_info("%s(%s) - devin=%s/%d, devout=%s/%d, saddr=0x%x, daddr=0x%x, proto=%d, "
		"spt=0x%x, dpt=0x%x, verdict=0x%x\n", func_name, table->name, devin,
					devidxin, devout, devidxout, saddr, daddr,
					proto, src, dst, verdict);

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
