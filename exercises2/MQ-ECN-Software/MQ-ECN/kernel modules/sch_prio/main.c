#include <linux/module.h>
#include <linux/types.h>
#include <linux/kernel.h>
#include <net/netlink.h>
#include <linux/pkt_sched.h>
#include <net/sch_generic.h>
#include <net/pkt_sched.h>
#include <linux/ip.h>
#include <net/dsfield.h>
#include <net/inet_ecn.h>

#include "params.h"

struct prio_rate_cfg
{
	u64 rate_bps;	//bit per second
	u32 mult;
	u32 shift;
};

struct prio_sched_data
{
/* Parameters */
	struct Qdisc **queues; /* Priority queues where queues[0] has the highest priority*/
	struct prio_rate_cfg rate;

/* Variables */
	s64 tokens;	/* Tokens in nanoseconds */
	u32 sum_len_bytes;	/* The sum of queue length in bytes */
	u32 *queue_len_bytes;	/* per-queue length in bytes */

	s64	time_ns;	/* Time check-point */
	struct Qdisc *sch;
	struct qdisc_watchdog watchdog;	/* Watchdog timer */
};

/*
 * We use this function to account for the true number of bytes sent on wire.
 * 20=frame check sequence(8B)+Interpacket gap(12B)
 * 4=Frame check sequence (4B)
 * PRIO_QDISC_MIN_PKT_BYTES=Minimum Ethernet frame size (64B)
 */
static inline unsigned int skb_size(struct sk_buff *skb)
{
	return max_t(unsigned int, skb->len + 4, PRIO_QDISC_MIN_PKT_BYTES) + 20;
}

/* Borrow from ptb */
static inline void prio_qdisc_precompute_ratedata(struct prio_rate_cfg *r)
{
	r->shift = 0;
	r->mult = 1;

	if (r->rate_bps > 0)
	{
		r->shift = 15;
		r->mult = div64_u64(8LLU * NSEC_PER_SEC * (1 << r->shift), r->rate_bps);
	}
}

/* Borrow from ptb: length (bytes) to time (nanosecond) */
static inline u64 l2t_ns(struct prio_rate_cfg *r, unsigned int len_bytes)
{
	return ((u64)len_bytes * r->mult) >> r->shift;
}

/* ECN marking */
static inline void prio_qdisc_ecn(struct sk_buff *skb)
{
	if (skb_make_writable(skb, sizeof(struct iphdr))&&ip_hdr(skb))
		IP_ECN_set_ce(ip_hdr(skb));
}

/* Classify packets and return queue ID */
static int prio_qdisc_classify(struct sk_buff *skb, struct Qdisc *sch)
{
	int i = 0;
	struct prio_sched_data *q = qdisc_priv(sch);
	struct iphdr* iph = ip_hdr(skb);
	int dscp;

	/* Return queue[0] by default*/
	if (unlikely(!iph || !(q->queues)))
		return 0;

	dscp = (const int)(iph->tos >> 2);

	for (i = 0; i < PRIO_QDISC_MAX_QUEUES; i++)
	{
		if(dscp == PRIO_QDISC_QUEUE_DSCP[i])
			return i;
	}

	return 0;
}

static struct sk_buff* prio_qdisc_dequeue_peeked(struct Qdisc *sch)
{
	struct prio_sched_data *q = qdisc_priv(sch);
	struct Qdisc *qdisc;
	struct sk_buff *skb;
	int i;

	for (i = 0; i < PRIO_QDISC_MAX_QUEUES && q->queues[i]; i++)
	{
		qdisc = q->queues[i];
		skb = qdisc_dequeue_peeked(qdisc);
		if (skb)
		{
			q->queue_len_bytes[i] -= skb_size(skb);	//update per-queue buffer occupancy
			return skb;
		}
	}
	return NULL;
}

static struct sk_buff* prio_qdisc_peek(struct Qdisc *sch)
{
	struct prio_sched_data *q = qdisc_priv(sch);
	struct Qdisc *qdisc;
	struct sk_buff *skb;
	int i;

	for (i = 0; i < PRIO_QDISC_MAX_QUEUES && q->queues[i]; i++)
	{
		qdisc = q->queues[i];
		skb = qdisc->ops->peek(qdisc);
		if (skb)
			return skb;
	}
	return NULL;
}

static struct sk_buff* prio_qdisc_dequeue(struct Qdisc *sch)
{
	struct prio_sched_data *q = qdisc_priv(sch);
	struct sk_buff *skb = NULL;

	skb = prio_qdisc_peek(sch);
	if(skb)
	{
		s64 now = ktime_get_ns();
		s64 toks = min_t(s64, now - q->time_ns, PRIO_QDISC_BUCKET_NS) + q->tokens;
		unsigned int len = skb_size(skb);
		toks -= (s64)l2t_ns(&q->rate, len);

		//If we have enough tokens to release this packet
		if (toks >= 0)
		{
			skb = prio_qdisc_dequeue_peeked(sch);
			if (unlikely(!skb))
				return NULL;

			q->time_ns = now;
			q->sum_len_bytes -= len;
			sch->q.qlen--;
			q->tokens = toks;

			//Bucket.
			if (q->tokens > PRIO_QDISC_BUCKET_NS)
				q->tokens = PRIO_QDISC_BUCKET_NS;

			if (PRIO_QDISC_ECN_SCHEME == PRIO_QDISC_DEQUE_ECN && skb->tstamp.tv64 > 0)
			{
				s64 sojourn_ns = now - skb->tstamp.tv64;
				s64 thresh_ns = (s64)l2t_ns(&q->rate, PRIO_QDISC_PORT_THRESH_BYTES);

				if (sojourn_ns > thresh_ns)
				{
					prio_qdisc_ecn(skb);
					if (PRIO_QDISC_DEBUG_MODE)
						printk(KERN_INFO "Sample sojurn time %lld ns > ECN marking threshold %lld ns (%d bytes)\n", sojourn_ns, thresh_ns, PRIO_QDISC_PORT_THRESH_BYTES);
				}
			}

			qdisc_unthrottled(sch);
			qdisc_bstats_update(sch, skb);
			//printk(KERN_INFO "sch_prio: dequeue a packet with len=%u\n",len);
			return skb;
		}
		else
		{
			//We use now+t due to absolute mode of hrtimer ((HRTIMER_MODE_ABS) )
			qdisc_watchdog_schedule_ns(&q->watchdog, now - toks, true);
			qdisc_qstats_overlimit(sch);
		}
	}

	return NULL;
}

static int prio_qdisc_enqueue(struct sk_buff *skb, struct Qdisc *sch)
{
	struct Qdisc *qdisc;
	unsigned int len = skb_size(skb);
	int ret;
	struct prio_sched_data *q = qdisc_priv(sch);
	int id = prio_qdisc_classify(skb, sch);

	qdisc = q->queues[id];
	/* No appropriate queue or per port shared buffer is overfilled or per queue static buffer is overfilled */
	if (!qdisc
	|| (PRIO_QDISC_BUFFER_MODE == PRIO_QDISC_SHARED_BUFFER && q->sum_len_bytes + len > PRIO_QDISC_SHARED_BUFFER_BYTES)
	|| (PRIO_QDISC_BUFFER_MODE == PRIO_QDISC_STATIC_BUFFER && q->queue_len_bytes[id] + len > PRIO_QDISC_QUEUE_BUFFER_BYTES[id]))
	{
		//printk(KERN_INFO "sch_prio: packet drop\n");
		qdisc_qstats_drop(sch);
		kfree_skb(skb);
		return NET_XMIT_DROP;
	}

	/* ECN marking here */
	/* Per-queue ECN marking */
	if (PRIO_QDISC_ECN_SCHEME == PRIO_QDISC_QUEUE_ECN && q->queue_len_bytes[id] + len > PRIO_QDISC_QUEUE_THRESH_BYTES[id])
		prio_qdisc_ecn(skb);
	/* Per-port ECN marking */
	else if (PRIO_QDISC_ECN_SCHEME == PRIO_QDISC_PORT_ECN && q->sum_len_bytes + len > PRIO_QDISC_PORT_THRESH_BYTES)
		prio_qdisc_ecn(skb);
	/* Dequeue latency-based ECN marking */
	else if (PRIO_QDISC_ECN_SCHEME == PRIO_QDISC_DEQUE_ECN)
		skb->tstamp = ktime_get();

	ret = qdisc_enqueue(skb, qdisc);
	if (ret == NET_XMIT_SUCCESS)
	{
		sch->q.qlen++;
		q->sum_len_bytes += len;
		q->queue_len_bytes[id] += len;
	}
	else if (net_xmit_drop_count(ret))
	{
		qdisc_qstats_drop(sch);
		qdisc_qstats_drop(qdisc);
	}

	return ret;
}

/* We don't need this */
static unsigned int prio_qdisc_drop(struct Qdisc *sch)
{
	return 0;
}

/* We don't need this */
static int prio_qdisc_dump(struct Qdisc *sch, struct sk_buff *skb)
{
	return 0;
}

/* Release Qdisc resources */
static void prio_qdisc_destroy(struct Qdisc *sch)
{
	struct prio_sched_data *q = qdisc_priv(sch);
	int i;

	if (q->queues)
	{
		for (i = 0; i < PRIO_QDISC_MAX_QUEUES && q->queues[i]; i++)
			qdisc_destroy(q->queues[i]);

		kfree(q->queues);
	}
	qdisc_watchdog_cancel(&q->watchdog);
}

static const struct nla_policy prio_qdisc_policy[TCA_TBF_MAX + 1] = {
	[TCA_TBF_PARMS] = { .len = sizeof(struct tc_tbf_qopt) },
	[TCA_TBF_RTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
	[TCA_TBF_PTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
};

/* We only leverage TC netlink interface to configure rate */
static int prio_qdisc_change(struct Qdisc *sch, struct nlattr *opt)
{
	int err;
	struct prio_sched_data *q = qdisc_priv(sch);
	struct nlattr *tb[TCA_TBF_PTAB + 1];
	struct tc_tbf_qopt *qopt;
	__u32 rate;

	err = nla_parse_nested(tb, TCA_TBF_PTAB, opt, prio_qdisc_policy);
	if(err < 0)
		return err;

	err = -EINVAL;
	if (tb[TCA_TBF_PARMS] == NULL)
		goto done;

	qopt = nla_data(tb[TCA_TBF_PARMS]);
	rate = qopt->rate.rate;
	/* convert from bytes/s to b/s */
	q->rate.rate_bps = (u64)rate << 3;
	prio_qdisc_precompute_ratedata(&q->rate);
	err = 0;
	printk(KERN_INFO "sch_prio: rate %llu Mbps\n", q->rate.rate_bps/1000000);

 done:
	return err;
}

/* Initialize Qdisc */
static int prio_qdisc_init(struct Qdisc *sch, struct nlattr *opt)
{
	int i;
	struct prio_sched_data *q = qdisc_priv(sch);
	struct Qdisc *child;

	if(sch->parent != TC_H_ROOT)
		return -EOPNOTSUPP;

	q->queues = kcalloc(PRIO_QDISC_MAX_QUEUES, sizeof(struct Qdisc *), GFP_KERNEL);
	q->queue_len_bytes = kcalloc(PRIO_QDISC_MAX_QUEUES, sizeof(u32), GFP_KERNEL);	//init per-queue buffer occupancy to 0
	if (q->queues == NULL || q->queue_len_bytes == NULL)
		return -ENOMEM;

	q->tokens = 0;
	q->time_ns = ktime_get_ns();
	q->sum_len_bytes = 0;	//init total buffer occupancy to 0
	q->sch = sch;
	qdisc_watchdog_init(&q->watchdog, sch);

	for (i = 0;i < PRIO_QDISC_MAX_QUEUES; i++)
	{
		/* bfifo is in bytes */
		child = fifo_create_dflt(sch, &bfifo_qdisc_ops, PRIO_QDISC_MAX_BUFFER_BYTES);
		if (child)
			q->queues[i] = child;
		else
			goto err;
	}
	return prio_qdisc_change(sch,opt);
err:
	prio_qdisc_destroy(sch);
	return -ENOMEM;
}

static struct Qdisc_ops prio_qdisc_ops __read_mostly = {
	.next = NULL,
	.cl_ops = NULL,
	.id = "tbf",
	.priv_size = sizeof(struct prio_sched_data),
	.init = prio_qdisc_init,
	.destroy = prio_qdisc_destroy,
	.enqueue = prio_qdisc_enqueue,
	.dequeue = prio_qdisc_dequeue,
	.peek = prio_qdisc_peek,
	.drop = prio_qdisc_drop,
	.change = prio_qdisc_change,
	.dump = prio_qdisc_dump,
	.owner = THIS_MODULE,
};

static int __init prio_qdisc_module_init(void)
{
	if (prio_qdisc_params_init() < 0)
		return -1;

	printk(KERN_INFO "sch_prio: start working\n");
	return register_qdisc(&prio_qdisc_ops);
}

static void __exit prio_qdisc_module_exit(void)
{
	prio_qdisc_params_exit();
	unregister_qdisc(&prio_qdisc_ops);
	printk(KERN_INFO "sch_prio: stop working\n");
}

module_init(prio_qdisc_module_init);
module_exit(prio_qdisc_module_exit);
MODULE_LICENSE("GPL");
