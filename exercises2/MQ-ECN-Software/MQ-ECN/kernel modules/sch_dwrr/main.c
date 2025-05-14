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

struct dwrr_rate_cfg
{
	u64	rate_bps;
	u32	mult;
	u32	shift;
};

/**
 *	struct dwrr_class - a Class of Service (CoS) queue
 *	@id: queue ID
 *	@prio: queue priority (0 is the highest)
 *	@len_bytes: queue length in bytes
 *	@qdisc: FIFO queue to store sk_buff
 *
 *	For DWRR scheduling
 *	@deficit: deficit counter of this queue (bytes)
 *	@start_time: time when this queue is inserted to active list
 *	@last_pkt_time: time when this queue transmits the last packet
 *	@quantum: quantum in bytes of this queue
 *	@alist: active linked list
 *
 *	For CoDel
 *	@count: how many marks since the last time we entered marking state
 *	@lastcount: count at entry to marking/dropping state
 *	@marking: set to true if in mark/drop state
 *	@rec_inv_sqrt: reciprocal value of sqrt(count) >> 1
 *	@first_above_time: when we went (or will go) continuously above target
 *	for interval
 *	@mark_next: time to mark next packet, or when we marked last
 *	@ldelay: sojourn time of last dequeued packet
 */
struct dwrr_class
{
	u8		id;
	u8		prio;
	u32		len_bytes;
	struct Qdisc	*qdisc;

	u32		deficit;
	s64		start_time;
	s64		last_pkt_time;
	u32		quantum;
	struct list_head	alist;

	u32		count;
	u32		lastcount;
	bool		marking;
	u16		rec_inv_sqrt;
	codel_time_t	first_above_time;
	codel_time_t	mark_next;
	codel_time_t	ldelay;
};

/**
 *	struct dwrr_sched_data - DWRR scheduler
 *	@queues: multiple Class of Service (CoS) queues
 *	@rate: shaping rate
 *	@watchdog: watchdog timer for token bucket rate limiter
 *	@active: active queues for different priorities
 *
 *	@tokens: tokens in ns
 *	@time_ns: time check-point
 *	@sum_len_bytes: the total buffer occupancy (in bytes)
 *	@prio_len_bytes: buffer occupancy (in bytes) for different priorities
 *	@round_time: smooth round time (in ns) for different priorities
 *	@last_idle_time: last time (in ns) when the buffer becomes empty for
 *	different priorities
 */
struct dwrr_sched_data
{
	struct dwrr_class	queues[dwrr_max_queues];
	struct dwrr_rate_cfg	rate;
	struct qdisc_watchdog	watchdog;
	struct list_head	active[dwrr_max_prio];

	s64	tokens;
	s64	time_ns;
	u32	sum_len_bytes;
	u32	prio_len_bytes[dwrr_max_prio];
	s64	round_time[dwrr_max_prio];
	s64	last_idle_time[dwrr_max_prio];
};

static inline void print_dwrr_sched_data(struct Qdisc *sch)
{
        int i;
	struct dwrr_sched_data *q = qdisc_priv(sch);

        printk(KERN_INFO "==========================================");
        printk(KERN_INFO "sch_dwrr on %s\n", sch->dev_queue->dev->name);
        printk(KERN_INFO "rate: %llu Mbps\n", q->rate.rate_bps / 1000000);
        printk(KERN_INFO "total buffer occupancy: %u\n", q->sum_len_bytes);

        printk(KERN_INFO "==========================================");
        printk(KERN_INFO "per-queue buffer occupancy\n");
        for (i = 0; i < dwrr_max_queues; i++)
                printk(KERN_INFO " queue %d: %u\n", i, q->queues[i].len_bytes);

        printk(KERN_INFO "==========================================");
        printk(KERN_INFO "per-priority buffer occupancy\n");
        for (i = 0; i < dwrr_max_prio; i++)
                printk(KERN_INFO " priority %d: %u\n", i, q->prio_len_bytes[i]);

        printk(KERN_INFO "==========================================");
        printk(KERN_INFO "per-priority smooth round time\n");
        for (i = 0; i < dwrr_max_prio; i++)
                printk(KERN_INFO " priority %d: %llu\n", i, q->round_time[i]);

        printk(KERN_INFO "==========================================");
}

/* nanosecond to codel time (1 << dwrr_codel_shift ns) */
static inline codel_time_t ns_to_codel_time(s64 ns)
{
	return ns >> dwrr_codel_shift;
}

/* Exponential Weighted Moving Average (EWMA) for s64 */
static inline s64 s64_ewma(s64 smooth, s64 sample, int weight, int shift)
{
	s64 val = smooth * weight;
	val += sample * ((1 << shift) - weight);
	return val >> shift;
}

/* Use EWMA to update round time */
static inline s64 ewma_round(s64 smooth, s64 sample)
{
	return s64_ewma(smooth, sample, dwrr_round_alpha, dwrr_round_shift);
}

/* Reset round time after a long period of idle time */
static void reset_round(struct dwrr_sched_data *q, int prio)
{
	int i;
	s64 interval, iter = 0;

	if (likely(q->prio_len_bytes[prio] == 0 && dwrr_idle_interval_ns > 0))
	{
		interval = ktime_get_ns() - q->last_idle_time[prio];
		iter = div_s64(interval, dwrr_idle_interval_ns);
	}

	if (iter > dwrr_max_iteration || unlikely(iter < 0))
	{
		q->round_time[prio] = 0;
		return;
	}

	for (i = 0; i < iter; i++)
		q->round_time[prio] = ewma_round(q->round_time[prio], 0);
}

static inline void print_round(s64 smooth, s64 sample)
{
	/* Print necessary information in debug mode */
	if (dwrr_enable_debug == dwrr_enable && dwrr_ecn_scheme == dwrr_mq_ecn)
	{
		printk(KERN_INFO "sample round time %lld\n", sample);
		printk(KERN_INFO "smooth round time %lld\n", smooth);
	}
}

/*
 * We use this function to account for the true number of bytes sent on wire.
 * 20 = frame check sequence(8B)+Interpacket gap(12B)
 * 4 = Frame check sequence (4B)
 * dwrr_min_pkt_bytes = Minimum Ethernet frame size (64B)
 */
static inline unsigned int skb_size(struct sk_buff *skb)
{
	return max_t(unsigned int, skb->len + 4, dwrr_min_pkt_bytes) + 20;
}

/* Borrow from ptb */
static inline void precompute_ratedata(struct dwrr_rate_cfg *r)
{
	r->shift = 0;
	r->mult = 1;

	if (r->rate_bps > 0)
	{
		r->shift = 15;
		r->mult = div64_u64(8LLU * NSEC_PER_SEC * (1 << r->shift),
				    r->rate_bps);
	}
}

/* Borrow from ptb: length (bytes) to time (nanosecond) */
static inline u64 l2t_ns(struct dwrr_rate_cfg *r, unsigned int len_bytes)
{
	return ((u64)len_bytes * r->mult) >> r->shift;
}

/* MQ-ECN marking */
static void mq_ecn_marking(struct sk_buff *skb,
		      	   struct dwrr_sched_data *q,
		      	   struct dwrr_class *cl)
{
	u64 ecn_thresh_bytes, estimate_rate_bps;
	s64 round_time = q->round_time[cl->prio];

	if (round_time > 0)
		estimate_rate_bps = div_u64((u64)cl->quantum << 33, round_time);
	else
		estimate_rate_bps = q->rate.rate_bps;

	/* rate <= link capacity */
	estimate_rate_bps = min_t(u64, estimate_rate_bps, q->rate.rate_bps);
	ecn_thresh_bytes = div64_u64(estimate_rate_bps * dwrr_port_thresh_bytes,
				     q->rate.rate_bps);

	if (cl->len_bytes > ecn_thresh_bytes)
		INET_ECN_set_ce(skb);

	if (dwrr_enable_debug == dwrr_enable)
		printk(KERN_INFO "queue %d quantum %u ECN threshold %llu\n",
	       	       cl->id,
		       cl->quantum,
		       ecn_thresh_bytes);
}


/* Queue length based ECN marking: per-queue, per-port and MQ-ECN */
void dwrr_qlen_marking(struct sk_buff *skb,
		       struct dwrr_sched_data *q,
		       struct dwrr_class *cl)
{
	switch (dwrr_ecn_scheme)
	{
		/* Per-queue ECN marking */
		case dwrr_queue_ecn:
		{
			if (cl->len_bytes > dwrr_queue_thresh_bytes[cl->id])
				INET_ECN_set_ce(skb);
			break;
		}
		/* Per-port ECN marking */
		case dwrr_port_ecn:
		{
			if (q->sum_len_bytes > dwrr_port_thresh_bytes)
				INET_ECN_set_ce(skb);
			break;
		}
		/* MQ-ECN */
		case dwrr_mq_ecn:
		{
			mq_ecn_marking(skb, q, cl);
			break;
		}
		default:
		{
			break;
		}
	}
}

/* TCN marking scheme */
static inline void tcn_marking(struct sk_buff *skb)
{
	codel_time_t delay;
	delay = ns_to_codel_time(ktime_get_ns() - skb->tstamp.tv64);

	if (codel_time_after(delay, (codel_time_t)dwrr_tcn_thresh))
		INET_ECN_set_ce(skb);
}

/* Borrow from codel_should_drop in Linux kernel */
static bool codel_should_mark(const struct sk_buff *skb,
	                      struct dwrr_class *cl,
			      s64 now_ns)
{
	bool ok_to_mark;
	codel_time_t now = ns_to_codel_time(now_ns);

	cl->ldelay = ns_to_codel_time(now_ns - skb->tstamp.tv64);

	if (codel_time_before(cl->ldelay, (codel_time_t)dwrr_codel_target) ||
	    cl->len_bytes <= dwrr_max_pkt_bytes)
	{
		/* went below - stay below for at least interval */
		cl->first_above_time = 0;
		return false;
	}

	ok_to_mark = false;
	if (cl->first_above_time == 0)
	{
		/* just went above from below. If we stay above
         	 * for at least interval we'll say it's ok to mark
         	 */
		cl->first_above_time = now + dwrr_codel_interval;
	}
	else if (codel_time_after(now, cl->first_above_time))
	{
		ok_to_mark = true;
	}

	return ok_to_mark;
}


/* or sizeof_in_bits(rec_inv_sqrt) */
#define REC_INV_SQRT_BITS (8 * sizeof(u16))
/* needed shift to get a Q0.32 number from rec_inv_sqrt */
#define REC_INV_SQRT_SHIFT (32 - REC_INV_SQRT_BITS)

/* Borrow from codel_Newton_step in Linux kernel */
static void codel_Newton_step(struct dwrr_class *cl)
{
	u32 invsqrt = ((u32)cl->rec_inv_sqrt) << REC_INV_SQRT_SHIFT;
	u32 invsqrt2 = ((u64)invsqrt * invsqrt) >> 32;
	u64 val = (3LL << 32) - ((u64)cl->count * invsqrt2);

	val >>= 2; /* avoid overflow in following multiply */
	val = (val * invsqrt) >> (32 - 2 + 1);

	cl->rec_inv_sqrt = val >> REC_INV_SQRT_SHIFT;
}

/*
 * CoDel control_law is t + interval/sqrt(count)
 * We maintain in rec_inv_sqrt the reciprocal value of sqrt(count) to avoid
 * both sqrt() and divide operation.
 *
 * Borrow from codel_control_law in Linux kernel
 */
static codel_time_t codel_control_law(codel_time_t t,
				      codel_time_t interval,
				      u32 rec_inv_sqrt)
{
	return t + reciprocal_scale(interval,
				    rec_inv_sqrt << REC_INV_SQRT_SHIFT);
}

/* CoDel ECN marking. Borrow from codel_dequeue in Linux kernel */
static void codel_marking(struct sk_buff *skb, struct dwrr_class *cl)
{
	s64 now_ns = ktime_get_ns();
	codel_time_t now = ns_to_codel_time(now_ns);
	bool mark = codel_should_mark(skb, cl, now_ns);

	if (cl->marking)
	{
		if (!mark)
		{
			/* sojourn time below target - leave marking state */
			cl->marking = false;
		}
		else if (codel_time_after_eq(now, cl->mark_next))
		{
			/* It's time for the next mark */
			cl->count++;
			codel_Newton_step(cl);
			cl->mark_next = codel_control_law(cl->mark_next,
					  	          dwrr_codel_interval,
					                  cl->rec_inv_sqrt);
			INET_ECN_set_ce(skb);
		}
	}
	else if (mark)
	{
		u32 delta;

		INET_ECN_set_ce(skb);
		cl->marking = true;
		/* if min went above target close to when we last went below it
         	 * assume that the drop rate that controlled the queue on the
         	 * last cycle is a good starting point to control it now.
         	 */
		delta = cl->count - cl->lastcount;
 		if (delta > 1 &&
 		    codel_time_before(now - cl->mark_next,
 				      (codel_time_t)dwrr_codel_interval * 16))
 		{
         		cl->count = delta;
             		/* we dont care if rec_inv_sqrt approximation
              		 * is not very precise :
              		 * Next Newton steps will correct it quadratically.
              		 */
         		codel_Newton_step(cl);
 		}
 		else
 		{
 			cl->count = 1;
 			cl->rec_inv_sqrt = ~0U >> REC_INV_SQRT_SHIFT;
 		}
 		cl->lastcount = cl->count;
 		cl->mark_next = codel_control_law(now,
 						  dwrr_codel_interval,
 						  cl->rec_inv_sqrt);
	}
}

static struct dwrr_class *dwrr_classify(struct sk_buff *skb, struct Qdisc *sch)
{
	struct dwrr_sched_data *q = qdisc_priv(sch);
	struct iphdr* iph = ip_hdr(skb);
	int i, dscp;

	if (unlikely(!(q->queues)))
		return NULL;

	/* Return queue[0] by default*/
	if (unlikely(!iph))
		return &(q->queues[0]);

	dscp = iph->tos >> 2;

	for (i = 0; i < dwrr_max_queues; i++)
	{
		if (dscp == dwrr_queue_dscp[i])
			return &(q->queues[i]);
	}

	return &(q->queues[0]);
}

/* We don't need this */
static struct sk_buff *dwrr_peek(struct Qdisc *sch)
{
	return NULL;
}


/* Decide whether the packet can be transmitted according to Token Bucket */
static s64 tbf_schedule(unsigned int len, struct dwrr_sched_data *q, s64 now)
{
	s64 pkt_ns, toks;

	toks = now - q->time_ns;
	toks = min_t(s64, toks, (s64)l2t_ns(&q->rate, dwrr_bucket_bytes));
	toks += q->tokens;

	pkt_ns = (s64)l2t_ns(&q->rate, len);

	return toks - pkt_ns;
}

/* Find the highest priority that is non-empty */
int prio_schedule(struct dwrr_sched_data *q)
{
	int i;

	for (i = 0; i < dwrr_max_prio; i++)
	{
		if (!list_empty(&q->active[i]))
			return i;
	}

	return -1;
}

static struct sk_buff *dwrr_dequeue(struct Qdisc *sch)
{
	struct dwrr_sched_data *q = qdisc_priv(sch);
	struct dwrr_class *cl = NULL;
	struct sk_buff *skb = NULL;
	s64 sample, result;
	s64 now = ktime_get_ns();
	s64 bucket_ns = (s64)l2t_ns(&q->rate, dwrr_bucket_bytes);
	unsigned int len;
	struct list_head *active = NULL;
	int prio = prio_schedule(q);

	if (prio < 0)
		return NULL;
	else
		active = &q->active[prio];

	while (1)
	{
		cl = list_first_entry(active, struct dwrr_class, alist);
		if (unlikely(!cl))
			return NULL;

		/* get head packet */
		skb = cl->qdisc->ops->peek(cl->qdisc);
		if (unlikely(!skb))
			return NULL;

		len = skb_size(skb);

		/* If this packet can be scheduled by DWRR */
		if (len <= cl->deficit)
		{
			result = tbf_schedule(len, q, now);
			/* If we don't have enough tokens */
			if (result < 0)
			{
				/* For hrtimer absolute mode, we use now + t */
				qdisc_watchdog_schedule_ns(&q->watchdog,
							   now - result,
							   true);
				qdisc_qstats_overlimit(sch);
				return NULL;
			}

			skb = qdisc_dequeue_peeked(cl->qdisc);
			if (unlikely(!skb))
				return NULL;

			q->prio_len_bytes[prio] -= len;
			if (q->prio_len_bytes[prio] == 0)
				q->last_idle_time[prio] = now;

			q->sum_len_bytes -= len;
			sch->q.qlen--;
			cl->len_bytes -= len;
			cl->deficit -= len;
			cl->last_pkt_time = now + l2t_ns(&q->rate, len);

			if (cl->qdisc->q.qlen == 0)
			{
				list_del(&cl->alist);
				sample = cl->last_pkt_time - cl->start_time;
				q->round_time[prio] = ewma_round(q->round_time[prio], sample);
				print_round(q->round_time[prio], sample);
			}

			/* Bucket */
			q->time_ns = now;
			q->tokens = min_t(s64, result, bucket_ns);
			qdisc_unthrottled(sch);
			qdisc_bstats_update(sch, skb);


			/* TCN */
			if (dwrr_ecn_scheme == dwrr_tcn)
				tcn_marking(skb);
			/* CoDel */
			else if (dwrr_ecn_scheme == dwrr_codel)
				codel_marking(skb, cl);
			/* dequeu equeue length based ECN marking */
			else if (dwrr_enable_dequeue_ecn == dwrr_enable)
				dwrr_qlen_marking(skb, q, cl);

			return skb;
		}

		/* This packet can not be scheduled by DWRR */
		sample = cl->last_pkt_time - cl->start_time;
		q->round_time[prio] = ewma_round(q->round_time[prio], sample);
		cl->start_time = cl->last_pkt_time;
		cl->quantum = dwrr_queue_quantum[cl->id];
		list_move_tail(&cl->alist, active);

		/* WRR */
		if (dwrr_enable_wrr == dwrr_enable)
			cl->deficit = cl->quantum;
		else
			cl->deficit += cl->quantum;

		print_round(q->round_time[prio], sample);
	}

	return NULL;
}

static bool dwrr_buffer_overfill(unsigned int len,
				 struct dwrr_class *cl,
				 struct dwrr_sched_data *q)
{
	/* per-port shared buffer */
	if (dwrr_buffer_mode == dwrr_shared_buffer &&
	    q->sum_len_bytes + len > dwrr_shared_buffer_bytes)
		return true;
	/* per-queue static buffer */
	else if (dwrr_buffer_mode == dwrr_static_buffer &&
		 cl->len_bytes + len > dwrr_queue_buffer_bytes[cl->id])
		return true;
	else
		return false;
}


static int dwrr_enqueue(struct sk_buff *skb, struct Qdisc *sch)
{
	struct dwrr_class *cl = NULL;
	unsigned int len = skb_size(skb);
	struct dwrr_sched_data *q = qdisc_priv(sch);
	int ret, prio;

	cl = dwrr_classify(skb, sch);
	if (likely(cl))
	{
		prio = dwrr_queue_prio[cl->id];
		if (q->prio_len_bytes[prio] == 0)
			reset_round(q, prio);
	}

	/* No appropriate queue or the switch buffer is overfilled */
	if (unlikely(!cl) || dwrr_buffer_overfill(len, cl, q))
	{
		qdisc_qstats_drop(sch);
		qdisc_qstats_drop(cl->qdisc);
		kfree_skb(skb);
		return NET_XMIT_DROP;
	}

	ret = qdisc_enqueue(skb, cl->qdisc);
	if (unlikely(ret != NET_XMIT_SUCCESS))
	{
		if (likely(net_xmit_drop_count(ret)))
		{
			qdisc_qstats_drop(sch);
			qdisc_qstats_drop(cl->qdisc);
		}
		return ret;
	}

	/* If the queue is empty, insert it to the linked list */
	if (cl->qdisc->q.qlen == 1)
	{
		cl->start_time = ktime_get_ns();
		cl->quantum = dwrr_queue_quantum[cl->id];
		cl->prio = prio;
		cl->deficit = cl->quantum;
		list_add_tail(&cl->alist, &(q->active[cl->prio]));
	}

	/* Update queue sizes (per port/priority/queue) */
	sch->q.qlen++;
	q->sum_len_bytes += len;
	q->prio_len_bytes[cl->prio] += len;
	cl->len_bytes += len;

	/* sojourn time based ECN marking: TCN and CoDel */
	if (dwrr_ecn_scheme == dwrr_tcn || dwrr_ecn_scheme == dwrr_codel)
		skb->tstamp = ktime_get();
	/* enqueue queue length based ECN marking */
	else if (dwrr_enable_dequeue_ecn == dwrr_disable)
		dwrr_qlen_marking(skb, q, cl);

	return ret;
}

/* We don't need this */
static unsigned int dwrr_drop(struct Qdisc *sch)
{
	return 0;
}

/* We don't need this */
static int dwrr_dump(struct Qdisc *sch, struct sk_buff *skb)
{
	return 0;
}

/* Release Qdisc resources */
static void dwrr_destroy(struct Qdisc *sch)
{
	struct dwrr_sched_data *q = qdisc_priv(sch);
	int i;

	if (likely(q->queues))
	{
		for (i = 0; i < dwrr_max_queues && (q->queues[i]).qdisc; i++)
			qdisc_destroy((q->queues[i]).qdisc);
	}
	qdisc_watchdog_cancel(&q->watchdog);
	printk(KERN_INFO "destroy sch_dwrr on %s\n", sch->dev_queue->dev->name);
	print_dwrr_sched_data(sch);
}

static const struct nla_policy dwrr_policy[TCA_TBF_MAX + 1] = {
	[TCA_TBF_PARMS] = { .len = sizeof(struct tc_tbf_qopt) },
	[TCA_TBF_RTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
	[TCA_TBF_PTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
};

/* We only leverage TC netlink interface to configure rate */
static int dwrr_change(struct Qdisc *sch, struct nlattr *opt)
{
	int err;
	struct dwrr_sched_data *q = qdisc_priv(sch);
	struct nlattr *tb[TCA_TBF_PTAB + 1];
	struct tc_tbf_qopt *qopt;
	__u32 rate;

	err = nla_parse_nested(tb, TCA_TBF_PTAB, opt, dwrr_policy);
	if(err < 0)
		return err;

	err = -EINVAL;
	if (!tb[TCA_TBF_PARMS])
		goto done;

	qopt = nla_data(tb[TCA_TBF_PARMS]);
	rate = qopt->rate.rate;
	/* convert from bytes/s to b/s */
	q->rate.rate_bps = (u64)rate << 3;
	precompute_ratedata(&q->rate);
	err = 0;

	printk(KERN_INFO "change sch_dwrr on %s\n", sch->dev_queue->dev->name);
        print_dwrr_sched_data(sch);
 done:
	return err;
}

/* Initialize Qdisc */
static int dwrr_init(struct Qdisc *sch, struct nlattr *opt)
{
	int i;
	struct dwrr_sched_data *q = qdisc_priv(sch);
	struct Qdisc *child;
	s64 now_ns = ktime_get_ns();

	if(sch->parent != TC_H_ROOT)
		return -EOPNOTSUPP;

	q->tokens = 0;
	q->time_ns = now_ns;
	q->sum_len_bytes = 0;
	qdisc_watchdog_init(&q->watchdog, sch);

	/* Initialize per-priority variables */
	for (i = 0; i < dwrr_max_prio; i++)
	{
		INIT_LIST_HEAD(&q->active[i]);
		q->prio_len_bytes[i] = 0;
		q->round_time[i] = 0;
		q->last_idle_time[i] = now_ns;
	}

	/* Initialize per-queue variables */
	for (i = 0; i < dwrr_max_queues; i++)
	{
		/* bfifo is in bytes */
		child = fifo_create_dflt(sch,
					&bfifo_qdisc_ops, dwrr_max_buffer_bytes);
		if (likely(child))
			(q->queues[i]).qdisc = child;
		else
			goto err;

		/* Initialize per-queue variables */
		INIT_LIST_HEAD(&(q->queues[i]).alist);
		(q->queues[i]).id = i;
		(q->queues[i]).len_bytes = 0;
		(q->queues[i]).prio = 0;
		(q->queues[i]).deficit = 0;
		(q->queues[i]).start_time = now_ns;
		(q->queues[i]).last_pkt_time = now_ns;
		(q->queues[i]).quantum = 0;
		(q->queues[i]).count = 0;
		(q->queues[i]).lastcount = 0;
		(q->queues[i]).marking = false;
		(q->queues[i]).rec_inv_sqrt = 0;
		(q->queues[i]).first_above_time = 0;
		(q->queues[i]).mark_next = 0;
		(q->queues[i]).ldelay = 0;
	}
	return dwrr_change(sch,opt);
err:
	dwrr_destroy(sch);
	return -ENOMEM;
}

static struct Qdisc_ops dwrr_ops __read_mostly = {
	.next		=	NULL,
	.cl_ops		=	NULL,
	.id		=	"tbf",
	.priv_size	=	sizeof(struct dwrr_sched_data),
	.init		=	dwrr_init,
	.destroy	=	dwrr_destroy,
	.enqueue	=	dwrr_enqueue,
	.dequeue	=	dwrr_dequeue,
	.peek		=	dwrr_peek,
	.drop		=	dwrr_drop,
	.change		=	dwrr_change,
	.dump		=	dwrr_dump,
	.owner 		= 	THIS_MODULE,
};

static int __init dwrr_module_init(void)
{
	if (unlikely(!dwrr_params_init()))
		return -1;

	printk(KERN_INFO "sch_dwrr: start working\n");
	return register_qdisc(&dwrr_ops);
}

static void __exit dwrr_module_exit(void)
{
	dwrr_params_exit();
	unregister_qdisc(&dwrr_ops);
	printk(KERN_INFO "sch_dwrr: stop working\n");
}

module_init(dwrr_module_init);
module_exit(dwrr_module_exit);
MODULE_LICENSE("GPL");
