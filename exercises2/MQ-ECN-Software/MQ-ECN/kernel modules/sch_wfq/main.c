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

struct wfq_rate_cfg
{
        u64     rate_bps;
        u32     mult;
        u32     shift;
};

/**
 *      struct wfq_class - a Class of Service (CoS) queue
 *      @id: queue ID
 *      @prio: queue priority (0 is the highest)
 *      @len_bytes: queue length in bytes
 *      @qdisc: FIFO queue to store sk_buff
 *
 *      For WFQ scheduling
 *      @head_fin_time: virtual finish time of the head packet
 *
 *      For CoDel
 *      @count: how many marks since the last time we entered marking state
 *      @lastcount: count at entry to marking/dropping state
 *      @marking: set to true if in mark/drop state
 *      @rec_inv_sqrt: reciprocal value of sqrt(count) >> 1
 *      @first_above_time: when we went (or will go) continuously above target
 *      @mark_next: time to mark next packet, or when we marked last
 *      @ldelay: sojourn time of last dequeued packet
 */
struct wfq_class
{
        u8		id;
	u8		prio;
        u32             len_bytes;
        struct Qdisc    *qdisc;

        u64             head_fin_time;

        u32             count;
        u32             lastcount;
        bool            marking;
        u16             rec_inv_sqrt;
        codel_time_t    first_above_time;
        codel_time_t    mark_next;
        codel_time_t    ldelay;
};

/**
 *      struct wfq_sched_data - WFQ scheduler
 *      @queues: multiple Class of Service (CoS) queues
 *      @rate: shaping rate
 *      @watchdog: watchdog timer for token bucket rate limiter
 *
 *      @tokens: tokens in ns
 *      @time_ns: time check-point
 *      @sum_len_bytes: the total buffer occupancy (in bytes) of the switch port
 *      @prio_len_bytes: buffer occupancy (in bytes) for different priorities
 *      @virtual_time: virtual system time of WFQ scheduler. We maintain a
 *      virtual system time for each priority.
 */
struct wfq_sched_data
{
        struct wfq_class        queues[wfq_max_queues];
        struct wfq_rate_cfg     rate;
        struct qdisc_watchdog   watchdog;

        s64     tokens;
        s64	time_ns;
        u32     sum_len_bytes;
        u32	prio_len_bytes[wfq_max_prio];
        u64     virtual_time[wfq_max_prio];
};

static inline void print_wfq_sched_data(struct Qdisc *sch)
{
        int i;
	struct wfq_sched_data *q = qdisc_priv(sch);

        printk(KERN_INFO "==========================================");
        printk(KERN_INFO "sch_wfq on %s\n", sch->dev_queue->dev->name);
        printk(KERN_INFO "rate: %llu Mbps\n", q->rate.rate_bps / 1000000);
        printk(KERN_INFO "total buffer occupancy: %u\n", q->sum_len_bytes);

        printk(KERN_INFO "==========================================");
        printk(KERN_INFO "per-queue buffer occupancy\n");
        for (i = 0; i < wfq_max_queues; i++)
                printk(KERN_INFO " queue %d: %u\n", i, q->queues[i].len_bytes);

        printk(KERN_INFO "==========================================");
        printk(KERN_INFO "per-priority buffer occupancy\n");
        for (i = 0; i < wfq_max_prio; i++)
                printk(KERN_INFO " priority %d: %u\n", i, q->prio_len_bytes[i]);

        printk(KERN_INFO "==========================================");
        printk(KERN_INFO "per-priority virtual system time\n");
        for (i = 0; i < wfq_max_prio; i++)
                printk(KERN_INFO " priority %d: %llu\n", i, q->virtual_time[i]);

        printk(KERN_INFO "==========================================");
}

/* return true if time1 is before (smaller) time2 */
static inline bool wfq_time_before(u64 time1, u64 time2)
{
    u64 thresh = (u64)1 << 63;

    if (time1 < time2 && time2 - time1 <= thresh)
        return true;
    else if (time1 > time2 && time1 - time2 > thresh)
        return true;
    else
        return false;
}

/* nanosecond to codel time (1 << wfq_codel_shift ns) */
static inline codel_time_t ns_to_codel_time(s64 ns)
{
	return ns >> wfq_codel_shift;
}

/*
 * We use this function to account for the true number of bytes sent on wire.
 * 20 = frame check sequence(8B)+Interpacket gap(12B)
 * 4 = Frame check sequence (4B)
 * wfq_min_pkt_bytes = Minimum Ethernet frame size (64B)
 */
static inline unsigned int skb_size(struct sk_buff *skb)
{
	return max_t(unsigned int, skb->len + 4, wfq_min_pkt_bytes) + 20;
}

/* Borrow from ptb */
static inline void precompute_ratedata(struct wfq_rate_cfg *r)
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
static inline u64 l2t_ns(struct wfq_rate_cfg *r, unsigned int len_bytes)
{
    return ((u64)len_bytes * r->mult) >> r->shift;
}

/* Queue length based ECN marking: per-queue abd  per-port */
void wfq_qlen_marking(struct sk_buff *skb,
                      struct wfq_sched_data *q,
		      struct wfq_class *cl)
{
	switch (wfq_ecn_scheme)
	{
		/* Per-queue ECN marking */
		case wfq_queue_ecn:
		{
			if (cl->len_bytes > wfq_queue_thresh_bytes[cl->id])
				INET_ECN_set_ce(skb);
			break;
		}
		/* Per-port ECN marking */
		case wfq_port_ecn:
		{
			if (q->sum_len_bytes > wfq_port_thresh_bytes)
				INET_ECN_set_ce(skb);
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

        if (codel_time_after(delay, (codel_time_t)wfq_tcn_thresh))
                INET_ECN_set_ce(skb);
}

/* Borrow from codel_should_drop in Linux kernel */
static bool codel_should_mark(const struct sk_buff *skb,
	                      struct wfq_class *cl,
			      s64 now_ns)
{
        bool ok_to_mark;
        codel_time_t now = ns_to_codel_time(now_ns);

        cl->ldelay = ns_to_codel_time(now_ns - skb->tstamp.tv64);

	if (codel_time_before(cl->ldelay, (codel_time_t)wfq_codel_target) ||
	    cl->len_bytes <= wfq_max_pkt_bytes)
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
                 cl->first_above_time = now + wfq_codel_interval;
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
static void codel_Newton_step(struct wfq_class *cl)
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
static void codel_marking(struct sk_buff *skb, struct wfq_class *cl)
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
					  	          wfq_codel_interval,
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
 				      (codel_time_t)wfq_codel_interval * 16))
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
 						  wfq_codel_interval,
 						  cl->rec_inv_sqrt);
	}
}


static struct wfq_class *wfq_classify(struct sk_buff *skb, struct Qdisc *sch)
{
        int i, dscp;
	struct wfq_sched_data *q = qdisc_priv(sch);
	struct iphdr* iph = ip_hdr(skb);

        if (unlikely(!(q->queues)))
                return NULL;

        /* Return queue[0] by default*/
        if (unlikely(!iph))
                return &(q->queues[0]);

	dscp = iph->tos >> 2;

	for (i = 0; i < wfq_max_queues; i++)
	{
                if(dscp == wfq_queue_dscp[i])
                        return &(q->queues[i]);
	}

	return &(q->queues[0]);
}

/* We don't need this */
static struct sk_buff *wfq_peek(struct Qdisc *sch)
{
    return NULL;
}

/* Decide whether the packet can be transmitted according to Token Bucket */
static s64 tbf_schedule(unsigned int len, struct wfq_sched_data *q, s64 now)
{
	s64 pkt_ns, toks;

	toks = now - q->time_ns;
	toks = min_t(s64, toks, (s64)l2t_ns(&q->rate, wfq_bucket_bytes));
	toks += q->tokens;

	pkt_ns = (s64)l2t_ns(&q->rate, len);

	return toks - pkt_ns;
}

/* Find the highest priority that is non-empty */
int prio_schedule(struct wfq_sched_data *q)
{
	int i;

	for (i = 0; i < wfq_max_prio; i++)
	{
		if (q->prio_len_bytes[i] > 0)
			return i;
	}

	return -1;
}

static struct sk_buff *wfq_dequeue(struct Qdisc *sch)
{
        struct wfq_sched_data *q = qdisc_priv(sch);
        int i, weight;
        struct wfq_class *cl = NULL;
        u64 min_time;
        struct sk_buff *skb = NULL;
        struct sk_buff *next_pkt = NULL;
        unsigned int len;
        s64 bucket_ns = (s64)l2t_ns(&q->rate, wfq_bucket_bytes);
        s64 result, now;
        int prio = prio_schedule(q);

        if (prio < 0)
                return NULL;

        /* Find the active queue with the smallest head finish time */
        for (i = 0; i < wfq_max_queues; i++)
        {
                if (q->queues[i].prio != prio || q->queues[i].len_bytes == 0 )
                        continue;

                if (!cl || wfq_time_before(q->queues[i].head_fin_time,
                                           min_time))
                {
                        cl = &q->queues[i];
                        min_time = cl->head_fin_time;
                }
        }

        /* get head packet */
        skb = cl->qdisc->ops->peek(cl->qdisc);
        if (unlikely(!skb))
                return NULL;

        len = skb_size(skb);
        now = ktime_get_ns();
        result = tbf_schedule(len, q, now);

        /* We don't have enough tokens */
        if (result < 0)
        {
                /* For hrtimer absolute mode, we use now + t */
                qdisc_watchdog_schedule_ns(&q->watchdog, now - result, true);
                qdisc_qstats_overlimit(sch);
                return NULL;
        }


        skb = qdisc_dequeue_peeked(cl->qdisc);
        if (unlikely(!skb))
                return NULL;

        q->sum_len_bytes -= len;
        sch->q.qlen--;
        cl->len_bytes -= len;
        q->prio_len_bytes[prio] -= len;

        /* Set the head_fin_time for the remaining head packet */
        if (cl->len_bytes > 0)
        {
                /* Get the current head packet */
                next_pkt = cl->qdisc->ops->peek(cl->qdisc);
                weight = wfq_queue_weight[cl->id];
                if (likely(next_pkt && weight))
                {
                        len = skb_size(next_pkt);
                        cl->head_fin_time += div_u64((u64)len, (u32)weight);
                        if (wfq_time_before(q->virtual_time[prio],
                                            cl->head_fin_time))
                                q->virtual_time[prio] = cl->head_fin_time;
                }
        }

        /* Bucket */
        q->time_ns = now;
        q->tokens = min_t(s64, result, bucket_ns);
        qdisc_unthrottled(sch);
        qdisc_bstats_update(sch, skb);

        /* TCN */
        if (wfq_ecn_scheme == wfq_tcn)
                tcn_marking(skb);
        /* CoDel */
        else if (wfq_ecn_scheme == wfq_codel)
                codel_marking(skb, cl);
        /* dequeue equeue length based ECN marking */
        else if (wfq_enable_dequeue_ecn == wfq_enable)
                wfq_qlen_marking(skb, q, cl);

        return skb;
}

static bool wfq_buffer_overfill(unsigned int len,
				 struct wfq_class *cl,
				 struct wfq_sched_data *q)
{
	/* per-port shared buffer */
	if (wfq_buffer_mode == wfq_shared_buffer &&
	    q->sum_len_bytes + len > wfq_shared_buffer_bytes)
		return true;
	/* per-queue static buffer */
	else if (wfq_buffer_mode == wfq_static_buffer &&
		 cl->len_bytes + len > wfq_queue_buffer_bytes[cl->id])
		return true;
	else
		return false;
}

static int wfq_enqueue(struct sk_buff *skb, struct Qdisc *sch)
{
        struct wfq_class *cl = NULL;
	unsigned int len = skb_size(skb);
	struct wfq_sched_data *q = qdisc_priv(sch);
	int ret, weight;


	cl = wfq_classify(skb, sch);
	/* No appropriate queue or the switch buffer is overfilled */
	if (unlikely(!cl) || wfq_buffer_overfill(len, cl, q))
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

	/* If the queue is empty, calculate its head finish time */
	if (cl->qdisc->q.qlen == 1)
	{
                weight = wfq_queue_weight[cl->id];
                /* We only change the priority when the queue is empty */
                cl->prio = (u8)wfq_queue_prio[cl->id];

                if (likely(weight > 0))
                {
                        cl->head_fin_time = div_u64((u64)len, (u32)weight) +
                                            q->virtual_time[cl->prio];
                        q->virtual_time[cl->prio] = cl->head_fin_time;

                }
	}

        /* Update queue sizes */
	sch->q.qlen++;
	q->sum_len_bytes += len;
	cl->len_bytes += len;
        q->prio_len_bytes[cl->prio] += len;

	/* sojourn time based ECN marking: TCN and CoDel */
	if (wfq_ecn_scheme == wfq_tcn || wfq_ecn_scheme == wfq_codel)
		skb->tstamp = ktime_get();
	/* enqueue queue length based ECN marking */
	else if (wfq_enable_dequeue_ecn == wfq_disable)
		wfq_qlen_marking(skb, q, cl);

	return ret;
}


/* We don't need this */
static unsigned int wfq_drop(struct Qdisc *sch)
{
    return 0;
}

/* We don't need this */
static int wfq_dump(struct Qdisc *sch, struct sk_buff *skb)
{
    return 0;
}

/* Release Qdisc resources */
static void wfq_destroy(struct Qdisc *sch)
{
        struct wfq_sched_data *q = qdisc_priv(sch);
        int i;

        if (likely(q->queues))
        {
                for (i = 0; i < wfq_max_queues && (q->queues[i]).qdisc; i++)
                        qdisc_destroy((q->queues[i]).qdisc);
	}
	qdisc_watchdog_cancel(&q->watchdog);
        printk(KERN_INFO "destroy sch_wfq on %s\n", sch->dev_queue->dev->name);
        print_wfq_sched_data(sch);
}

static const struct nla_policy wfq_policy[TCA_TBF_MAX + 1] = {
	[TCA_TBF_PARMS] = { .len = sizeof(struct tc_tbf_qopt) },
	[TCA_TBF_RTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
	[TCA_TBF_PTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
};

/* We only leverage TC netlink interface to configure rate */
static int wfq_change(struct Qdisc *sch, struct nlattr *opt)
{
        int err;
	struct wfq_sched_data *q = qdisc_priv(sch);
	struct nlattr *tb[TCA_TBF_PTAB + 1];
	struct tc_tbf_qopt *qopt;
	__u32 rate;

	err = nla_parse_nested(tb, TCA_TBF_PTAB, opt, wfq_policy);
	if(err < 0)
		return err;

	err = -EINVAL;
	if (tb[TCA_TBF_PARMS] == NULL)
		goto done;

	qopt = nla_data(tb[TCA_TBF_PARMS]);
	rate = qopt->rate.rate;
	/* convert from bytes/s to b/s */
	q->rate.rate_bps = (u64)rate << 3;
        precompute_ratedata(&q->rate);
	err = 0;

        printk(KERN_INFO "change sch_wfq on %s\n", sch->dev_queue->dev->name);
        print_wfq_sched_data(sch);
 done:
	return err;
}

/* Initialize Qdisc */
static int wfq_init(struct Qdisc *sch, struct nlattr *opt)
{
	int i;
	struct wfq_sched_data *q = qdisc_priv(sch);
	struct Qdisc *child;

	if(sch->parent != TC_H_ROOT)
		return -EOPNOTSUPP;

        q->tokens = 0;
        q->time_ns = ktime_get_ns();
        q->sum_len_bytes = 0;
	qdisc_watchdog_init(&q->watchdog, sch);

        /* Initialize per-priority variables */
	for (i = 0; i < wfq_max_prio; i++)
	{
		q->prio_len_bytes[i] = 0;
		q->virtual_time[i] = 0;
	}

	/* Initialize per-queue variables */
	for (i = 0; i < wfq_max_queues; i++)
	{
		/* bfifo is in bytes */
		child = fifo_create_dflt(sch,
                                         &bfifo_qdisc_ops,
                                         wfq_max_buffer_bytes);
		if (likely(child))
			(q->queues[i]).qdisc = child;
		else
			goto err;

                (q->queues[i]).id = i;
		(q->queues[i]).head_fin_time = 0;
                (q->queues[i]).len_bytes = 0;
                (q->queues[i]).count = 0;
                (q->queues[i]).lastcount = 0;
                (q->queues[i]).marking = false;
                (q->queues[i]).rec_inv_sqrt = 0;
                (q->queues[i]).first_above_time = 0;
                (q->queues[i]).mark_next = 0;
                (q->queues[i]).ldelay = 0;
	}

	return wfq_change(sch, opt);
err:
	wfq_destroy(sch);
	return -ENOMEM;
}

static struct Qdisc_ops wfq_ops __read_mostly = {
	.next          =       NULL,
	.cl_ops        =       NULL,
	.id            =       "tbf",
	.priv_size     =       sizeof(struct wfq_sched_data),
	.init          =       wfq_init,
	.destroy       =       wfq_destroy,
	.enqueue       =       wfq_enqueue,
	.dequeue       =       wfq_dequeue,
	.peek          =       wfq_peek,
	.drop          =       wfq_drop,
	.change        =       wfq_change,
	.dump          =       wfq_dump,
	.owner         =       THIS_MODULE,
};

static int __init wfq_module_init(void)
{
	if (unlikely(!wfq_params_init()))
		return -1;

	printk(KERN_INFO "sch_wfq: start working\n");
	return register_qdisc(&wfq_ops);
}

static void __exit wfq_module_exit(void)
{
	wfq_params_exit();
	unregister_qdisc(&wfq_ops);
	printk(KERN_INFO "sch_wfq: stop working\n");
}

module_init(wfq_module_init);
module_exit(wfq_module_exit);
MODULE_LICENSE("GPL");
