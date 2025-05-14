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

struct prio_wfq_rate_cfg
{
    u64 rate_bps;	//bit per second
    u32 mult;
    u32 shift;
};

/* struct of priority queue */
struct prio_class
{
	int id;	//id of this queue
	struct Qdisc *qdisc;	//inner FIFO queue
	u32 len_bytes;	//queue length in bytes
};

/* struct of WFQ queue */
struct wfq_class
{
    int id; //id of this queue
    struct Qdisc *qdisc;    //inner FIFO queue
    u64 head_finish_time;   //virtual finish time of the head packet
    u32 len_bytes;  //queue length in bytes
};

struct prio_wfq_sched_data
{
/* Parameters */
    struct wfq_class *wfq_queues;   //WFQ queues
    struct prio_class *prio_queues;	//priority queues
    struct prio_wfq_rate_cfg rate;

/* Variables */
    s64 tokens; //tokens in nanoseconds
    u32 sum_len_bytes;  //sum of queue length in bytes
    u32 sum_prio_len_bytes; //The sum of length of all priority queues in bytes
    u64 virtual_time;   //virtual time
    s64	time_ns;    //time check-point
    struct Qdisc *sch;
    struct qdisc_watchdog watchdog; //watchdog timer
};

/* return true if time1 is before (smaller) time2 */
static inline bool timestamp_before(u64 time1, u64 time2)
{
    u64 thresh = (u64)(1)<<63;

    if (time1 < time2 && time2 - time1 <= thresh)
        return true;
    else if (time1 > time2 && time1 - time2 > thresh)
        return true;
    else
        return false;
}

/* return larger (later) timestamp. e.g. if time1 is before time2, return time 2 */
static inline u64 max_timestamp(u64 time1, u64 time2)
{
    if (timestamp_before(time1, time2))
        return time2;
    else
        return time1;
}

/*
 * We use this function to account for the true number of bytes sent on wire.
 * 20=frame check sequence(8B)+Interpacket gap(12B)
 * 4=Frame check sequence (4B)
 * WFQ_QDISC_MIN_PKT_BYTES=Minimum Ethernet frame size (64B)
 */
static inline unsigned int skb_size(struct sk_buff *skb)
{
    return max_t(unsigned int, skb->len + 4, PRIO_WFQ_QDISC_MIN_PKT_BYTES) + 20;
}

/* Borrow from ptb */
static inline void prio_wfq_qdisc_precompute_ratedata(struct prio_wfq_rate_cfg *r)
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
static inline u64 l2t_ns(struct prio_wfq_rate_cfg *r, unsigned int len_bytes)
{
    return ((u64)len_bytes * r->mult) >> r->shift;
}

static inline void prio_wfq_qdisc_ecn(struct sk_buff *skb)
{
    if (skb_make_writable(skb, sizeof(struct iphdr)) && ip_hdr(skb))
        IP_ECN_set_ce(ip_hdr(skb));
}

/* return queue ID (-1 if no matched queue) */
static int prio_wfq_qdisc_classify(struct sk_buff *skb, struct Qdisc *sch)
{
    int i = 0;
	struct prio_wfq_sched_data *q = qdisc_priv(sch);
	struct iphdr* iph = ip_hdr(skb);
	int dscp;

    if (unlikely(!(q->wfq_queues) && !(q->prio_queues)))
		return -1;

    /* Return 0 by default*/
    if (unlikely(!iph))
        return 0;

	dscp = (const int)(iph->tos >> 2);

	for (i = 0; i < PRIO_WFQ_QDISC_MAX_QUEUES; i++)
    {
        if (dscp == PRIO_WFQ_QDISC_QUEUE_DSCP[i])
            return i;
	}

	return 0;
}

/* We don't need this */
static struct sk_buff* prio_wfq_qdisc_peek(struct Qdisc *sch)
{
    return NULL;
}

/* Peek the first packet from priority queues */
static struct sk_buff* prio_queues_peek(struct Qdisc *sch)
{
	struct prio_wfq_sched_data *q = qdisc_priv(sch);
	struct Qdisc *qdisc = NULL;
	struct sk_buff *skb = NULL;
	int i;

	/* If priority queues are not empty */
	if (q->sum_prio_len_bytes > 0)
	{
		for (i = 0; i < PRIO_WFQ_QDISC_MAX_PRIO_QUEUES && q->prio_queues[i].qdisc; i++)
		{
			qdisc = q->prio_queues[i].qdisc;
			skb = qdisc->ops->peek(qdisc);
			if (skb)
				return skb;
		}
	}

	return NULL;
}

static struct sk_buff* prio_queues_dequeue_peeked(struct Qdisc *sch)
{
	struct prio_wfq_sched_data *q = qdisc_priv(sch);
	struct Qdisc *qdisc = NULL;
	struct sk_buff *skb = NULL;
	int i;

	/* If priority queues are not empty */
	if (likely(q->sum_prio_len_bytes > 0))
	{
		for (i = 0; i < PRIO_WFQ_QDISC_MAX_PRIO_QUEUES && q->prio_queues[i].qdisc; i++)
		{
			qdisc = q->prio_queues[i].qdisc;
			skb = qdisc_dequeue_peeked(qdisc);
			if (skb)
			{
				q->prio_queues[i].len_bytes -= skb_size(skb);	//update per-queue buffer occupancy
				return skb;
			}
		}
	}

	return NULL;
}

/* Dequeue a packet from priority queues */
static struct sk_buff* prio_queues_dequeue(struct Qdisc *sch)
{
	struct prio_wfq_sched_data *q = qdisc_priv(sch);
	struct sk_buff *skb = NULL;
	unsigned int len;

	skb = prio_queues_peek(sch);
	if (skb)
	{
		s64 now = ktime_get_ns();
		s64 toks = min_t(s64, now - q->time_ns, PRIO_WFQ_QDISC_BUCKET_NS) + q->tokens;
		len = skb_size(skb);
		toks -= (s64)l2t_ns(&q->rate, len);

		//If we have enough tokens to release this packet
		if (toks >= 0)
		{
			skb = prio_queues_dequeue_peeked(sch);
			if (unlikely(!skb))
				return NULL;

			q->time_ns = now;
			q->sum_len_bytes -= len;
			q->sum_prio_len_bytes -= len;
			sch->q.qlen--;
			q->tokens = toks;

			//Bucket.
			if (q->tokens > PRIO_WFQ_QDISC_BUCKET_NS)
				q->tokens = PRIO_WFQ_QDISC_BUCKET_NS;

			if (PRIO_WFQ_QDISC_ECN_SCHEME == PRIO_WFQ_QDISC_DEQUE_ECN && skb->tstamp.tv64 > 0)
			{
				s64 sojourn_ns = now - skb->tstamp.tv64;
				s64 thresh_ns = (s64)l2t_ns(&q->rate, PRIO_WFQ_QDISC_PORT_THRESH_BYTES);

				if (sojourn_ns > thresh_ns)
				{
					prio_wfq_qdisc_ecn(skb);
					if (PRIO_WFQ_QDISC_DEBUG_MODE)
						printk(KERN_INFO "Sample sojurn time %lld ns > ECN marking threshold %lld ns (%d bytes)\n", sojourn_ns, thresh_ns, PRIO_WFQ_QDISC_PORT_THRESH_BYTES);
				}
			}

			qdisc_unthrottled(sch);
			qdisc_bstats_update(sch, skb);
			//printk(KERN_INFO "sch_prio_wfq: dequeue a packet with len=%u\n", len);
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

/* Dequeue a packet from WFQ queues */
static struct sk_buff* wfq_queues_dequeue(struct Qdisc *sch)
{
    struct prio_wfq_sched_data *q = qdisc_priv(sch);
    int i = 0;
    int min_index = -1;
    u64 min_time = 0;
    struct sk_buff *skb = NULL;
    struct sk_buff *next_pkt = NULL;
    unsigned int len;
    s64 now, toks, pkt_ns;

    if (q->sum_len_bytes == q->sum_prio_len_bytes)
        return NULL;

    for (i = 0; i < PRIO_WFQ_QDISC_MAX_WFQ_QUEUES; i++)
    {
        if (q->wfq_queues[i].len_bytes > 0 && (min_index < 0 || timestamp_before(q->wfq_queues[i].head_finish_time, min_time)))
        {
            min_index = i;
            min_time = q->wfq_queues[i].head_finish_time;
        }
    }

    if (unlikely(min_index == -1))
    {
        if (PRIO_WFQ_QDISC_DEBUG_MODE == PRIO_WFQ_QDISC_DEBUG_ON)
            printk(KERN_INFO "Not work conserving\n");
        return NULL;
    }

    /* get head packet */
    skb = q->wfq_queues[min_index].qdisc->ops->peek(q->wfq_queues[min_index].qdisc);
    if (unlikely(!skb))
    {
        qdisc_warn_nonwc(__func__, q->wfq_queues[min_index].qdisc);
        return NULL;
    }

    len = skb_size(skb);
    if (unlikely(len > PRIO_WFQ_QDISC_MTU_BYTES))
        printk(KERN_INFO "Error: packet length %u is larger than MTU\n", len);

    now = ktime_get_ns();
    toks = min_t(s64, now - q->time_ns, PRIO_WFQ_QDISC_BUCKET_NS) + q->tokens;
    pkt_ns = (s64)l2t_ns(&q->rate, len);

    if (toks > pkt_ns)
    {
        skb = qdisc_dequeue_peeked(q->wfq_queues[min_index].qdisc);
        if (unlikely(!skb))
            return NULL;

        /* Print necessary information in debug mode */
        /*if (PRIO_WFQ_QDISC_DEBUG_MODE == PRIO_WFQ_QDISC_DEBUG_ON)
        {
            printk(KERN_INFO "total buffer occupancy %u\n", q->sum_len_bytes);
            printk(KERN_INFO "queue %d buffer occupancy %u\n", min_index, q->queues[min_index].len_bytes);
        }*/
        q->sum_len_bytes -= len;
        sch->q.qlen--;
        q->wfq_queues[min_index].len_bytes -= len;

        /* Set the head_finish_time for the remaining head packet in the queue */
        if (q->wfq_queues[min_index].len_bytes > 0)
        {
            if (likely(PRIO_WFQ_QDISC_QUEUE_WEIGHT[min_index] > 0))
            {
                /* Get the current head packet */
                next_pkt = q->wfq_queues[min_index].qdisc->ops->peek(q->wfq_queues[min_index].qdisc);
                if (likely(next_pkt))
                {
                    q->wfq_queues[min_index].head_finish_time = q->wfq_queues[min_index].head_finish_time + skb_size(next_pkt) / PRIO_WFQ_QDISC_QUEUE_WEIGHT[min_index];
                    q->virtual_time = max_timestamp(q->virtual_time, q->wfq_queues[min_index].head_finish_time);
                    //printk(KERN_INFO "Virtual timestamp is %lld\n", q->virtual_time);
                }
            }
        }

        /* Dequeue latency-based ECN marking */
        if (PRIO_WFQ_QDISC_ECN_SCHEME == PRIO_WFQ_QDISC_DEQUE_ECN && skb->tstamp.tv64 > 0)
        {
            s64 sojourn_ns = ktime_get().tv64 - skb->tstamp.tv64;
            s64 thresh_ns = (s64)l2t_ns(&q->rate, PRIO_WFQ_QDISC_PORT_THRESH_BYTES);

            if (sojourn_ns > thresh_ns)
            {
                prio_wfq_qdisc_ecn(skb);
                if (PRIO_WFQ_QDISC_DEBUG_MODE == PRIO_WFQ_QDISC_DEBUG_ON)
                    printk(KERN_INFO "Sample sojurn time %lld > ECN marking threshold %lld\n", sojourn_ns, thresh_ns);
            }
        }
        //printk(KERN_INFO "Dequeue from queue %d\n", min_index);
        /* Bucket */
        q->time_ns = now;
        q->tokens = min_t(s64, toks - pkt_ns, PRIO_WFQ_QDISC_BUCKET_NS);
        qdisc_unthrottled(sch);
        qdisc_bstats_update(sch, skb);
        return skb;
    }
    else
    {
        /* we use now+t due to absolute mode of hrtimer (HRTIMER_MODE_ABS) */
        qdisc_watchdog_schedule_ns(&q->watchdog, now + pkt_ns - toks, true);
        qdisc_qstats_overlimit(sch);
        return NULL;
    }
}

static struct sk_buff* prio_wfq_qdisc_dequeue(struct Qdisc *sch)
{
	struct prio_wfq_sched_data *q = qdisc_priv(sch);

	//dequeue from priority queues
	if (q->sum_prio_len_bytes > 0)
		return prio_queues_dequeue(sch);
	//dequeue from WFQ queues
	else if (q->sum_len_bytes > 0)
		return wfq_queues_dequeue(sch);
	else
		return NULL;
}


static int prio_wfq_qdisc_enqueue(struct sk_buff *skb, struct Qdisc *sch)
{
	struct prio_wfq_sched_data *q = qdisc_priv(sch);
    struct prio_class *prio_queue = NULL;
    struct wfq_class *wfq_queue = NULL;
    unsigned int len = skb_size(skb);
    int id = 0;
	int ret;

	id = prio_wfq_qdisc_classify(skb, sch);
    if (id >= 0)
	{
		if (id < PRIO_WFQ_QDISC_MAX_PRIO_QUEUES)
			prio_queue = &(q->prio_queues[id]);
		else
			wfq_queue = &(q->wfq_queues[id - PRIO_WFQ_QDISC_MAX_PRIO_QUEUES]);
	}

	/* No appropriate queue or per port shared buffer is overfilled or per queue static buffer is overfilled */
	if (id < 0
	|| (PRIO_WFQ_QDISC_BUFFER_MODE == PRIO_WFQ_QDISC_SHARED_BUFFER && q->sum_len_bytes + len > PRIO_WFQ_QDISC_SHARED_BUFFER_BYTES)
	|| (PRIO_WFQ_QDISC_BUFFER_MODE == PRIO_WFQ_QDISC_STATIC_BUFFER && prio_queue && prio_queue->len_bytes + len > PRIO_WFQ_QDISC_QUEUE_BUFFER_BYTES[id])
    || (PRIO_WFQ_QDISC_BUFFER_MODE == PRIO_WFQ_QDISC_STATIC_BUFFER && wfq_queue && wfq_queue->len_bytes + len > PRIO_WFQ_QDISC_QUEUE_BUFFER_BYTES[id]))
	{
        if (prio_queue)
			qdisc_qstats_drop(prio_queue->qdisc);
		else if (wfq_queue)
			qdisc_qstats_drop(wfq_queue->qdisc);

		qdisc_qstats_drop(sch);
		kfree_skb(skb);
		return NET_XMIT_DROP;
	}
    /* The packet is enqueued to a priority queue */
    else if (prio_queue)
    {
        ret = qdisc_enqueue(skb, prio_queue->qdisc);
		if (ret == NET_XMIT_SUCCESS)
		{
			/* Update queue sizes */
			sch->q.qlen++;
			q->sum_len_bytes += len;
			q->sum_prio_len_bytes += len;
			prio_queue->len_bytes += len;

            /* Per-queue ECN marking */
            if (PRIO_WFQ_QDISC_ECN_SCHEME == PRIO_WFQ_QDISC_QUEUE_ECN && prio_queue->len_bytes > PRIO_WFQ_QDISC_QUEUE_THRESH_BYTES[prio_queue->id])
                //printk(KERN_INFO "ECN marking\n");
                prio_wfq_qdisc_ecn(skb);
            /* Per-port ECN marking */
            else if (PRIO_WFQ_QDISC_ECN_SCHEME == PRIO_WFQ_QDISC_PORT_ECN && q->sum_len_bytes > PRIO_WFQ_QDISC_PORT_THRESH_BYTES)
                prio_wfq_qdisc_ecn(skb);
			else if (PRIO_WFQ_QDISC_ECN_SCHEME == PRIO_WFQ_QDISC_DEQUE_ECN)
                //Get enqueue time stamp
                skb->tstamp = ktime_get();
		}
		else if (net_xmit_drop_count(ret))
		{
			qdisc_qstats_drop(sch);
			qdisc_qstats_drop(prio_queue->qdisc);
		}

		return ret;
    }
    /* The packet is enqueued to a WFQ queue */
	else
	{
        ret = qdisc_enqueue(skb, wfq_queue->qdisc);
        if (ret == NET_XMIT_SUCCESS)
        {
            if (wfq_queue->len_bytes == 0 && likely(PRIO_WFQ_QDISC_QUEUE_WEIGHT[id - PRIO_WFQ_QDISC_MAX_PRIO_QUEUES] > 0))
            {
                wfq_queue->head_finish_time = q->virtual_time + len / PRIO_WFQ_QDISC_QUEUE_WEIGHT[id - PRIO_WFQ_QDISC_MAX_PRIO_QUEUES];
                q->virtual_time = wfq_queue->head_finish_time;
            }

			/* Update queue sizes */
			sch->q.qlen++;
			q->sum_len_bytes += len;
			wfq_queue->len_bytes += len;

            /* Per-queue ECN marking */
            if (PRIO_WFQ_QDISC_ECN_SCHEME == PRIO_WFQ_QDISC_QUEUE_ECN && wfq_queue->len_bytes > PRIO_WFQ_QDISC_QUEUE_THRESH_BYTES[wfq_queue->id])
                //printk(KERN_INFO "ECN marking\n");
                prio_wfq_qdisc_ecn(skb);
            /* Per-port ECN marking */
            else if (PRIO_WFQ_QDISC_ECN_SCHEME == PRIO_WFQ_QDISC_PORT_ECN && q->sum_len_bytes > PRIO_WFQ_QDISC_PORT_THRESH_BYTES)
                prio_wfq_qdisc_ecn(skb);
			else if (PRIO_WFQ_QDISC_ECN_SCHEME == PRIO_WFQ_QDISC_DEQUE_ECN)
                //Get enqueue time stamp
                skb->tstamp = ktime_get();
		}
		else
		{
			if (net_xmit_drop_count(ret))
			{
				qdisc_qstats_drop(sch);
				qdisc_qstats_drop(wfq_queue->qdisc);
			}
		}
		return ret;
	}
}

/* We don't need this */
static unsigned int prio_wfq_qdisc_drop(struct Qdisc *sch)
{
    return 0;
}

/* We don't need this */
static int prio_wfq_qdisc_dump(struct Qdisc *sch, struct sk_buff *skb)
{
    return 0;
}

/* Release Qdisc resources */
static void prio_wfq_qdisc_destroy(struct Qdisc *sch)
{
	struct prio_wfq_sched_data *q = qdisc_priv(sch);
	int i;

	if (likely(q->wfq_queues))
	{
		for (i = 0; i < PRIO_WFQ_QDISC_MAX_WFQ_QUEUES && (q->wfq_queues[i]).qdisc; i++)
			qdisc_destroy((q->wfq_queues[i]).qdisc);

		kfree(q->wfq_queues);
	}

	if (likely(q->prio_queues))
	{
		for (i = 0; i < PRIO_WFQ_QDISC_MAX_PRIO_QUEUES && (q->prio_queues[i]).qdisc; i++)
			qdisc_destroy((q->prio_queues[i]).qdisc);

		kfree(q->prio_queues);
	}

	qdisc_watchdog_cancel(&q->watchdog);
}

static const struct nla_policy prio_wfq_qdisc_policy[TCA_TBF_MAX + 1] = {
	[TCA_TBF_PARMS] = { .len = sizeof(struct tc_tbf_qopt) },
	[TCA_TBF_RTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
	[TCA_TBF_PTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
};

/* We only leverage TC netlink interface to configure rate */
static int prio_wfq_qdisc_change(struct Qdisc *sch, struct nlattr *opt)
{
	int err;
	struct prio_wfq_sched_data *q = qdisc_priv(sch);
	struct nlattr *tb[TCA_TBF_PTAB + 1];
	struct tc_tbf_qopt *qopt;
	__u32 rate;

	err = nla_parse_nested(tb, TCA_TBF_PTAB, opt, prio_wfq_qdisc_policy);
	if(err < 0)
		return err;

	err = -EINVAL;
	if (tb[TCA_TBF_PARMS] == NULL)
		goto done;

	qopt = nla_data(tb[TCA_TBF_PARMS]);
	rate = qopt->rate.rate;
	/* convert from bytes/s to b/s */
	q->rate.rate_bps = (u64)rate << 3;
	prio_wfq_qdisc_precompute_ratedata(&q->rate);
	err = 0;
	printk(KERN_INFO "sch_prio_wfq: rate %llu Mbps\n", q->rate.rate_bps/1000000);

 done:
	return err;
}

/* Initialize Qdisc */
static int prio_wfq_qdisc_init(struct Qdisc *sch, struct nlattr *opt)
{
	int i;
	struct prio_wfq_sched_data *q = qdisc_priv(sch);
	struct Qdisc *child;

	if(sch->parent != TC_H_ROOT)
		return -EOPNOTSUPP;

	q->tokens = 0;
	q->time_ns = ktime_get_ns();
    q->virtual_time = 0;
	q->sum_len_bytes = 0;  //Total buffer occupation
    q->sum_prio_len_bytes = 0;	//Total buffer occupation of priority queues
	q->sch = sch;
	qdisc_watchdog_init(&q->watchdog, sch);

    q->prio_queues = kcalloc(PRIO_WFQ_QDISC_MAX_PRIO_QUEUES, sizeof(struct prio_class), GFP_KERNEL);
	q->wfq_queues = kcalloc(PRIO_WFQ_QDISC_MAX_WFQ_QUEUES, sizeof(struct wfq_class), GFP_KERNEL);
	if (!(q->wfq_queues) || !(q->prio_queues))
		return -ENOMEM;

    /* Initialize priority queues */
    for (i = 0; i < PRIO_WFQ_QDISC_MAX_PRIO_QUEUES; i++)
    {
        /* bfifo is in bytes */
    	child = fifo_create_dflt(sch, &bfifo_qdisc_ops, PRIO_WFQ_QDISC_MAX_BUFFER_BYTES);
    	if (child)
    		(q->prio_queues[i]).qdisc = child;
    	else
    		goto err;

    	(q->prio_queues[i]).id = i;
    	(q->prio_queues[i]).len_bytes = 0;
    }

	/* Initialize WFQ queues */
	for (i = 0; i < PRIO_WFQ_QDISC_MAX_WFQ_QUEUES; i++)
	{
		/* bfifo is in bytes */
		child = fifo_create_dflt(sch, &bfifo_qdisc_ops, PRIO_WFQ_QDISC_MAX_BUFFER_BYTES);
		if (child)
			(q->wfq_queues[i]).qdisc = child;
		else
			goto err;

		/* Initialize variables for wfq_class */
        (q->wfq_queues[i]).id = i + PRIO_WFQ_QDISC_MAX_PRIO_QUEUES;
		(q->wfq_queues[i]).head_finish_time = 0;
        (q->wfq_queues[i]).len_bytes = 0;
	}
	return prio_wfq_qdisc_change(sch, opt);
err:
	prio_wfq_qdisc_destroy(sch);
	return -ENOMEM;
}

static struct Qdisc_ops prio_wfq_qdisc_ops __read_mostly = {
	.next = NULL,
	.cl_ops = NULL,
	.id = "tbf",
	.priv_size = sizeof(struct prio_wfq_sched_data),
	.init = prio_wfq_qdisc_init,
	.destroy = prio_wfq_qdisc_destroy,
	.enqueue = prio_wfq_qdisc_enqueue,
	.dequeue = prio_wfq_qdisc_dequeue,
	.peek = prio_wfq_qdisc_peek,
	.drop = prio_wfq_qdisc_drop,
	.change = prio_wfq_qdisc_change,
	.dump = prio_wfq_qdisc_dump,
	.owner = THIS_MODULE,
};

static int __init prio_wfq_qdisc_module_init(void)
{
	if (prio_wfq_qdisc_params_init() < 0)
		return -1;

	printk(KERN_INFO "sch_prio_wfq: start working\n");
	return register_qdisc(&prio_wfq_qdisc_ops);
}

static void __exit prio_wfq_qdisc_module_exit(void)
{
	prio_wfq_qdisc_params_exit();
	unregister_qdisc(&prio_wfq_qdisc_ops);
	printk(KERN_INFO "sch_prio_wfq: stop working\n");
}

module_init(prio_wfq_qdisc_module_init);
module_exit(prio_wfq_qdisc_module_exit);
MODULE_LICENSE("GPL");
