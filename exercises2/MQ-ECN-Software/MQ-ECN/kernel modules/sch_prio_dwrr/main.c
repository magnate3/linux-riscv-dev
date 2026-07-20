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

struct prio_dwrr_rate_cfg
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

/* struct of DWRR queue */
struct dwrr_class
{
	int id; //id of this queue
	struct Qdisc *qdisc;	//inner FIFO queue
	u32	deficitCounter;	//deficit counter of this queue (bytes)
	u8 active;	//whether the queue is not ampty (1) or not (0)
	u8 curr;	//whether this queue is crr being served
	u32 len_bytes;	//queue length in bytes
	s64 start_time_ns;	//time when this queue is inserted to active list
	s64 last_pkt_time_ns;	//time when this queue transmits the last packet
	s64 last_pkt_len_ns;	//length of last packet/rate
	u32 quantum;	//quantum of this queue
	struct list_head alist;	//structure of active link list
};

struct prio_dwrr_sched_data
{
/* Parameters */
	struct dwrr_class *dwrr_queues;	//DWRR queues
	struct prio_class *prio_queues;	//priority queues
	struct prio_dwrr_rate_cfg rate;	//rate
	struct list_head activeList;	//The head point of link list for active DWRR queues

/* Variables */
	s64 tokens;	//Tokens in nanoseconds
	u32 sum_len_bytes;	//The sum of lengh of all queues in bytes
	u32 sum_prio_len_bytes;	//The sum of length of all priority queues in bytes
	struct Qdisc *sch;
	s64	time_ns;	//Time check-point
	struct qdisc_watchdog watchdog;	//Watchdog timer
	s64 round_time_ns;	//Estimation of round time
	s64 last_idle_time_ns;	//Last idle time
	u32 quantum_sum;	//Quantum sum of all active queues
	u32 quantum_sum_estimate;	//Estimation of quantums aum of all active queues
};

/*
 * We use this function to account for the true number of bytes sent on wire.
 * 20=frame check sequence(8B)+Interpacket gap(12B)
 * 4=Frame check sequence (4B)
 * DWRR_QDISC_MIN_PKT_BYTES=Minimum Ethernet frame size (64B)
 */
static inline unsigned int skb_size(struct sk_buff *skb)
{
	return max_t(unsigned int, skb->len + 4, PRIO_DWRR_QDISC_MIN_PKT_BYTES) + 20;
}

/* Borrow from ptb */
static inline void prio_dwrr_qdisc_precompute_ratedata(struct prio_dwrr_rate_cfg *r)
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
static inline u64 l2t_ns(struct prio_dwrr_rate_cfg *r, unsigned int len_bytes)
{
	return ((u64)len_bytes * r->mult) >> r->shift;
}

static inline void prio_dwrr_qdisc_ecn(struct sk_buff *skb)
{
	if (skb_make_writable(skb, sizeof(struct iphdr))&&ip_hdr(skb))
		IP_ECN_set_ce(ip_hdr(skb));
}

/* return queue ID (-1 if no matched queue) */
static int prio_dwrr_qdisc_classify(struct sk_buff *skb, struct Qdisc *sch)
{
	int i = 0;
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	struct iphdr* iph = ip_hdr(skb);
	int dscp;

	if (unlikely(!(q->dwrr_queues) && !(q->prio_queues)))
		return -1;

	/* Return 0 by default*/
	if (unlikely(!iph))
		return 0;

	dscp = (const int)(iph->tos >> 2);

	for (i = 0; i < PRIO_DWRR_QDISC_MAX_QUEUES; i++)
	{
		if (dscp == PRIO_DWRR_QDISC_QUEUE_DSCP[i])
			return i;
	}

	return 0;
}

/* We don't need this */
static struct sk_buff* prio_dwrr_qdisc_peek(struct Qdisc *sch)
{
	return NULL;
}

/* Peek the first packet from priority queues */
static struct sk_buff* prio_queues_peek(struct Qdisc *sch)
{
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	struct Qdisc *qdisc = NULL;
	struct sk_buff *skb = NULL;
	int i;

	/* If priority queues are not empty */
	if (q->sum_prio_len_bytes > 0)
	{
		for (i = 0; i < PRIO_DWRR_QDISC_MAX_PRIO_QUEUES && q->prio_queues[i].qdisc; i++)
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
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	struct Qdisc *qdisc = NULL;
	struct sk_buff *skb = NULL;
	int i;

	/* If priority queues are not empty */
	if (likely(q->sum_prio_len_bytes > 0))
	{
		for (i = 0; i < PRIO_DWRR_QDISC_MAX_PRIO_QUEUES && q->prio_queues[i].qdisc; i++)
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
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	struct sk_buff *skb = NULL;
	unsigned int len;

	skb = prio_queues_peek(sch);
	if (skb)
	{
		s64 now = ktime_get_ns();
		s64 toks = min_t(s64, now - q->time_ns, PRIO_DWRR_QDISC_BUCKET_NS) + q->tokens;
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
			if (q->tokens > PRIO_DWRR_QDISC_BUCKET_NS)
				q->tokens = PRIO_DWRR_QDISC_BUCKET_NS;

			if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_DEQUE_ECN && skb->tstamp.tv64 > 0)
			{
				s64 sojourn_ns = now - skb->tstamp.tv64;
				s64 thresh_ns = (s64)l2t_ns(&q->rate, PRIO_DWRR_QDISC_PORT_THRESH_BYTES);

				if (sojourn_ns > thresh_ns)
				{
					prio_dwrr_qdisc_ecn(skb);
					if (PRIO_DWRR_QDISC_DEBUG_MODE)
						printk(KERN_INFO "Sample sojurn time %lld ns > ECN marking threshold %lld ns (%d bytes)\n", sojourn_ns, thresh_ns, PRIO_DWRR_QDISC_PORT_THRESH_BYTES);
				}
			}

			qdisc_unthrottled(sch);
			qdisc_bstats_update(sch, skb);
			//printk(KERN_INFO "sch_prio_dwrr: dequeue a packet with len=%u\n", len);
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

/* dequeue from DWRR queues */
static struct sk_buff* dwrr_queues_dequeue(struct Qdisc *sch)
{
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	struct dwrr_class *cl = NULL;
	struct sk_buff *skb = NULL;
	s64 sample_ns = 0;
	unsigned int len;

	/* No active DWRR queue */
	if (list_empty(&q->activeList))
		return NULL;

	/* dequeue from DWRR queues */
	while (1)
	{
		cl = list_first_entry(&q->activeList, struct dwrr_class, alist);
		if (unlikely(!cl))
			return NULL;

		/* update deficit counter for this round*/
		if (cl->curr == 0)
		{
			cl->curr = 1;
			cl->deficitCounter += cl->quantum;
		}

		/* get head packet */
		skb = cl->qdisc->ops->peek(cl->qdisc);
		if (unlikely(!skb))
		{
			qdisc_warn_nonwc(__func__, cl->qdisc);
			return NULL;
		}

		len = skb_size(skb);
		if (unlikely(len > PRIO_DWRR_QDISC_MTU_BYTES))
			printk(KERN_INFO "Error: packet length %u is larger than MTU\n", len);

		/* If this packet can be scheduled by DWRR */
		if (len <= cl->deficitCounter)
		{
			s64 now = ktime_get_ns();
			s64 toks = min_t(s64, now - q->time_ns, PRIO_DWRR_QDISC_BUCKET_NS) + q->tokens;
			s64 pkt_ns = (s64)l2t_ns(&q->rate, len);

			/* If we have enough tokens to release this packet */
			if (toks > pkt_ns)
			{
				skb = qdisc_dequeue_peeked(cl->qdisc);
				if (unlikely(skb == NULL))
					return NULL;

				/* Print necessary information in debug mode */
				/*if (PRIO_DWRR_QDISC_DEBUG_MODE)
				{
					printk(KERN_INFO "total buffer occupancy %u\n", q->sum_len_bytes);
					printk(KERN_INFO "queue %d buffer occupancy %u\n", cl->id, cl->len_bytes);
				}*/
				q->sum_len_bytes -= len;
				sch->q.qlen--;
				cl->len_bytes -= len;
				cl->deficitCounter -= len;
				cl->last_pkt_len_ns = pkt_ns;
				cl->last_pkt_time_ns = ktime_get_ns();

				if (cl->qdisc->q.qlen == 0)
				{
					cl->active = 0;
					cl->curr = 0;
					list_del(&cl->alist);
					q->quantum_sum -= cl->quantum;
					sample_ns = max_t(s64, cl->last_pkt_time_ns - cl->start_time_ns, cl->last_pkt_len_ns);
					q->round_time_ns = (PRIO_DWRR_QDISC_ROUND_ALPHA * q->round_time_ns + (1000 - PRIO_DWRR_QDISC_ROUND_ALPHA) * sample_ns) / 1000;

					/* Get start time of idle period */
					if (q->sum_len_bytes == q->sum_prio_len_bytes)
						q->last_idle_time_ns = ktime_get_ns();

					/* Print necessary information in debug mode with MQ-ECN-RR*/
					if (PRIO_DWRR_QDISC_DEBUG_MODE && PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_RR)
					{
						printk(KERN_INFO "sample round time %llu \n", sample_ns);
						printk(KERN_INFO "round time %llu\n", q->round_time_ns);
					}
				}

				/* Update quantum_sum_estimate */
				q->quantum_sum_estimate = (PRIO_DWRR_QDISC_QUANTUM_ALPHA * q->quantum_sum_estimate + (1000 - PRIO_DWRR_QDISC_QUANTUM_ALPHA) * q->quantum_sum) / 1000;
				/* Print necessary information in debug mode with MQ-ECN-GENER*/
				if (PRIO_DWRR_QDISC_DEBUG_MODE && PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_GENER)
				{
					printk(KERN_INFO "sample quantum sum %u\n", q->quantum_sum);
					printk(KERN_INFO "quantum sum %u\n", q->quantum_sum_estimate);
				}

				/* Dequeue latency-based ECN marking */
				if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_DEQUE_ECN && skb->tstamp.tv64 > 0)
				{
					s64 sojourn_ns = ktime_get().tv64 - skb->tstamp.tv64;
					s64 thresh_ns = (s64)l2t_ns(&q->rate, PRIO_DWRR_QDISC_PORT_THRESH_BYTES);

					if (sojourn_ns > thresh_ns)
					{
						prio_dwrr_qdisc_ecn(skb);
						if (PRIO_DWRR_QDISC_DEBUG_MODE)
							printk(KERN_INFO "Sample sojurn time %lld > ECN marking threshold %lld\n", sojourn_ns, thresh_ns);
					}
				}
				//printk(KERN_INFO "Dequeue from queue %d\n",cl->id);
				/* Bucket */
				q->time_ns = now;
				q->tokens = min_t(s64,toks - pkt_ns, PRIO_DWRR_QDISC_BUCKET_NS);
				qdisc_unthrottled(sch);
				qdisc_bstats_update(sch, skb);
				return skb;
			}
			/* if we don't have enough tokens to realse this packet */
			else
			{
				/* we use now+t due to absolute mode of hrtimer (HRTIMER_MODE_ABS) */
				qdisc_watchdog_schedule_ns(&q->watchdog, now + pkt_ns - toks, true);
				qdisc_qstats_overlimit(sch);
				return NULL;
			}
		}
		/* This packet can not be scheduled by DWRR */
		else
		{
			cl->curr = 0;
			sample_ns = max_t(s64, cl->last_pkt_time_ns - cl->start_time_ns, cl->last_pkt_len_ns);
			q->round_time_ns = (PRIO_DWRR_QDISC_ROUND_ALPHA * q->round_time_ns + (1000 - PRIO_DWRR_QDISC_ROUND_ALPHA) * sample_ns) / 1000;
			cl->start_time_ns = ktime_get_ns();
			q->quantum_sum -= cl->quantum;
			cl->quantum = PRIO_DWRR_QDISC_QUEUE_QUANTUM[cl->id - PRIO_DWRR_QDISC_MAX_PRIO_QUEUES];
			q->quantum_sum += cl->quantum;
			list_move_tail(&cl->alist, &q->activeList);

			/* Print necessary information in debug mode with MQ-ECN-RR */
			if (PRIO_DWRR_QDISC_DEBUG_MODE && PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_RR)
			{
				printk(KERN_INFO "sample round time %llu\n", sample_ns);
				printk(KERN_INFO "round time %llu\n", q->round_time_ns);
			}
		}
	}

	return NULL;
}

static struct sk_buff* prio_dwrr_qdisc_dequeue(struct Qdisc *sch)
{
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);

	//dequeue from priority queues
	if (q->sum_prio_len_bytes > 0)
		return prio_queues_dequeue(sch);
	//dequeue from DWRR queues
	else if (!list_empty(&q->activeList))
		return dwrr_queues_dequeue(sch);
	else
		return NULL;
}

static int prio_dwrr_qdisc_enqueue(struct sk_buff *skb, struct Qdisc *sch)
{
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	int ret;
	u64 ecn_thresh_bytes = 0;
	s64 interval = ktime_get_ns() - q->last_idle_time_ns;
	s64 intervalNum = 0;
	int i = 0;
	int id = 0;
	struct prio_class *prio_queue = NULL;
	struct dwrr_class *dwrr_queue = NULL;
	unsigned int len = skb_size(skb);

	if (q->sum_len_bytes == 0 && (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_RR || PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_GENER))
	{
		if (PRIO_DWRR_QDISC_IDLE_INTERVAL_NS > 0)
		{
			intervalNum = interval / PRIO_DWRR_QDISC_IDLE_INTERVAL_NS;
			if (intervalNum <= PRIO_DWRR_QDISC_MAX_ITERATION)
			{
				for (i = 0; i < intervalNum; i++)
				{
					q->round_time_ns = q->round_time_ns * PRIO_DWRR_QDISC_ROUND_ALPHA / 1000;
					q->quantum_sum_estimate = q->quantum_sum_estimate * PRIO_DWRR_QDISC_QUANTUM_ALPHA / 1000;
				}
			}
			else
			{
				q->round_time_ns = 0;
				q->quantum_sum_estimate = 0;
			}
		}
		else
		{
			q->round_time_ns = 0;
			q->quantum_sum_estimate = 0;
		}
		if (PRIO_DWRR_QDISC_DEBUG_MODE)
		{
			if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_RR)
				printk(KERN_INFO "round time is set to %llu\n", q->round_time_ns);
			else
				printk(KERN_INFO "quantum sum is reset to %u\n", q->quantum_sum_estimate);
		}
	}

	id = prio_dwrr_qdisc_classify(skb, sch);
	if (id >= 0)
	{
		if (id < PRIO_DWRR_QDISC_MAX_PRIO_QUEUES)
			prio_queue = &(q->prio_queues[id]);
		else
			dwrr_queue = &(q->dwrr_queues[id - PRIO_DWRR_QDISC_MAX_PRIO_QUEUES]);
	}

	/* No appropriate queue or per port shared buffer is overfilled or per queue static buffer is overfilled */
	if (id < 0
	|| (PRIO_DWRR_QDISC_BUFFER_MODE == PRIO_DWRR_QDISC_SHARED_BUFFER && q->sum_len_bytes + len > PRIO_DWRR_QDISC_SHARED_BUFFER_BYTES)
	|| (PRIO_DWRR_QDISC_BUFFER_MODE == PRIO_DWRR_QDISC_STATIC_BUFFER && prio_queue && prio_queue->len_bytes + len > PRIO_DWRR_QDISC_QUEUE_BUFFER_BYTES[id])
	|| (PRIO_DWRR_QDISC_BUFFER_MODE == PRIO_DWRR_QDISC_STATIC_BUFFER && dwrr_queue && dwrr_queue->len_bytes + len > PRIO_DWRR_QDISC_QUEUE_BUFFER_BYTES[id]))
	{
		if (prio_queue)
			qdisc_qstats_drop(prio_queue->qdisc);
		else if (dwrr_queue)
			qdisc_qstats_drop(dwrr_queue->qdisc);

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

			/* Per-queue ECN marking
			 * MQ-ECN for any packet scheduling algorithm
             * MQ-ECN for round robin algorithms
			 */
			if ((PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_QUEUE_ECN ||
				PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_GENER ||
				PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_RR)
				&& prio_queue->len_bytes > PRIO_DWRR_QDISC_QUEUE_THRESH_BYTES[id])
				//printk(KERN_INFO "ECN marking\n");
				prio_dwrr_qdisc_ecn(skb);
			/* Per-port ECN marking */
			else if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_PORT_ECN && q->sum_len_bytes > PRIO_DWRR_QDISC_PORT_THRESH_BYTES)
				prio_dwrr_qdisc_ecn(skb);
			/* Dequeue latency-based ECN marking */
			else if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_DEQUE_ECN)
				skb->tstamp = ktime_get();
		}
		else if (net_xmit_drop_count(ret))
		{
			qdisc_qstats_drop(sch);
			qdisc_qstats_drop(prio_queue->qdisc);
		}

		return ret;
	}
	/* The packet is enqueued to a DWRR queue */
	else
	{
		ret = qdisc_enqueue(skb, dwrr_queue->qdisc);
		if (ret == NET_XMIT_SUCCESS)
		{
			/* Update queue sizes */
			sch->q.qlen++;
			q->sum_len_bytes += len;
			dwrr_queue->len_bytes += len;

			if (dwrr_queue->active == 0)
			{
				dwrr_queue->deficitCounter = 0;
				dwrr_queue->active = 1;
				dwrr_queue->curr = 0;
				dwrr_queue->start_time_ns = ktime_get_ns();
				dwrr_queue->quantum = PRIO_DWRR_QDISC_QUEUE_QUANTUM[id - PRIO_DWRR_QDISC_MAX_PRIO_QUEUES];
				list_add_tail(&(dwrr_queue->alist), &(q->activeList));
				q->quantum_sum += dwrr_queue->quantum;
			}

			/* Per-queue ECN marking */
			if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_QUEUE_ECN && dwrr_queue->len_bytes > PRIO_DWRR_QDISC_QUEUE_THRESH_BYTES[id])
				//printk(KERN_INFO "ECN marking\n");
				prio_dwrr_qdisc_ecn(skb);
			/* Per-port ECN marking */
			else if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_PORT_ECN && q->sum_len_bytes > PRIO_DWRR_QDISC_PORT_THRESH_BYTES)
				prio_dwrr_qdisc_ecn(skb);
			/* MQ-ECN for any packet scheduling algorithm */
			else if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_GENER)
			{
				if (q->quantum_sum_estimate > 0)
					ecn_thresh_bytes = min_t(u64, dwrr_queue->quantum * PRIO_DWRR_QDISC_PORT_THRESH_BYTES / q->quantum_sum_estimate, PRIO_DWRR_QDISC_PORT_THRESH_BYTES);
				else
					ecn_thresh_bytes = PRIO_DWRR_QDISC_PORT_THRESH_BYTES;

				if (dwrr_queue->len_bytes > ecn_thresh_bytes)
					prio_dwrr_qdisc_ecn(skb);

				if (PRIO_DWRR_QDISC_DEBUG_MODE)
					printk(KERN_INFO "queue %d quantum %u ECN threshold %llu\n", id, dwrr_queue->quantum, ecn_thresh_bytes);
			}
			/* MQ-ECN for round robin algorithms */
			else if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_MQ_ECN_RR)
			{
				if (q->round_time_ns > 0)
					ecn_thresh_bytes = min_t(u64, dwrr_queue->quantum * 8000000000 / q->round_time_ns, q->rate.rate_bps) * PRIO_DWRR_QDISC_PORT_THRESH_BYTES / q->rate.rate_bps;
				else
					ecn_thresh_bytes = PRIO_DWRR_QDISC_PORT_THRESH_BYTES;

				if (dwrr_queue->len_bytes > ecn_thresh_bytes)
					prio_dwrr_qdisc_ecn(skb);

				if (PRIO_DWRR_QDISC_DEBUG_MODE)
					printk(KERN_INFO "queue %d quantum %u ECN threshold %llu\n", dwrr_queue->id, dwrr_queue->quantum, ecn_thresh_bytes);
			}
			/* Dequeue latency-based ECN marking */
			else if (PRIO_DWRR_QDISC_ECN_SCHEME == PRIO_DWRR_QDISC_DEQUE_ECN)
				//Get enqueue time stamp
				skb->tstamp = ktime_get();
		}
		else
		{
			if (net_xmit_drop_count(ret))
			{
				qdisc_qstats_drop(sch);
				qdisc_qstats_drop(dwrr_queue->qdisc);
			}
		}
		return ret;
	}
}

/* We don't need this */
static unsigned int prio_dwrr_qdisc_drop(struct Qdisc *sch)
{
	return 0;
}

/* We don't need this */
static int prio_dwrr_qdisc_dump(struct Qdisc *sch, struct sk_buff *skb)
{
	return 0;
}

/* Release Qdisc resources */
static void prio_dwrr_qdisc_destroy(struct Qdisc *sch)
{
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	int i;

	if (likely(q->dwrr_queues))
	{
		for (i = 0; i < PRIO_DWRR_QDISC_MAX_DWRR_QUEUES && (q->dwrr_queues[i]).qdisc; i++)
			qdisc_destroy((q->dwrr_queues[i]).qdisc);

		kfree(q->dwrr_queues);
	}

	if (likely(q->prio_queues))
	{
		for (i = 0; i < PRIO_DWRR_QDISC_MAX_PRIO_QUEUES && (q->prio_queues[i]).qdisc; i++)
			qdisc_destroy((q->prio_queues[i]).qdisc);

		kfree(q->prio_queues);
	}

	qdisc_watchdog_cancel(&q->watchdog);
}

static const struct nla_policy prio_dwrr_qdisc_policy[TCA_TBF_MAX + 1] = {
	[TCA_TBF_PARMS] = { .len = sizeof(struct tc_tbf_qopt) },
	[TCA_TBF_RTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
	[TCA_TBF_PTAB]	= { .type = NLA_BINARY, .len = TC_RTAB_SIZE },
};

/* We only leverage TC netlink interface to configure rate */
static int prio_dwrr_qdisc_change(struct Qdisc *sch, struct nlattr *opt)
{
	int err;
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	struct nlattr *tb[TCA_TBF_PTAB + 1];
	struct tc_tbf_qopt *qopt;
	__u32 rate;

	err = nla_parse_nested(tb, TCA_TBF_PTAB, opt, prio_dwrr_qdisc_policy);
	if(err < 0)
		return err;

	err = -EINVAL;
	if (tb[TCA_TBF_PARMS] == NULL)
		goto done;

	qopt = nla_data(tb[TCA_TBF_PARMS]);
	rate = qopt->rate.rate;
	/* convert from bytes/s to b/s */
	q->rate.rate_bps = (u64)rate << 3;
	prio_dwrr_qdisc_precompute_ratedata(&q->rate);
	err = 0;
	printk(KERN_INFO "sch_prio_dwrr: rate %llu Mbps\n", q->rate.rate_bps/1000000);

 done:
	return err;
}

/* Initialize Qdisc */
static int prio_dwrr_qdisc_init(struct Qdisc *sch, struct nlattr *opt)
{
	int i;
	struct prio_dwrr_sched_data *q = qdisc_priv(sch);
	struct Qdisc *child;

	if(sch->parent != TC_H_ROOT)
		return -EOPNOTSUPP;

	q->tokens = 0;
	q->time_ns = ktime_get_ns();
	q->last_idle_time_ns = ktime_get_ns();
	q->sum_len_bytes = 0;	//Total buffer occupation
	q->sum_prio_len_bytes = 0;	//Total buffer occupation of priority queues
	q->round_time_ns = 0;	//Estimation of round time
	q->quantum_sum = 0;	//Quantum sum of all active queues
	q->quantum_sum_estimate = 0;	//Estimation of quantum sum of all active queues
	q->sch = sch;
	qdisc_watchdog_init(&q->watchdog, sch);
	INIT_LIST_HEAD(&(q->activeList));

	q->prio_queues = kcalloc(PRIO_DWRR_QDISC_MAX_PRIO_QUEUES, sizeof(struct prio_class), GFP_KERNEL);
	q->dwrr_queues = kcalloc(PRIO_DWRR_QDISC_MAX_DWRR_QUEUES, sizeof(struct dwrr_class), GFP_KERNEL);
	if (!(q->dwrr_queues) || !(q->prio_queues))
		return -ENOMEM;

	/* Initialize priority queues */
	for (i = 0; i < PRIO_DWRR_QDISC_MAX_PRIO_QUEUES; i++)
	{
		/* bfifo is in bytes */
		child = fifo_create_dflt(sch, &bfifo_qdisc_ops, PRIO_DWRR_QDISC_MAX_BUFFER_BYTES);
		if (child)
			(q->prio_queues[i]).qdisc = child;
		else
			goto err;

		(q->prio_queues[i]).id = i;
		(q->prio_queues[i]).len_bytes = 0;
	}

	/* Initialize DWRR queues */
	for (i = 0; i < PRIO_DWRR_QDISC_MAX_DWRR_QUEUES; i++)
	{
		/* bfifo is in bytes */
		child = fifo_create_dflt(sch, &bfifo_qdisc_ops, PRIO_DWRR_QDISC_MAX_BUFFER_BYTES);
		if (child)
			(q->dwrr_queues[i]).qdisc = child;
		else
			goto err;

		/* Initialize variables for dwrr_class */
		INIT_LIST_HEAD(&((q->dwrr_queues[i]).alist));
		(q->dwrr_queues[i]).id = i + PRIO_DWRR_QDISC_MAX_PRIO_QUEUES;
		(q->dwrr_queues[i]).deficitCounter = 0;
		(q->dwrr_queues[i]).active = 0;
		(q->dwrr_queues[i]).curr = 0;
		(q->dwrr_queues[i]).len_bytes = 0;
		(q->dwrr_queues[i]).start_time_ns = ktime_get_ns();
		(q->dwrr_queues[i]).last_pkt_time_ns = ktime_get_ns();
		(q->dwrr_queues[i]).last_pkt_len_ns = 0;
		(q->dwrr_queues[i]).quantum = 0;
	}

	return prio_dwrr_qdisc_change(sch,opt);
err:
	prio_dwrr_qdisc_destroy(sch);
	return -ENOMEM;
}

static struct Qdisc_ops prio_dwrr_qdisc_ops __read_mostly = {
	.next = NULL,
	.cl_ops = NULL,
	.id = "tbf",
	.priv_size = sizeof(struct prio_dwrr_sched_data),
	.init = prio_dwrr_qdisc_init,
	.destroy = prio_dwrr_qdisc_destroy,
	.enqueue = prio_dwrr_qdisc_enqueue,
	.dequeue = prio_dwrr_qdisc_dequeue,
	.peek = prio_dwrr_qdisc_peek,
	.drop = prio_dwrr_qdisc_drop,
	.change = prio_dwrr_qdisc_change,
	.dump = prio_dwrr_qdisc_dump,
	.owner = THIS_MODULE,
};

static int __init prio_dwrr_qdisc_module_init(void)
{
	if (prio_dwrr_qdisc_params_init() < 0)
		return -1;

	printk(KERN_INFO "sch_prio_dwrr: start working\n");
	return register_qdisc(&prio_dwrr_qdisc_ops);
}

static void __exit prio_dwrr_qdisc_module_exit(void)
{
	prio_dwrr_qdisc_params_exit();
	unregister_qdisc(&prio_dwrr_qdisc_ops);
	printk(KERN_INFO "sch_prio_dwrr: stop working\n");
}

module_init(prio_dwrr_qdisc_module_init);
module_exit(prio_dwrr_qdisc_module_exit);
MODULE_LICENSE("GPL");
