#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/netfilter.h>
#include <linux/netfilter_ipv4.h>
#include <linux/skbuff.h>
#include <linux/udp.h>
#include <linux/icmp.h>
#include <linux/ip.h>
#include <linux/inet.h>
#include <net/pkt_sched.h>
#include <linux/prandom.h> // PRANDOM_ADD_NOISE
#include<linux/circ_buf.h>
#include<net/pkt_cls.h>
#include<net/sch_generic.h>
#include <linux/dma-mapping.h>
#include <linux/platform_device.h>
#include "macb.h"
int dev_tx_weight __read_mostly = 64;
#define DIP "1.2.3.4"
#define TEST_XMIT 1
#define SKB_XOFF_MAGIC ((struct sk_buff *)1UL)
static struct nf_hook_ops local_out, local_in; 
static struct nf_hook_ops nfho;     // net filter hook option struct 
struct sk_buff *sock_buff;          // socket buffer used in linux kernel
struct udphdr *udp_header;          // udp header struct (not used)
struct iphdr *ip_header;            // ip header struct
struct ethhdr *mac_header;          // mac header struct


MODULE_DESCRIPTION("Redirect_Packet");
MODULE_AUTHOR("Andy Lee <a1106052000 AT gmail.com>");
MODULE_LICENSE("GPL");
#define skb_update_prio(skb)
/* Ring buffer accessors */
static unsigned int macb_tx_ring_wrap(struct macb *bp, unsigned int index)
{
		return index & (bp->tx_ring_size - 1);
}
unsigned int dbg_macb_tx_map(struct macb *bp, struct macb_queue *queue, struct sk_buff *skb, unsigned int hdrlen)
{
	 dma_addr_t mapping;
	 unsigned int offset, size;
	 struct macb_tx_skb *tx_skb = NULL;
	 unsigned int  entry, tx_head = queue->tx_head;
	/* first buffer length */
	 size = hdrlen;
	 offset = 0;
	 entry = macb_tx_ring_wrap(bp, tx_head);
      	 tx_skb = &queue->tx_skb[entry];
	 pr_info("dbg tx head %u and entry %u \n", tx_head, entry);
	 mapping = dma_map_single(&bp->pdev->dev, skb->data + offset, size, DMA_TO_DEVICE);
	 if (dma_mapping_error(&bp->pdev->dev, mapping))
	 {
	       pr_info("dma map error happens \n");
	 }
         else {
	        //tx_skb->skb = NULL;
		tx_skb->mapping = mapping;
		tx_skb->size = size;
	        tx_skb->mapped_as_page = false;
		dma_unmap_single(&bp->pdev->dev, tx_skb->mapping, tx_skb->size, DMA_TO_DEVICE);
	  }
	 return 0;
}
static int dbg_hardware_info(struct sk_buff *skb,struct net_device *dev)
{

	u16 queue_index = skb_get_queue_mapping(skb);
        struct macb *bp = netdev_priv(dev);
        struct macb_queue *queue = &bp->queues[queue_index];
	unsigned int hdrlen = min(skb_headlen(skb), bp->max_tx_length);
	if (CIRC_SPACE(queue->tx_head, queue->tx_tail, bp->tx_ring_size) < 1)
	{
	     pr_info("queue_index %u dma desc is no available \n", queue_index);
	}
	if(__netif_subqueue_stopped(bp->dev, queue_index))
	{
	     pr_info("netif subqueue stopped \n");
	}
	dbg_macb_tx_map(bp, queue, skb,hdrlen);
	return 0;
}
static int dbg_hardware_queue_common_info(struct net_device *dev,unsigned int queue_index)
{

        struct macb *bp = netdev_priv(dev);
        struct macb_queue *queue = &bp->queues[queue_index];
        pr_info("queue index %u, tx head %u and  tx tail %u, irq num:  %d \n", queue_index, queue->tx_head, queue->tx_tail, queue->irq);
	return 0;
}
static int dbg_hardware_queue_info(struct net_device *dev,unsigned int queue_index)
{

        struct macb *bp = netdev_priv(dev);
        struct macb_queue *queue = &bp->queues[queue_index];
	if (CIRC_SPACE(queue->tx_head, queue->tx_tail, bp->tx_ring_size) < 1)
	{
	     pr_info("dma desc is no available, tx head %u and  tx tail %u \n", queue->tx_head, queue->tx_tail);
	}
	if(__netif_subqueue_stopped(bp->dev, queue_index))
	{
	     pr_info("netif subqueue stopped \n");
	}
	return 0;
}
static void dbg_netif_tx_queues(struct net_device *dev)
{
	unsigned int i;
        struct macb *bp = netdev_priv(dev);
	pr_info("dev->num_tx_queues %u , bp->num_queues %u \n", dev->num_tx_queues,  bp->num_queues);
	for (i = 0; i < dev->num_tx_queues; i++) {
             struct netdev_queue *txq = netdev_get_tx_queue(dev, i);
	     dbg_hardware_queue_common_info(dev, i);
	     //if(netif_xmit_frozen_or_stopped(txq))
	     //if(test_bit(QUEUE_STATE_ANY_XOFF, &txq->state))
	     // dev_queue->state & QUEUE_STATE_ANY_XOFF
	     //if(netif_xmit_stopped(txq))
	     if(test_bit(__QUEUE_STATE_DRV_XOFF, &txq->state))
	     {
		    pr_info(" netif queue %u  stopped by drv \n", i );
		    dbg_hardware_queue_info(dev, i);
	     }

	     if(test_bit(__QUEUE_STATE_STACK_XOFF, &txq->state))
	     {
		    pr_info(" netif queue %u  stopped by stack \n", i );
	     }
	     if(test_bit(__QUEUE_STATE_FROZEN, &txq->state))
	     {
		    pr_info(" netif queue %u  frozen \n", i );
#if 0
		    clear_bit(__QUEUE_STATE_FROZEN, &txq->state);
		    netif_schedule_queue(txq);
#endif
	     }
	}
}
static int xmit_one(struct sk_buff *skb, struct net_device *dev,
		    struct netdev_queue *txq, bool more)
{
	unsigned int len;
	int rc;

	if (dev_nit_active(dev))
		dev_queue_xmit_nit(skb, dev); //  for tcpdump 

	len = skb->len;
	PRANDOM_ADD_NOISE(skb, dev, txq, len + jiffies);
	//trace_net_dev_start_xmit(skb, dev);
	dbg_hardware_info(skb, dev);
	rc = netdev_start_xmit(skb, dev, txq, more);
	//trace_net_dev_xmit(skb, rc, dev, len);

	return rc;
}
struct sk_buff *dev_hard_start_xmit(struct sk_buff *first, struct net_device *dev,
				    struct netdev_queue *txq, int *ret)
{
	struct sk_buff *skb = first;
	int rc = NETDEV_TX_OK;

	while (skb) {
		struct sk_buff *next = skb->next;

		skb_mark_not_on_list(skb);
		rc = xmit_one(skb, dev, txq, next != NULL);
		if (unlikely(!dev_xmit_complete(rc))) {
			skb->next = next;
			goto out;
		}

		skb = next;
		pr_info("%s find netif tx queue stop ?  %d \n",__func__, netif_tx_queue_stopped(txq));
		if (netif_tx_queue_stopped(txq) && skb) {
			rc = NETDEV_TX_BUSY;
			break;
		}
	}

out:
	pr_info("%s return  %d \n",__func__, NETDEV_TX_OK == rc);
	*ret = rc;
	return skb;
}

static inline void dev_requeue_skb(struct sk_buff *skb, struct Qdisc *q)
{
	spinlock_t *lock = NULL;

	if (q->flags & TCQ_F_NOLOCK) {
		lock = qdisc_lock(q);
		spin_lock(lock);
	}

	while (skb) {
		struct sk_buff *next = skb->next;

		__skb_queue_tail(&q->gso_skb, skb);

		/* it's still part of the queue */
		if (qdisc_is_percpu_stats(q)) {
			qdisc_qstats_cpu_requeues_inc(q);
			qdisc_qstats_cpu_backlog_inc(q, skb);
			qdisc_qstats_cpu_qlen_inc(q);
		} else {
			q->qstats.requeues++;
			qdisc_qstats_backlog_inc(q, skb);
			q->q.qlen++;
		}

		skb = next;
	}

	if (lock) {
		spin_unlock(lock);
		set_bit(__QDISC_STATE_MISSED, &q->state);
	} else {
		__netif_schedule(q);
	}
}
static inline void qdisc_enqueue_skb_bad_txq(struct Qdisc *q,
					     struct sk_buff *skb)
{
	spinlock_t *lock = NULL;

	if (q->flags & TCQ_F_NOLOCK) {
		lock = qdisc_lock(q);
		spin_lock(lock);
	}

	__skb_queue_tail(&q->skb_bad_txq, skb);

	if (qdisc_is_percpu_stats(q)) {
		qdisc_qstats_cpu_backlog_inc(q, skb);
		qdisc_qstats_cpu_qlen_inc(q);
	} else {
		qdisc_qstats_backlog_inc(q, skb);
		q->q.qlen++;
	}

	if (lock)
		spin_unlock(lock);
}
static void try_bulk_dequeue_skb(struct Qdisc *q,
				 struct sk_buff *skb,
				 const struct netdev_queue *txq,
				 int *packets)
{
	int bytelimit = qdisc_avail_bulklimit(txq) - skb->len;

	while (bytelimit > 0) {
		struct sk_buff *nskb = q->dequeue(q);

		if (!nskb)
			break;

		bytelimit -= nskb->len; /* covers GSO len */
		skb->next = nskb;
		skb = nskb;
		(*packets)++; /* GSO counts as one pkt */
	}
	skb_mark_not_on_list(skb);
}

/* This variant of try_bulk_dequeue_skb() makes sure
 * all skbs in the chain are for the same txq
 */
static void try_bulk_dequeue_skb_slow(struct Qdisc *q,
				      struct sk_buff *skb,
				      int *packets)
{
	int mapping = skb_get_queue_mapping(skb);
	struct sk_buff *nskb;
	int cnt = 0;

	do {
		nskb = q->dequeue(q);
		if (!nskb)
			break;
		if (unlikely(skb_get_queue_mapping(nskb) != mapping)) {
			qdisc_enqueue_skb_bad_txq(q, nskb);
			break;
		}
		skb->next = nskb;
		skb = nskb;
	} while (++cnt < 8);
	(*packets) += cnt;
	skb_mark_not_on_list(skb);
}
static void qdisc_maybe_clear_missed(struct Qdisc *q,
				     const struct netdev_queue *txq)
{
	clear_bit(__QDISC_STATE_MISSED, &q->state);

	/* Make sure the below netif_xmit_frozen_or_stopped()
	 * checking happens after clearing STATE_MISSED.
	 */
	smp_mb__after_atomic();

	/* Checking netif_xmit_frozen_or_stopped() again to
	 * make sure STATE_MISSED is set if the STATE_MISSED
	 * set by netif_tx_wake_queue()'s rescheduling of
	 * net_tx_action() is cleared by the above clear_bit().
	 */
	if (!netif_xmit_frozen_or_stopped(txq))
		set_bit(__QDISC_STATE_MISSED, &q->state);
	else
		set_bit(__QDISC_STATE_DRAINING, &q->state);
}
static inline struct sk_buff *__skb_dequeue_bad_txq(struct Qdisc *q)
{
	const struct netdev_queue *txq = q->dev_queue;
	spinlock_t *lock = NULL;
	struct sk_buff *skb;

	if (q->flags & TCQ_F_NOLOCK) {
		lock = qdisc_lock(q);
		spin_lock(lock);
	}

	skb = skb_peek(&q->skb_bad_txq);
	if (skb) {
		/* check the reason of requeuing without tx lock first */
		txq = skb_get_tx_queue(txq->dev, skb);
		if (!netif_xmit_frozen_or_stopped(txq)) {
			skb = __skb_dequeue(&q->skb_bad_txq);
			if (qdisc_is_percpu_stats(q)) {
				qdisc_qstats_cpu_backlog_dec(q, skb);
				qdisc_qstats_cpu_qlen_dec(q);
			} else {
				qdisc_qstats_backlog_dec(q, skb);
				q->q.qlen--;
			}
		} else {
			skb = SKB_XOFF_MAGIC;
			qdisc_maybe_clear_missed(q, txq);
		}
	}

	if (lock)
		spin_unlock(lock);

	return skb;
}

static inline struct sk_buff *qdisc_dequeue_skb_bad_txq(struct Qdisc *q)
{
	struct sk_buff *skb = skb_peek(&q->skb_bad_txq);

	if (unlikely(skb))
		skb = __skb_dequeue_bad_txq(q);

	return skb;
}
static struct sk_buff *dequeue_skb(struct Qdisc *q, bool *validate,
				   int *packets)
{
	const struct netdev_queue *txq = q->dev_queue;
	struct sk_buff *skb = NULL;

	*packets = 1;
	if (unlikely(!skb_queue_empty(&q->gso_skb))) {
		spinlock_t *lock = NULL;

		pr_info("%s enter branch1 \n",__func__);
		if (q->flags & TCQ_F_NOLOCK) {
			lock = qdisc_lock(q);
			spin_lock(lock);
		}

		skb = skb_peek(&q->gso_skb);

		/* skb may be null if another cpu pulls gso_skb off in between
		 * empty check and lock.
		 */
		if (!skb) {
			if (lock)
				spin_unlock(lock);
			goto validate;
		}

		/* skb in gso_skb were already validated */
		*validate = false;
#if 0
		if (xfrm_offload(skb))
			*validate = true;
#endif
		/* check the reason of requeuing without tx lock first */
		txq = skb_get_tx_queue(txq->dev, skb);
		if (!netif_xmit_frozen_or_stopped(txq)) {
			skb = __skb_dequeue(&q->gso_skb);
			if (qdisc_is_percpu_stats(q)) {
				qdisc_qstats_cpu_backlog_dec(q, skb);
				qdisc_qstats_cpu_qlen_dec(q);
			} else {
				qdisc_qstats_backlog_dec(q, skb);
				q->q.qlen--;
			}
		} else {
		        pr_info("%s enter branch1.2 xmit frozen\n",__func__);
			skb = NULL;
			qdisc_maybe_clear_missed(q, txq);
		}
		if (lock)
			spin_unlock(lock);
		goto trace;
	}
validate:
	*validate = true;
	if ((q->flags & TCQ_F_ONETXQUEUE) &&
	    netif_xmit_frozen_or_stopped(txq)) {
		pr_info("%s enter branch2 \n",__func__);
		qdisc_maybe_clear_missed(q, txq);
		return skb;
	}
	skb = qdisc_dequeue_skb_bad_txq(q);
	if (unlikely(skb)) {
		if (skb == SKB_XOFF_MAGIC)
		{
		        pr_info("%s enter branch3 \n",__func__);
			return NULL;
		}
		goto bulk;
	}
	skb = q->dequeue(q);
	pr_info("%s qdsic->ops  name %s dequeue skb== null: %d \n",__func__,q->ops->id, NULL == skb);
	if (skb) {
bulk:
		if (qdisc_may_bulk(q))
		{
		        pr_info("%s enter branch4 \n",__func__);
			try_bulk_dequeue_skb(q, skb, txq, packets);

		}
		else
		{
		        pr_info("%s enter branch5 \n",__func__);
			try_bulk_dequeue_skb_slow(q, skb, packets);
		}
	}
trace:
	//trace_qdisc_dequeue(q, txq, *packets, skb);
	return skb;
}
bool sch_direct_xmit(struct sk_buff *skb, struct Qdisc *q,
		     struct net_device *dev, struct netdev_queue *txq,
		     spinlock_t *root_lock, bool validate)
{
	int ret = NETDEV_TX_BUSY;
	bool again = false;

	/* And release qdisc */
	if (root_lock)
		spin_unlock(root_lock);

	/* Note that we validate skb (GSO, checksum, ...) outside of locks */
	if (validate)
		skb = validate_xmit_skb_list(skb, dev, &again);

#ifdef CONFIG_XFRM_OFFLOAD
	if (unlikely(again)) {
		if (root_lock)
			spin_lock(root_lock);

		dev_requeue_skb(skb, q);
		return false;
	}
#endif

	if (likely(skb)) {
		HARD_TX_LOCK(dev, txq, smp_processor_id());
		if (!netif_xmit_frozen_or_stopped(txq))
		{
			skb = dev_hard_start_xmit(skb, dev, txq, &ret);
			pr_info("%s call dev hard xmit ret %d \n",__func__, ret);
		}
		else
		{
			pr_info("%s qdisc clear miss \n",__func__);
			qdisc_maybe_clear_missed(q, txq);
                }
		HARD_TX_UNLOCK(dev, txq);
	} else {
		if (root_lock)
			spin_lock(root_lock);
		return true;
	}

	if (root_lock)
		spin_lock(root_lock);

	if (!dev_xmit_complete(ret)) {
		/* Driver returned NETDEV_TX_BUSY - requeue skb */
		if (unlikely(ret != NETDEV_TX_BUSY))
			net_warn_ratelimited("BUG %s code %d qlen %d\n",
					     dev->name, ret, q->q.qlen);

		dev_requeue_skb(skb, q);
		return false;
	}
        pr_info("%s do successfully  \n",__func__);
	return true;
}
static inline bool qdisc_restart(struct Qdisc *q, int *packets)
{
	spinlock_t *root_lock = NULL;
	struct netdev_queue *txq;
	struct net_device *dev;
	struct sk_buff *skb;
	bool validate;

	/* Dequeue packet */
	skb = dequeue_skb(q, &validate, packets);
	if (unlikely(!skb)) {
	        pr_info("%s dequeue null \n",__func__);
		return false;
        }
	pr_info("%s begin xmit \n",__func__);
	if (!(q->flags & TCQ_F_NOLOCK))
		root_lock = qdisc_lock(q);

	dev = qdisc_dev(q);
	txq = skb_get_tx_queue(dev, skb);

	return sch_direct_xmit(skb, q, dev, txq, root_lock, validate);
}

void __qdisc_run(struct Qdisc *q)
{
	int quota = dev_tx_weight;
	int packets;

	pr_info("%s run \n",__func__);
	while (qdisc_restart(q, &packets)) {
		quota -= packets;
		if (quota <= 0) {
			if (q->flags & TCQ_F_NOLOCK)
				set_bit(__QDISC_STATE_MISSED, &q->state);
			else
				__netif_schedule(q);

			break;
		}
	}
}
static int dev_qdisc_enqueue(struct sk_buff *skb, struct Qdisc *q,
		                             struct sk_buff **to_free,
					                                  struct netdev_queue *txq)
{
       int rc;
       rc = q->enqueue(skb, q, to_free) & NET_XMIT_MASK;
       return rc;
}
#if 0
static inline int __dev_xmit_skb(struct sk_buff *skb, struct Qdisc *q, struct net_device *dev, struct netdev_queue *txq)
{
	spinlock_t *root_lock = qdisc_lock(q);
	struct sk_buff *to_free = NULL;
        bool contended;
        int rc;
	contended = qdisc_is_running(q);
	if (unlikely(contended))
	     spin_lock(&q->busylock);
        spin_lock(root_lock);
        if (unlikely(test_bit(__QDISC_STATE_DEACTIVATED, &q->state))) {
		pr_info("qdsic dead \n");
         __qdisc_drop(skb, &to_free);
        rc = NET_XMIT_DROP;
	} else {
	       rc = dev_qdisc_enqueue(skb, q, &to_free, txq);
	       if (qdisc_run_begin(q)) {
	            if (unlikely(contended)) {
	                 spin_unlock(&q->busylock);
	                 contended = false;
	            }
	        __qdisc_run(q);
	        qdisc_run_end(q);
				                      }
        }
	spin_unlock(root_lock);
	if (unlikely(to_free))
	      kfree_skb_list(to_free);
	if (unlikely(contended))
	      spin_unlock(&q->busylock);
          return rc;
}
#else
static inline int __dev_xmit_skb(struct sk_buff *skb, struct Qdisc *q,
				 struct net_device *dev,
				 struct netdev_queue *txq)
{
	spinlock_t *root_lock = qdisc_lock(q);
	struct sk_buff *to_free = NULL;
	bool contended;
	int rc;

	qdisc_calculate_pkt_len(skb, q);

	if (q->flags & TCQ_F_NOLOCK) {
		if (q->flags & TCQ_F_CAN_BYPASS && nolock_qdisc_is_empty(q) &&
		    qdisc_run_begin(q)) {
		        pr_info("%s enter branch 1  \n",__func__);
			/* Retest nolock_qdisc_is_empty() within the protection
			 * of q->seqlock to protect from racing with requeuing.
			 */
			if (unlikely(!nolock_qdisc_is_empty(q))) {
				rc = dev_qdisc_enqueue(skb, q, &to_free, txq);
				__qdisc_run(q);
				qdisc_run_end(q);

		                pr_info("%s enter branch 1.1 and goto lock out \n",__func__);
				goto no_lock_out;
			}
                        
			qdisc_bstats_cpu_update(q, skb);
			if (sch_direct_xmit(skb, q, dev, txq, NULL, true) &&
			    !nolock_qdisc_is_empty(q))
				__qdisc_run(q);

			qdisc_run_end(q);
			return NET_XMIT_SUCCESS;
		}

		pr_info("%s return value %d \n",__func__, rc);
		rc = dev_qdisc_enqueue(skb, q, &to_free, txq);
		qdisc_run(q);

no_lock_out:
		pr_info("%s enter branch 1.2 and do free skb \n",__func__);
		if (unlikely(to_free))
			kfree_skb_list(to_free);
		return rc;
	}

	/*
	 * Heuristic to force contended enqueues to serialize on a
	 * separate lock before trying to get qdisc main lock.
	 * This permits qdisc->running owner to get the lock more
	 * often and dequeue packets faster.
	 */
	contended = qdisc_is_running(q);
	if (unlikely(contended))
		spin_lock(&q->busylock);

	spin_lock(root_lock);
	if (unlikely(test_bit(__QDISC_STATE_DEACTIVATED, &q->state))) {
	        pr_info("%s enter branch 2  \n",__func__);
		__qdisc_drop(skb, &to_free);
		rc = NET_XMIT_DROP;
	} else if ((q->flags & TCQ_F_CAN_BYPASS) && !qdisc_qlen(q) &&
		   qdisc_run_begin(q)) {
	        pr_info("%s enter branch 3 \n",__func__);
		/*
		 * This is a work-conserving queue; there are no old skbs
		 * waiting to be sent out; and the qdisc is not running -
		 * xmit the skb directly.
		 */

		qdisc_bstats_update(q, skb);

		if (sch_direct_xmit(skb, q, dev, txq, root_lock, true)) {
			if (unlikely(contended)) {
				spin_unlock(&q->busylock);
				contended = false;
			}
			__qdisc_run(q);
		}

		qdisc_run_end(q);
		rc = NET_XMIT_SUCCESS;
	} else {
	        pr_info("%s enter branch 4 \n",__func__);
		rc = dev_qdisc_enqueue(skb, q, &to_free, txq);
		if (qdisc_run_begin(q)) {
			if (unlikely(contended)) {
				spin_unlock(&q->busylock);
				contended = false;
			}
			__qdisc_run(q);
			qdisc_run_end(q);
		}
	}
	spin_unlock(root_lock);
	if (unlikely(to_free))
		kfree_skb_list(to_free);
	if (unlikely(contended))
		spin_unlock(&q->busylock);
	return rc;
}
#endif
static void qdisc_pkt_len_init(struct sk_buff *skb)
{
	 qdisc_skb_cb(skb)->pkt_len = skb->len;
}
#if 0
struct netdev_queue *netdev_core_pick_tx(struct net_device *dev, struct sk_buff *skb, struct net_device *sb_dev)
{
     int queue_index = 1;
     skb_set_queue_mapping(skb, queue_index);
     pr_info("%s queue_index %d, num_tx_queues %d\n", __func__, queue_index, dev->num_tx_queues);
     return netdev_get_tx_queue(dev, queue_index);
}
#else
struct netdev_queue *netdev_core_pick_tx(struct net_device *dev,
					 struct sk_buff *skb,
					 struct net_device *sb_dev)
{
	int queue_index = 0;

#ifdef CONFIG_XPS
	pr_info("%s encter branch 1\n", __func__);
	u32 sender_cpu = skb->sender_cpu - 1;

	if (sender_cpu >= (u32)NR_CPUS)
		skb->sender_cpu = raw_smp_processor_id() + 1;
#endif

	if (dev->real_num_tx_queues != 1) {
		const struct net_device_ops *ops = dev->netdev_ops;

		if (ops->ndo_select_queue)
		{
	                pr_info("%s encter branch 2\n", __func__);
			queue_index = ops->ndo_select_queue(dev, skb, sb_dev);
		}
		else
		{
	                pr_info("%s encter branch 3\n", __func__);
			queue_index = netdev_pick_tx(dev, skb, sb_dev);
                }
		queue_index = netdev_cap_txqueue(dev, queue_index);
	}

	pr_info("%s queue_index %d, num_tx_queues %d\n", __func__, queue_index, dev->num_tx_queues);
	skb_set_queue_mapping(skb, queue_index);
	return netdev_get_tx_queue(dev, queue_index);
}
#endif
static struct sk_buff *
sch_handle_egress(struct sk_buff *skb, int *ret, struct net_device *dev)
{
	struct mini_Qdisc *miniq = rcu_dereference_bh(dev->miniq_egress);
	struct tcf_result cl_res;

	if (!miniq)
		return skb;

	/* qdisc_skb_cb(skb)->pkt_len was already set by the caller. */
	qdisc_skb_cb(skb)->mru = 0;
	qdisc_skb_cb(skb)->post_ct = false;
	mini_qdisc_bstats_cpu_update(miniq, skb);

	switch (tcf_classify(skb, miniq->filter_list, &cl_res, false)) {
	case TC_ACT_OK:
	case TC_ACT_RECLASSIFY:
		skb->tc_index = TC_H_MIN(cl_res.classid);
		break;
	case TC_ACT_SHOT:
		mini_qdisc_qstats_cpu_drop(miniq);
		*ret = NET_XMIT_DROP;
		kfree_skb(skb);
		return NULL;
	case TC_ACT_STOLEN:
	case TC_ACT_QUEUED:
	case TC_ACT_TRAP:
		*ret = NET_XMIT_SUCCESS;
		consume_skb(skb);
		return NULL;
	case TC_ACT_REDIRECT:
		/* No need to push/pop skb's mac_header here on egress! */
#if 0
		skb_do_redirect(skb);
#else
		consume_skb(skb);
#endif
		pr_info("%s call  skb_do_redirect \n",__func__);
		*ret = NET_XMIT_SUCCESS;
		return NULL;
	default:
		break;
	}

	return skb;
}
int dbg_dev_queue_xmit(struct sk_buff *skb, struct net_device *sb_dev)
{
	struct net_device *dev = skb->dev;
	struct netdev_queue *txq;
	struct Qdisc *q;
	int rc = -ENOMEM;
	bool again = false;

	skb_reset_mac_header(skb);

	if (unlikely(skb_shinfo(skb)->tx_flags & SKBTX_SCHED_TSTAMP))
		__skb_tstamp_tx(skb, NULL, NULL, skb->sk, SCM_TSTAMP_SCHED);

	/* Disable soft irqs for various locks below. Also
	 * stops preemption for RCU.
	 */
	rcu_read_lock_bh();
#if TEST_XMIT
	skb_update_prio(skb);

	qdisc_pkt_len_init(skb);
#else
	skb_update_prio(skb);

	qdisc_pkt_len_init(skb);
#ifdef CONFIG_NET_CLS_ACT
	skb->tc_at_ingress = 0;
# ifdef CONFIG_NET_EGRESS
		pr_info("%s call  sch_handle_egress  \n",__func__);
		skb = sch_handle_egress(skb, &rc, dev);
		if (!skb)
		{
			pr_info(" sch_handle_egress cause  goto out \n");
			goto out;
		}
#if 0
	if (static_branch_unlikely(&egress_needed_key)) {
		skb = sch_handle_egress(skb, &rc, dev);
		if (!skb)
			goto out;
	}
#endif
# endif
#endif
#endif
	/* If device/qdisc don't need skb->dst, release it right now while
	 * its hot in this cpu cache.
	 */
	if (dev->priv_flags & IFF_XMIT_DST_RELEASE)
		skb_dst_drop(skb);
	else
		skb_dst_force(skb);
	txq = netdev_core_pick_tx(dev, skb, sb_dev);
	q = rcu_dereference_bh(txq->qdisc);

	if (q->enqueue) {
		rc = __dev_xmit_skb(skb, q, dev, txq);
		goto out;
	}
	else
	{
		pr_info("qdisc have no queue");
	}
out:
	rcu_read_unlock_bh();
	return rc;
}	   
#if 0
unsigned int hook_func(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
        sock_buff = skb;
        ip_header = (struct iphdr *)skb_network_header(sock_buff); //grab network header using accessor
        mac_header = (struct ethhdr *)skb_mac_header(sock_buff);

        if(!sock_buff) { return NF_DROP;}

        if (ip_header->protocol==IPPROTO_ICMP) { //icmp=1 udp=17 tcp=6
                printk(KERN_INFO "Got ICMP Reply packet and dropped it. \n");     //log we’ve got udp packet to /var/log/messages
		printk(KERN_INFO "src_ip: %pI4 \n", &ip_header->saddr);
        	printk(KERN_INFO "dst_ip: %pI4\n", &ip_header->daddr);
		ip_header->daddr = in_aton(DIP);
		printk(KERN_INFO "modified_dst_ip: %pI4\n", &ip_header->daddr);
        }
        return NF_ACCEPT;
}
#else
static void igb_tx_timeout(struct net_device *netdev)
{
	pr_info("** %s *** \n", __func__);
}
unsigned int hook_func(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{

	int ret=0;
	struct dst_entry *dst;
	struct net_device *dev;

	// 30:d0:42:fa:ae:11
	char mac[ETH_ALEN] = {0x30,0xd0,0x42,0xfa,0xae,0x11};
	//char mac[ETH_ALEN] = {0x48,0x57,0x02,0x64,0xea,0x1b};
        sock_buff = skb;
        ip_header = (struct iphdr *)skb_network_header(sock_buff); //grab network header using accessor
        mac_header = (struct ethhdr *)skb_mac_header(sock_buff);

        if(!sock_buff) { return NF_DROP;}

        if (ip_header->protocol==IPPROTO_ICMP) { //icmp=1 udp=17 tcp=6
                //printk(KERN_INFO "preroute Got ICMP  packet and print it. \n");     //log we’ve got udp packet to /var/log/messages
		//printk(KERN_INFO "src_ip: %pI4 \n", &ip_header->saddr);
        	//printk(KERN_INFO "dst_ip: %pI4\n", &ip_header->daddr);
		if ( NULL == (dst = skb_dst(skb)) || (NULL == (dev = dst->dev)))
		{
		      printk(KERN_INFO "****************dst dev is null *************\n");
		      return NF_ACCEPT;
		}
                skb->protocol = htons(ETH_P_IP);
		__skb_pull(skb, skb_network_offset(skb));
		ret = dev_hard_header(skb, dev, ntohs(skb->protocol),  mac, NULL, skb->len);
#if 0
		 const struct net_device_ops *ops = dev->netdev_ops;
		 ops->ndo_start_xmit(skb, dev);
#else
		//ret = dev_queue_xmit(skb);
		//dev->netdev_ops->tx_timeout = &igb_tx_timeout;
		dbg_netif_tx_queues(dev);
		ret = dbg_dev_queue_xmit(skb,NULL);
#endif
		printk(KERN_INFO "icmp dev_queue_xmit returned %d\n", ret);
		return NF_STOLEN;
        }
        return NF_ACCEPT;
}
unsigned int local_in_hook_func(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
        sock_buff = skb;
        ip_header = (struct iphdr *)skb_network_header(sock_buff); //grab network header using accessor
        mac_header = (struct ethhdr *)skb_mac_header(sock_buff);

        if(!sock_buff) { return NF_DROP;}

        if (ip_header->protocol==IPPROTO_ICMP) { //icmp=1 udp=17 tcp=6
                printk(KERN_INFO "local in Got ICMP Request packet and print it. \n");     //log we’ve got udp packet to /var/log/messages
		printk(KERN_INFO "src_ip: %pI4 \n", &ip_header->saddr);
        	printk(KERN_INFO "dst_ip: %pI4\n", &ip_header->daddr);
        }
        return NF_ACCEPT;
}
unsigned int local_out_hook_func(void *priv, struct sk_buff *skb, const struct nf_hook_state *state)
{
        sock_buff = skb;
        ip_header = (struct iphdr *)skb_network_header(sock_buff); //grab network header using accessor
        mac_header = (struct ethhdr *)skb_mac_header(sock_buff);

        if(!sock_buff) { return NF_DROP;}

        if (ip_header->protocol==IPPROTO_ICMP) { //icmp=1 udp=17 tcp=6
                printk(KERN_INFO "local out Got ICMP Reply packet and print it. \n");     //log we’ve got udp packet to /var/log/messages
		printk(KERN_INFO "src_ip: %pI4 \n", &ip_header->saddr);
        	printk(KERN_INFO "dst_ip: %pI4\n", &ip_header->daddr);
        }
        return NF_ACCEPT;
}
#endif 
//static int __init init_module()
int init_icmp_hook_module(void)
{
        nfho.hook = hook_func;
        nfho.hooknum = 4; //NF_IP_PRE_ROUTING=0(capture ICMP Request.)  NF_IP_POST_ROUTING=4(capture ICMP reply.)
        nfho.pf = PF_INET;//IPV4 packets
        nfho.priority = NF_IP_PRI_FIRST;//set to highest priority over all other hook functions
        nf_register_net_hook(&init_net, &nfho);
#if 0
        local_in.hook = local_in_hook_func;
        local_in.hooknum = NF_INET_LOCAL_IN; 
        local_in.pf = PF_INET;
        local_in.priority = NF_IP_PRI_FIRST;//set to highest priority over all other hook functions
        nf_register_net_hook(&init_net, &local_in);

        local_out.hook = local_out_hook_func;
        local_out.hooknum = NF_INET_LOCAL_OUT; 
        local_out.pf = PF_INET;
        local_out.priority = NF_IP_PRI_FIRST;//set to highest priority over all other hook functions
        nf_register_net_hook(&init_net, &local_out);
#endif
        printk(KERN_INFO "---------------------------------------\n");
        printk(KERN_INFO "Loading  kernel module...\n");
        return 0;

}
 
//static void __exit  cleanup_module()
void   cleanup_icmp_hook_module(void)
{
	printk(KERN_INFO "Cleaning up dropicmp module.\n");
        //nf_unregister_hook(&nfho);     
	nf_unregister_net_hook(&init_net, &nfho);
	//nf_unregister_net_hook(&init_net, &local_in);
	//nf_unregister_net_hook(&init_net, &local_out);
}

module_init(init_icmp_hook_module);
module_exit(cleanup_icmp_hook_module);
MODULE_LICENSE("GPL");
