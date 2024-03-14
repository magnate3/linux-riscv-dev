/*
 * Huawei HiNIC PCI Express Linux driver
 * Copyright(c) 2017 Huawei Technologies Co., Ltd
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * for more details.
 *
 */

#include <linux/kernel.h>
#include <linux/netdevice.h>
#include <linux/u64_stats_sync.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/pci.h>
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/slab.h>
#include <linux/interrupt.h>
#include <linux/skbuff.h>
#include <linux/smp.h>
#include <asm/byteorder.h>
#include <net/icmp.h>
#include <net/ip.h>
#include <linux/highmem.h>
#include <linux/skbuff.h>

#include "hinic_common.h"
#include "hinic_hw_if.h"
#include "hinic_hw_wqe.h"
#include "hinic_hw_wq.h"
#include "hinic_hw_qp.h"
#include "hinic_hw_dev.h"
#include "hinic_dev.h"
#include "hinic_tx.h"

#define TX_IRQ_NO_PENDING               0
#define TX_IRQ_NO_COALESC               0
#define TX_IRQ_NO_LLI_TIMER             0
#define TX_IRQ_NO_CREDIT                0
#define TX_IRQ_NO_RESEND_TIMER          0

#define CI_UPDATE_NO_PENDING            0
#define CI_UPDATE_NO_COALESC            0

#define HW_CONS_IDX(sq)         be16_to_cpu(*(u16 *)((sq)->hw_ci_addr))

#define MIN_SKB_LEN             64

/**
 * hinic_txq_clean_stats - Clean the statistics of specific queue
 * @txq: Logical Tx Queue
 **/
void hinic_txq_clean_stats(struct hinic_txq *txq)
{
	struct hinic_txq_stats *txq_stats = &txq->txq_stats;

	u64_stats_update_begin(&txq_stats->syncp);
	txq_stats->pkts    = 0;
	txq_stats->bytes   = 0;
	txq_stats->tx_busy = 0;
	txq_stats->tx_wake = 0;
	txq_stats->tx_dropped = 0;
	u64_stats_update_end(&txq_stats->syncp);
}

/**
 * hinic_txq_get_stats - get statistics of Tx Queue
 * @txq: Logical Tx Queue
 * @stats: return updated stats here
 **/
void hinic_txq_get_stats(struct hinic_txq *txq, struct hinic_txq_stats *stats)
{
	struct hinic_txq_stats *txq_stats = &txq->txq_stats;
	unsigned int start;

	u64_stats_update_begin(&stats->syncp);
	do {
		start = u64_stats_fetch_begin(&txq_stats->syncp);
		stats->pkts    = txq_stats->pkts;
		stats->bytes   = txq_stats->bytes;
		stats->tx_busy = txq_stats->tx_busy;
		stats->tx_wake = txq_stats->tx_wake;
		stats->tx_dropped = txq_stats->tx_dropped;
	} while (u64_stats_fetch_retry(&txq_stats->syncp, start));
	u64_stats_update_end(&stats->syncp);
}

/**
 * txq_stats_init - Initialize the statistics of specific queue
 * @txq: Logical Tx Queue
 **/
static void txq_stats_init(struct hinic_txq *txq)
{
	struct hinic_txq_stats *txq_stats = &txq->txq_stats;

	u64_stats_init(&txq_stats->syncp);
	hinic_txq_clean_stats(txq);
}

/**
 * tx_map_skb - dma mapping for skb and return sges
 * @nic_dev: nic device
 * @skb: the skb
 * @sges: returned sges
 *
 * Return 0 - Success, negative - Failure
 **/
static int tx_map_skb(struct hinic_dev *nic_dev, struct sk_buff *skb,
		      struct hinic_sge *sges)
{
	struct hinic_hwdev *hwdev = nic_dev->hwdev;
	struct hinic_hwif *hwif = hwdev->hwif;
	struct pci_dev *pdev = hwif->pdev;
	struct skb_frag_struct *frag;
	dma_addr_t dma_addr;
	int i, j;

	dma_addr = dma_map_single(&pdev->dev, skb->data, skb_headlen(skb),
				  DMA_TO_DEVICE);
	if (dma_mapping_error(&pdev->dev, dma_addr)) {
		dev_err(&pdev->dev, "Failed to map Tx skb data\n");
		return -EFAULT;
	}

	hinic_set_sge(&sges[0], dma_addr, skb_headlen(skb));
        pr_err("skb_shinfo(skb)->nr_frags %d" , skb_shinfo(skb)->nr_frags);

	for (i = 0 ; i < skb_shinfo(skb)->nr_frags; i++) {
		frag = &skb_shinfo(skb)->frags[i];

		dma_addr = skb_frag_dma_map(&pdev->dev, frag, 0,
					    skb_frag_size(frag),
					    DMA_TO_DEVICE);
		if (dma_mapping_error(&pdev->dev, dma_addr)) {
			dev_err(&pdev->dev, "Failed to map Tx skb frag\n");
			goto err_tx_map;
		}

		hinic_set_sge(&sges[i + 1], dma_addr, skb_frag_size(frag));
	}

	return 0;

err_tx_map:
	for (j = 0; j < i; j++)
		dma_unmap_page(&pdev->dev, hinic_sge_to_dma(&sges[j + 1]),
			       sges[j + 1].len, DMA_TO_DEVICE);

	dma_unmap_single(&pdev->dev, hinic_sge_to_dma(&sges[0]), sges[0].len,
			 DMA_TO_DEVICE);
	return -EFAULT;
}

/**
 * tx_unmap_skb - unmap the dma address of the skb
 * @nic_dev: nic device
 * @skb: the skb
 * @sges: the sges that are connected to the skb
 **/
static void tx_unmap_skb(struct hinic_dev *nic_dev, struct sk_buff *skb,
			 struct hinic_sge *sges)
{
	struct hinic_hwdev *hwdev = nic_dev->hwdev;
	struct hinic_hwif *hwif = hwdev->hwif;
	struct pci_dev *pdev = hwif->pdev;
	int i;

	for (i = 0; i < skb_shinfo(skb)->nr_frags ; i++)
		dma_unmap_page(&pdev->dev, hinic_sge_to_dma(&sges[i + 1]),
			       sges[i + 1].len, DMA_TO_DEVICE);

	dma_unmap_single(&pdev->dev, hinic_sge_to_dma(&sges[0]), sges[0].len,
			 DMA_TO_DEVICE);
}

void mydump_skb(const char *level, struct sk_buff* skb, bool full_pkt)
{
        struct sk_buff *frag_iter;
        struct skb_shared_info *sh = skb_shinfo(skb);
	struct net_device *dev = skb->dev;
	struct sock *sk = skb->sk;
	struct sk_buff *list_skb;
	bool has_mac, has_trans;
	int headroom, tailroom;
	int i, len, seg_len, page_count;

	if (full_pkt)
		len = skb->len;
	else
		len = min_t(int, skb->len, MAX_HEADER + 128);

	headroom = skb_headroom(skb);
	tailroom = skb_tailroom(skb);
	has_mac = skb_mac_header_was_set(skb);
	has_trans = skb_transport_header_was_set(skb);
	printk("%sskb len=%u headroom=%u headlen=%u tailroom=%u\n"
	       "mac=(%d,%d) net=(%d,%d) trans=%d\n"
	       "shinfo(txflags=%u nr_frags=%u gso(size=%hu type=%u segs=%hu))\n"
	       "csum(0x%x ip_summed=%u complete_sw=%u valid=%u level=%u)\n"
	       "hash(0x%x sw=%u l4=%u) proto=0x%04x pkttype=%u iif=%d\n",
	       level, skb->len, headroom, skb_headlen(skb), tailroom,
	       has_mac ? skb->mac_header : -1,
	       has_mac ? skb_mac_header_len(skb) : -1,
	       skb->network_header,
	       has_trans ? skb_network_header_len(skb) : -1,
	       has_trans ? skb->transport_header : -1,
	       sh->tx_flags, sh->nr_frags,
	       sh->gso_size, sh->gso_type, sh->gso_segs,
	       skb->csum, skb->ip_summed, skb->csum_complete_sw,
	       skb->csum_valid, skb->csum_level,
	       skb->hash, skb->sw_hash, skb->l4_hash,
	       ntohs(skb->protocol), skb->pkt_type, skb->skb_iif);
    if (skb_is_nonlinear(skb)) {
        printk("is nonlinear");
    } else {
         printk("is linear");
    }
    printk("sk_buff: len:%d  skb->data_len:%d  truesize:%d head:%0X  data:%0X tail:%d end:%d"
    ,skb->len,skb->data_len,skb->truesize,(skb->head),(skb->data),(skb->tail),(skb->end));
    struct skb_shared_info *sp = skb_shinfo(skb);
    page_count =0; 
    for (i = 0; i < skb_shinfo(skb)->nr_frags; i++) {
    	skb_frag_t *frag = &skb_shinfo(skb)->frags[i];
    	u32 p_off, p_len, copied;
    	struct page *p;
    	u8 *vaddr;
        printk("******** No.%d page , \n", ++page_count);
    	skb_frag_foreach_page(frag, frag->page_offset, skb_frag_size(frag),
    			      p, p_off, p_len, copied) {
    		vaddr = kmap_atomic(p);
    		print_hex_dump(level, "skb frag : ", DUMP_PREFIX_OFFSET,
    			       16, 1, vaddr + p_off, p_len, false);
    		kunmap_atomic(vaddr);
    	}
    }
    
    if (skb_has_frag_list(skb))
    	printk("%s ****************** skb frags list:\n", level);
    i = 0;
    skb_walk_frags(skb, frag_iter)
    {
 
    	        printk("%s ****************** wallk skb frags list %d timers\n", level, ++i);
    		mydump_skb(level, frag_iter, false);
    }
}

netdev_tx_t hinic_xmit_frame(struct sk_buff *skb, struct net_device *netdev)
{
	struct hinic_dev *nic_dev = netdev_priv(netdev);
	struct netdev_queue *netdev_txq;
	int nr_sges, err = NETDEV_TX_OK;
	struct hinic_sq_wqe *sq_wqe;
	unsigned int wqe_size;
	struct hinic_txq *txq;
	struct hinic_qp *qp;
	u16 prod_idx;

	txq = &nic_dev->txqs[skb->queue_mapping];
	qp = container_of(txq->sq, struct hinic_qp, sq);

#if 1
       printk("dev->features&NETIF_F_SG: %x \n", netdev->features&NETIF_F_SG);
        struct iphdr *iph = ip_hdr(skb);
        struct icmphdr *icmph;
        __be16 morefrag;
        morefrag = iph->frag_off & htons(IP_MF);
        printk("**************hinic_xmit_frame  morefrag: %x \n",  morefrag);
        if(iph->protocol == IPPROTO_ICMP) {
            icmph = icmp_hdr(skb);
            if (icmph->type == ICMP_ECHO) {
                 pr_info("************************************* dump hinix xmit skb \n");
                  mydump_skb(KERN_ERR, skb, false);
            }
       }
#endif
	if (skb->len < MIN_SKB_LEN) {
		if (skb_pad(skb, MIN_SKB_LEN - skb->len)) {
			netdev_err(netdev, "Failed to pad skb\n");
			goto update_error_stats;
		}

		skb->len = MIN_SKB_LEN;
	}

	nr_sges = skb_shinfo(skb)->nr_frags + 1;
	if (nr_sges > txq->max_sges) {
		netdev_err(netdev, "Too many Tx sges\n");
		goto skb_error;
	}

	err = tx_map_skb(nic_dev, skb, txq->sges);
	if (err)
		goto skb_error;

	wqe_size = HINIC_SQ_WQE_SIZE(nr_sges);

	sq_wqe = hinic_sq_get_wqe(txq->sq, wqe_size, &prod_idx);
	if (!sq_wqe) {
		tx_unmap_skb(nic_dev, skb, txq->sges);

		netif_stop_subqueue(netdev, qp->q_id);

		u64_stats_update_begin(&txq->txq_stats.syncp);
		txq->txq_stats.tx_busy++;
		u64_stats_update_end(&txq->txq_stats.syncp);
		err = NETDEV_TX_BUSY;
		goto flush_skbs;
	}

	hinic_sq_prepare_wqe(txq->sq, prod_idx, sq_wqe, txq->sges, nr_sges);

	hinic_sq_write_wqe(txq->sq, prod_idx, sq_wqe, skb, wqe_size);

flush_skbs:
	netdev_txq = netdev_get_tx_queue(netdev, skb->queue_mapping);
	if ((!skb->xmit_more) || (netif_xmit_stopped(netdev_txq)))
		hinic_sq_write_db(txq->sq, prod_idx, wqe_size, 0);

	return err;

skb_error:
	dev_kfree_skb_any(skb);

update_error_stats:
	u64_stats_update_begin(&txq->txq_stats.syncp);
	txq->txq_stats.tx_dropped++;
	u64_stats_update_end(&txq->txq_stats.syncp);
	return err;
}

/**
 * tx_free_skb - unmap and free skb
 * @nic_dev: nic device
 * @skb: the skb
 * @sges: the sges that are connected to the skb
 **/
static void tx_free_skb(struct hinic_dev *nic_dev, struct sk_buff *skb,
			struct hinic_sge *sges)
{
	tx_unmap_skb(nic_dev, skb, sges);

	dev_kfree_skb_any(skb);
}

/**
 * free_all_rx_skbs - free all skbs in tx queue
 * @txq: tx queue
 **/
static void free_all_tx_skbs(struct hinic_txq *txq)
{
	struct hinic_dev *nic_dev = netdev_priv(txq->netdev);
	struct hinic_sq *sq = txq->sq;
	struct hinic_sq_wqe *sq_wqe;
	unsigned int wqe_size;
	struct sk_buff *skb;
	int nr_sges;
	u16 ci;

	while ((sq_wqe = hinic_sq_read_wqe(sq, &skb, &wqe_size, &ci))) {
		nr_sges = skb_shinfo(skb)->nr_frags + 1;

		hinic_sq_get_sges(sq_wqe, txq->free_sges, nr_sges);

		hinic_sq_put_wqe(sq, wqe_size);

		tx_free_skb(nic_dev, skb, txq->free_sges);
	}
}

/**
 * free_tx_poll - free finished tx skbs in tx queue that connected to napi
 * @napi: napi
 * @budget: number of tx
 *
 * Return 0 - Success, negative - Failure
 **/
static int free_tx_poll(struct napi_struct *napi, int budget)
{
	struct hinic_txq *txq = container_of(napi, struct hinic_txq, napi);
	struct hinic_qp *qp = container_of(txq->sq, struct hinic_qp, sq);
	struct hinic_dev *nic_dev = netdev_priv(txq->netdev);
	struct netdev_queue *netdev_txq;
	struct hinic_sq *sq = txq->sq;
	struct hinic_wq *wq = sq->wq;
	struct hinic_sq_wqe *sq_wqe;
	unsigned int wqe_size;
	int nr_sges, pkts = 0;
	struct sk_buff *skb;
	u64 tx_bytes = 0;
	u16 hw_ci, sw_ci;

	do {
		hw_ci = HW_CONS_IDX(sq) & wq->mask;

		sq_wqe = hinic_sq_read_wqe(sq, &skb, &wqe_size, &sw_ci);
		if ((!sq_wqe) ||
		    (((hw_ci - sw_ci) & wq->mask) * wq->wqebb_size < wqe_size))
			break;

		tx_bytes += skb->len;
		pkts++;

		nr_sges = skb_shinfo(skb)->nr_frags + 1;

		hinic_sq_get_sges(sq_wqe, txq->free_sges, nr_sges);

		hinic_sq_put_wqe(sq, wqe_size);

		tx_free_skb(nic_dev, skb, txq->free_sges);
	} while (pkts < budget);

	if (__netif_subqueue_stopped(nic_dev->netdev, qp->q_id) &&
	    hinic_get_sq_free_wqebbs(sq) >= HINIC_MIN_TX_NUM_WQEBBS(sq)) {
		netdev_txq = netdev_get_tx_queue(txq->netdev, qp->q_id);

		__netif_tx_lock(netdev_txq, smp_processor_id());

		netif_wake_subqueue(nic_dev->netdev, qp->q_id);

		__netif_tx_unlock(netdev_txq);

		u64_stats_update_begin(&txq->txq_stats.syncp);
		txq->txq_stats.tx_wake++;
		u64_stats_update_end(&txq->txq_stats.syncp);
	}

	u64_stats_update_begin(&txq->txq_stats.syncp);
	txq->txq_stats.bytes += tx_bytes;
	txq->txq_stats.pkts += pkts;
	u64_stats_update_end(&txq->txq_stats.syncp);

	if (pkts < budget) {
		napi_complete(napi);
		enable_irq(sq->irq);
		return pkts;
	}

	return budget;
}

static void tx_napi_add(struct hinic_txq *txq, int weight)
{
	netif_napi_add(txq->netdev, &txq->napi, free_tx_poll, weight);
	napi_enable(&txq->napi);
}

static void tx_napi_del(struct hinic_txq *txq)
{
	napi_disable(&txq->napi);
	netif_napi_del(&txq->napi);
}

static irqreturn_t tx_irq(int irq, void *data)
{
	struct hinic_txq *txq = data;
	struct hinic_dev *nic_dev;

	nic_dev = netdev_priv(txq->netdev);

	/* Disable the interrupt until napi will be completed */
	disable_irq_nosync(txq->sq->irq);

	hinic_hwdev_msix_cnt_set(nic_dev->hwdev, txq->sq->msix_entry);

	napi_schedule(&txq->napi);
	return IRQ_HANDLED;
}

static int tx_request_irq(struct hinic_txq *txq)
{
	struct hinic_dev *nic_dev = netdev_priv(txq->netdev);
	struct hinic_hwdev *hwdev = nic_dev->hwdev;
	struct hinic_hwif *hwif = hwdev->hwif;
	struct pci_dev *pdev = hwif->pdev;
	struct hinic_sq *sq = txq->sq;
	int err;

	tx_napi_add(txq, nic_dev->tx_weight);

	hinic_hwdev_msix_set(nic_dev->hwdev, sq->msix_entry,
			     TX_IRQ_NO_PENDING, TX_IRQ_NO_COALESC,
			     TX_IRQ_NO_LLI_TIMER, TX_IRQ_NO_CREDIT,
			     TX_IRQ_NO_RESEND_TIMER);

	err = request_irq(sq->irq, tx_irq, 0, txq->irq_name, txq);
	if (err) {
		dev_err(&pdev->dev, "Failed to request Tx irq\n");
		tx_napi_del(txq);
		return err;
	}

	return 0;
}

static void tx_free_irq(struct hinic_txq *txq)
{
	struct hinic_sq *sq = txq->sq;

	free_irq(sq->irq, txq);
	tx_napi_del(txq);
}

/**
 * hinic_init_txq - Initialize the Tx Queue
 * @txq: Logical Tx Queue
 * @sq: Hardware Tx Queue to connect the Logical queue with
 * @netdev: network device to connect the Logical queue with
 *
 * Return 0 - Success, negative - Failure
 **/
int hinic_init_txq(struct hinic_txq *txq, struct hinic_sq *sq,
		   struct net_device *netdev)
{
	struct hinic_qp *qp = container_of(sq, struct hinic_qp, sq);
	struct hinic_dev *nic_dev = netdev_priv(netdev);
	struct hinic_hwdev *hwdev = nic_dev->hwdev;
	int err, irqname_len;
	size_t sges_size;

	txq->netdev = netdev;
	txq->sq = sq;

	txq_stats_init(txq);

	txq->max_sges = HINIC_MAX_SQ_BUFDESCS;

	sges_size = txq->max_sges * sizeof(*txq->sges);
	txq->sges = devm_kzalloc(&netdev->dev, sges_size, GFP_KERNEL);
	if (!txq->sges)
		return -ENOMEM;

	sges_size = txq->max_sges * sizeof(*txq->free_sges);
	txq->free_sges = devm_kzalloc(&netdev->dev, sges_size, GFP_KERNEL);
	if (!txq->free_sges) {
		err = -ENOMEM;
		goto err_alloc_free_sges;
	}

	irqname_len = snprintf(NULL, 0, "hinic_txq%d", qp->q_id) + 1;
	txq->irq_name = devm_kzalloc(&netdev->dev, irqname_len, GFP_KERNEL);
	if (!txq->irq_name) {
		err = -ENOMEM;
		goto err_alloc_irqname;
	}

	sprintf(txq->irq_name, "hinic_txq%d", qp->q_id);

	err = hinic_hwdev_hw_ci_addr_set(hwdev, sq, CI_UPDATE_NO_PENDING,
					 CI_UPDATE_NO_COALESC);
	if (err)
		goto err_hw_ci;

	err = tx_request_irq(txq);
	if (err) {
		netdev_err(netdev, "Failed to request Tx irq\n");
		goto err_req_tx_irq;
	}

	return 0;

err_req_tx_irq:
err_hw_ci:
	devm_kfree(&netdev->dev, txq->irq_name);

err_alloc_irqname:
	devm_kfree(&netdev->dev, txq->free_sges);

err_alloc_free_sges:
	devm_kfree(&netdev->dev, txq->sges);
	return err;
}

/**
 * hinic_clean_txq - Clean the Tx Queue
 * @txq: Logical Tx Queue
 **/
void hinic_clean_txq(struct hinic_txq *txq)
{
	struct net_device *netdev = txq->netdev;

	tx_free_irq(txq);

	free_all_tx_skbs(txq);

	devm_kfree(&netdev->dev, txq->irq_name);
	devm_kfree(&netdev->dev, txq->free_sges);
	devm_kfree(&netdev->dev, txq->sges);
}
