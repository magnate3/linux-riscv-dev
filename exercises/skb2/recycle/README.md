
# skb_recycle_check

```
static inline bool skb_is_recycleable(const struct sk_buff *skb, int skb_size)
{
	if (irqs_disabled())
		return false;

	if (skb_shinfo(skb)->tx_flags & SKBTX_DEV_ZEROCOPY)
		return false;

	if (skb_is_nonlinear(skb) || skb->fclone != SKB_FCLONE_UNAVAILABLE)
		return false;

	skb_size = SKB_DATA_ALIGN(skb_size + NET_SKB_PAD);
	if (skb_end_pointer(skb) - skb->head < skb_size)
		return false;

	if (skb_shared(skb) || skb_cloned(skb))
		return false;

	return true;
}
```

```
void skb_recycle(struct sk_buff *skb)
{
	struct skb_shared_info *shinfo;
	skb_release_head_state(skb);
	shinfo = skb_shinfo(skb);
	memset(shinfo, 0, offsetof(struct skb_shared_info, dataref));
	atomic_set(&shinfo->dataref, 1);
	memset(skb, 0, offsetof(struct sk_buff, tail));
	skb->data = skb->head + NET_SKB_PAD;
	skb_reset_tail_pointer(skb);
}
EXPORT_SYMBOL(skb_recycle);
/**
 *	skb_recycle_check - check if skb can be reused for receive
 *	@skb: buffer
 *	@skb_size: minimum receive buffer size
 *
 *	Checks that the skb passed in is not shared or cloned, and
 *	that it is linear and its head portion at least as large as
 *	skb_size so that it can be recycled as a receive buffer.
 *	If these conditions are met, this function does any necessary
 *	reference count dropping and cleans up the skbuff as if it
 *	just came from __alloc_skb().
 */
bool skb_recycle_check(struct sk_buff *skb, int skb_size)
{
	if (!skb_is_recycleable(skb, skb_size))
		return false;
	skb_recycle(skb);
	return true;
}
EXPORT_SYMBOL(skb_recycle_check);

```
#  use  skb_recycle_check
https://github.com/imhcyx/nscscc-linux/blob/b0643d23f1a304be8bb590831261cc2c7c309752/drivers/net/stmmac/stmmac_main.c
```
/**
 * stmmac_tx:
 * @priv: private driver structure
 * Description: it reclaims resources after transmission completes.
 */
static void stmmac_tx(struct stmmac_priv *priv)
{
	unsigned int txsize = priv->dma_tx_size;

	spin_lock(&priv->tx_lock);

	while (priv->dirty_tx != priv->cur_tx) {
		int last;
		unsigned int entry = priv->dirty_tx % txsize;
		struct sk_buff *skb = priv->tx_skbuff[entry];
		struct dma_desc *p = priv->dma_tx + entry;

		/* Check if the descriptor is owned by the DMA. */
		if (priv->hw->desc->get_tx_owner(p))
			break;

		/* Verify tx error by looking at the last segment */
		last = priv->hw->desc->get_tx_ls(p);
		if (likely(last)) {
			int tx_error =
				priv->hw->desc->tx_status(&priv->dev->stats,
							  &priv->xstats, p,
							  priv->ioaddr);
			if (likely(tx_error == 0)) {
				priv->dev->stats.tx_packets++;
				priv->xstats.tx_pkt_n++;
			} else
				priv->dev->stats.tx_errors++;
		}
		TX_DBG("%s: curr %d, dirty %d\n", __func__,
			priv->cur_tx, priv->dirty_tx);

		if (likely(p->des2))
			dma_unmap_single(priv->device, p->des2,
					 priv->hw->desc->get_tx_len(p),
					 DMA_TO_DEVICE);
		priv->hw->ring->clean_desc3(p);

		if (likely(skb != NULL)) {
			/*
			 * If there's room in the queue (limit it to size)
			 * we add this skb back into the pool,
			 * if it's the right size.
			 */
			if ((skb_queue_len(&priv->rx_recycle) <
				priv->dma_rx_size) &&
				skb_recycle_check(skb, priv->dma_buf_sz))
				__skb_queue_head(&priv->rx_recycle, skb);
			else
				dev_kfree_skb(skb);

			priv->tx_skbuff[entry] = NULL;
		}

		priv->hw->desc->release_tx_desc(p);

		entry = (++priv->dirty_tx) % txsize;
	}
	if (unlikely(netif_queue_stopped(priv->dev) &&
		     stmmac_tx_avail(priv) > STMMAC_TX_THRESH(priv))) {
		netif_tx_lock(priv->dev);
		if (netif_queue_stopped(priv->dev) &&
		     stmmac_tx_avail(priv) > STMMAC_TX_THRESH(priv)) {
			TX_DBG("%s: restart transmit\n", __func__);
			netif_wake_queue(priv->dev);
		}
		netif_tx_unlock(priv->dev);
	}
	spin_unlock(&priv->tx_lock);
}

```
```
/**
 *	skb_recycle_check - check if skb can be reused for receive
 *	@skb: buffer
 *	@skb_size: minimum receive buffer size
 *
 *	Checks that the skb passed in is not shared or cloned, and
 *	that it is linear and its head portion at least as large as
 *	skb_size so that it can be recycled as a receive buffer.
 *	If these conditions are met, this function does any necessary
 *	reference count dropping and cleans up the skbuff as if it
 *	just came from __alloc_skb().
 */
bool skb_recycle_check(struct sk_buff *skb, int skb_size)
{
	struct skb_shared_info *shinfo;

	//	if (irqs_disabled())
	//		return false;

	if (skb_is_nonlinear(skb) || skb->fclone != SKB_FCLONE_UNAVAILABLE)
		return false;

	skb_size = SKB_DATA_ALIGN(skb_size + NET_SKB_PAD);
	if (skb_end_pointer(skb) - skb->head < skb_size)
		return false;

	if (skb_shared(skb) || skb_cloned(skb))
		return false;

	skb_release_head_state(skb);

	shinfo = skb_shinfo(skb);
	memset(shinfo, 0, offsetof(struct skb_shared_info, dataref));
	atomic_set(&shinfo->dataref, 1);

	memset(skb, 0, offsetof(struct sk_buff, tail));
	skb->data = skb->head + NET_SKB_PAD;
	skb_reset_tail_pointer(skb);

	return true;
} EXPORT_SYMBOL(skb_recycle_check)
```

# references
https://github.com/ProtouProject/android_kernel_htc_protou/blob/0e395015315c8950e85b70271b973a7d3c1a3ac5/drivers/net/ethernet/freescale/gianfar.c
https://android.googlesource.com/kernel/msm/+/android-msm-flo-3.4-jb-mr2/net/core/skbuff.c
https://github.com/TeamEpsilon/linux-3.8-test_context/blob/80db7b7268c5541dacee47935297bc2c82c9eafd/drivers/net/ethernet/freescale/gianfar.c
https://github.com/imhcyx/nscscc-linux/blob/b0643d23f1a304be8bb590831261cc2c7c309752/drivers/net/stmmac/stmmac_main.c