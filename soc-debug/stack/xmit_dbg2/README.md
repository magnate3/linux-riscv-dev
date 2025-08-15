
# macb debug

```
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
         pr_info("tx head %u and entry %u \n", tx_head, entry);
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
             pr_info("dma desc is no available \n");
        }
        if(__netif_subqueue_stopped(bp->dev, queue_index))
        {
             pr_info("netif subqueue stopped \n");
        }
        dbg_macb_tx_map(bp, queue, skb,hdrlen);
        return 0;
}
```