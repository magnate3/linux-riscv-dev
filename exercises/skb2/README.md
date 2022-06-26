

# skb_headroom
```
/**
 *      skb_headroom - bytes at buffer head
 *      @skb: buffer to check
 *
 *      Return the number of bytes of free space at the head of an &sk_buff.
 */
static inline unsigned int skb_headroom(const struct sk_buff *skb)
{
        return skb->data - skb->head;
}
```
# NET_SKBUFF_DATA_USES_OFFSET

 skb->end: sk_buff_data_t 
 skb->head : unsigned char *
```
#ifdef NET_SKBUFF_DATA_USES_OFFSET
static inline unsigned char *skb_end_pointer(const struct sk_buff *skb)
{
        return skb->head + skb->end;
}

static inline unsigned int skb_end_offset(const struct sk_buff *skb)
{
        return skb->end;
}
#else
static inline unsigned char *skb_end_pointer(const struct sk_buff *skb)
{
        return skb->end;
}

static inline unsigned int skb_end_offset(const struct sk_buff *skb)
{
        return skb->end - skb->head;
}
#endif
```


```
struct sk_buff *skb_copy(const struct sk_buff *skb, gfp_t gfp_mask)
{
        int headerlen = skb_headroom(skb);
        unsigned int size = skb_end_offset(skb) + skb->data_len;
        struct sk_buff *n = __alloc_skb(size, gfp_mask,
                                        skb_alloc_rx_flag(skb), NUMA_NO_NODE);

        if (!n)
                return NULL;

        /* Set the data pointer */
        skb_reserve(n, headerlen);
        /* Set the tail pointer and length */
        skb_put(n, skb->len);

        BUG_ON(skb_copy_bits(skb, -headerlen, n->head, headerlen + skb->len));

        skb_copy_header(n, skb);
        return n;
}
```
## netdev_alloc_skb will skb_reserve(skb, NET_SKB_PAD)
```
 478 struct sk_buff *__netdev_alloc_skb(struct net_device *dev, unsigned int len,
 479                                    gfp_t gfp_mask)
 480 {
 481         struct page_frag_cache *nc;
 482         struct sk_buff *skb;
 483         bool pfmemalloc;
 484         void *data;
 485 
 486         len += NET_SKB_PAD;
 487 
 488         /* If requested length is either too small or too big,
 489          * we use kmalloc() for skb->head allocation.
 490          */
 491         if (len <= SKB_WITH_OVERHEAD(1024) ||
 492             len > SKB_WITH_OVERHEAD(PAGE_SIZE) ||
 493             (gfp_mask & (__GFP_DIRECT_RECLAIM | GFP_DMA))) {
 494                 skb = __alloc_skb(len, gfp_mask, SKB_ALLOC_RX, NUMA_NO_NODE);
 495                 if (!skb)
 496                         goto skb_fail;
 497                 goto skb_success;
 498         }
 499 
 500         len += SKB_DATA_ALIGN(sizeof(struct skb_shared_info));
 501         len = SKB_DATA_ALIGN(len);
 502 
 503         if (sk_memalloc_socks())
 504                 gfp_mask |= __GFP_MEMALLOC;
 505 
 506         if (in_hardirq() || irqs_disabled()) {
 507                 nc = this_cpu_ptr(&netdev_alloc_cache);
 508                 data = page_frag_alloc(nc, len, gfp_mask);
 509                 pfmemalloc = nc->pfmemalloc;
 510         } else {
 511                 local_bh_disable();
 512                 nc = this_cpu_ptr(&napi_alloc_cache.page);
 513                 data = page_frag_alloc(nc, len, gfp_mask);

 514                 pfmemalloc = nc->pfmemalloc;
 515                 local_bh_enable();
 516         }
 517 
 518         if (unlikely(!data))
 519                 return NULL;
 520 
 521         skb = __build_skb(data, len);
 522         if (unlikely(!skb)) {
 523                 skb_free_frag(data);
 524                 return NULL;
 525         }
 526 
 527         if (pfmemalloc)
 528                 skb->pfmemalloc = 1;
 529         skb->head_frag = 1;
 530 
 531 skb_success:
 532         skb_reserve(skb, NET_SKB_PAD);
 533         skb->dev = dev;
 534 
 535 skb_fail:
 536         return skb;
 537 }
```

# skb_copy_and_csum_dev
skb_copy_and_csum_dev 对frag 是单独处理的
skb_copy会把非线性skb 转换为 线性skb

# skb_copy_to_linear_data_offset