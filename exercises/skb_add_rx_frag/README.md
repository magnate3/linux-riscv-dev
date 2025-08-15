

# mt7663s_build_rx_skb

```
static struct sk_buff *mt7663s_build_rx_skb(void *data, int data_len,
					    int buf_len)
{
	int len = min_t(int, data_len, MT_SKB_HEAD_LEN);
	struct sk_buff *skb;

	skb = alloc_skb(len, GFP_KERNEL);
	if (!skb)
		return NULL;

	skb_put_data(skb, data, len);
	if (data_len > len) {
		struct page *page;

		data += len;
		page = virt_to_head_page(data);
		skb_add_rx_frag(skb, skb_shinfo(skb)->nr_frags,
				page, data - page_address(page),
				data_len - len, buf_len);
		get_page(page);
	}

	return skb;
}
```




# insmod  skb_test.ko

```
   struct iphdr *iph = ip_hdr(skb);
   if(iph->protocol == IPPROTO_ICMP) {
        struct page *page;
        struct kmem_cache *k_cache_ptr;
        struct sk_buff *skb2 = skb_copy(skb, GFP_ATOMIC);
        page = virt_to_head_page(skb->head);
        pr_info("skb->head page %p \n", page);
        if (PageSlab(page)) {
			k_cache_ptr = page->slab_cache; 
			pr_info("[skb][kmem_cache]name : %s, size : %x\n", k_cache_ptr->name, k_cache_ptr->size);
	} 
        page = virt_to_head_page(skb2->head);
        pr_info("skb2->head page %p \n", page);
        if (PageSlab(page)) {
			k_cache_ptr = page->slab_cache; 
			pr_info("[skb2][kmem_cache]name : %s, size : %x\n", k_cache_ptr->name, k_cache_ptr->size);
	} 
        kfree_skb(skb2);
    }
```


```
[root@centos7 skb_linear]# insmod  skb_test.ko 
[root@centos7 skb_linear]# dmesg | tail -n 10
[80982.121349] skb2->head page ffff7fe00ff2d240 
[80982.125685] [skb2][kmem_cache]name : kmalloc-1024, size : 400
[80983.118481] skb->head page ffff7fe00ff6eec0 
[80983.122734] skb2->head page ffff7fe00ff2d240 
[80983.127071] [skb2][kmem_cache]name : kmalloc-1024, size : 400
[80984.119875] skb->head page ffff7fe00ff6eec0 
[80984.124127] skb2->head page ffff7fe00ff2d240 
[80984.128464] [skb2][kmem_cache]name : kmalloc-1024, size : 400
[80985.121276] skb->head page ffff7fe00ff6eec0 
[80985.125528] skb2->head page ffff7fe00ff2d240 
```
***no [skb][kmem_cache]name output , only skb2***


# the zone is NORMAL
```
[root@centos7 skb_linear]# dmesg | tail -n 10
[80989.135470] [skb2][kmem_cache]name : kmalloc-1024, size : 400
[80990.128287] skb->head page ffff7fe00ff6eec0 
[80990.132540] skb2->head page ffff7fe00ff2d240 
[80990.136877] [skb2][kmem_cache]name : kmalloc-1024, size : 400
[80991.129711] skb->head page ffff7fe00ff6eec0 
[80991.133963] skb2->head page ffff7fe00ff2d240 
[80991.138299] [skb2][kmem_cache]name : kmalloc-1024, size : 400
[81155.193351] ===== test_exit =====
[81455.400194] <0>alloc_pages Successfully!
[81455.404114] <0>the zone is NORMAL.
[root@centos7 skb_linear]# 
```
# r8152.c

// linux-5.15.24/drivers/net/usb/r8152.c 有skb_add_rx_frag

//linux-4.14.115/drivers/net/usb/r8152.c 没有skb_add_rx_frag
```
static int rx_bottom(struct r8152 *tp, int budget)
{
        

                        skb->ip_summed = r8152_rx_csum(tp, rx_desc);
                        memcpy(skb->data, rx_data, rx_frag_head_sz);
                        skb_put(skb, rx_frag_head_sz);
                        pkt_len -= rx_frag_head_sz;
                        rx_data += rx_frag_head_sz;
                        if (pkt_len) {
                                skb_add_rx_frag(skb, 0, agg->page,
                                                agg_offset(agg, rx_data),
                                                pkt_len,
                                                SKB_DATA_ALIGN(pkt_len));
                                get_page(agg->page);
                        }
                           
```

```

static struct rx_agg *alloc_rx_agg(struct r8152 *tp, gfp_t mflags)
{
        struct net_device *netdev = tp->netdev;
        int node = netdev->dev.parent ? dev_to_node(netdev->dev.parent) : -1;
        unsigned int order = get_order(tp->rx_buf_sz);
        struct rx_agg *rx_agg;
        unsigned long flags;

        rx_agg = kmalloc_node(sizeof(*rx_agg), mflags, node);
        if (!rx_agg)
                return NULL;

        rx_agg->page = alloc_pages(mflags | __GFP_COMP, order);
        if (!rx_agg->page)
                goto free_rx;

        rx_agg->buffer = page_address(rx_agg->page);

        rx_agg->urb = usb_alloc_urb(0, mflags);
        if (!rx_agg->urb)
                goto free_buf;

        rx_agg->context = tp;

        INIT_LIST_HEAD(&rx_agg->list);
        INIT_LIST_HEAD(&rx_agg->info_list);
        spin_lock_irqsave(&tp->rx_lock, flags);
        list_add_tail(&rx_agg->info_list, &tp->rx_info);
        spin_unlock_irqrestore(&tp->rx_lock, flags);

        atomic_inc(&tp->rx_count);
}
```

[Markuze/ktcp](https://github.com/Markuze/ktcp/tree/3afa97f5984bee3a863b75849c474724f6d7b480)