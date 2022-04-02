 # skb_reset_network_header to obtain arp header
*skb_reset_mac_header*<br>
*skb_reset_network_header*<br>
*skb_reset_transport_header*<br>
  
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/by_pass_arp/pic/arp.png)  
  
  
```
static inline unsigned char *skb_transport_header(const struct sk_buff *skb)
{
        return skb->head + skb->transport_header;
}

static inline void skb_reset_transport_header(struct sk_buff *skb)
{
        skb->transport_header = skb->data - skb->head;
}

static inline void skb_set_transport_header(struct sk_buff *skb,
                                            const int offset)
{
        skb_reset_transport_header(skb);
        skb->transport_header += offset;
}

static inline unsigned char *skb_network_header(const struct sk_buff *skb)
{
        return skb->head + skb->network_header;
}

static inline void skb_reset_network_header(struct sk_buff *skb)
{
        skb->network_header = skb->data - skb->head;
}

static inline void skb_set_network_header(struct sk_buff *skb, const int offset)
{
        skb_reset_network_header(skb);
        skb->network_header += offset;
}

static inline unsigned char *skb_mac_header(const struct sk_buff *skb)
{
        return skb->head + skb->mac_header;
}
```

##  skb_get to increase  skb reference

```

/**
 *      skb_get - reference buffer
 *      @skb: buffer to reference
 *
 *      Makes another reference to a socket buffer and returns a pointer
 *      to the buffer.
 */
static inline struct sk_buff *skb_get(struct sk_buff *skb)
{
        refcount_inc(&skb->users);
        return skb;
}
```

## netif_receive_skb

in netif_receive_skb ,even vlan ,should  skb_reset_network_header and skb_reset_transport_header
```
static int __netif_receive_skb_core(struct sk_buff *skb, bool pfmemalloc)
{
     skb_reset_network_header(skb);
     if (!skb_transport_header_was_set(skb))
          skb_reset_transport_header(skb);
     skb_reset_mac_len(skb);
     if (skb_vlan_tag_present(skb)) {
         /* 处理prev */
         if (pt_prev) {
             ret = deliver_skb(skb, pt_prev, orig_dev);
             pt_prev = NULL;
         }
 
         /* 根据实际的vlan设备调整信息，再走一遍 */
         if (vlan_do_receive(&skb))
             goto another_round;
         else if (unlikely(!skb))
             goto out;
     }
}
```

# dev_kfree_skb_any to kfree uselss or err skb

```
static netdev_tx_t macb_start_xmit(struct sk_buff *skb, struct net_device *dev)
{
    /* Map socket buffer for DMA transfer */
	if (!macb_tx_map(bp, queue, skb, hdrlen)) {
		dev_kfree_skb_any(skb);
		goto unlock;
	}
}
```
void kfree_skb(struct sk_buff *skb);
void dev_kfree_skb(struct sk_buff *skb);
void dev_kfree_skb_irq(struct sk_buff *skb);
void dev_kfree_skb_any(struct sk_buff *skb);
上述函数用于释放被alloc_skb( )函数分配的套接字缓冲区和数据缓冲区。
inux内核内部使用 kfree_skb( ) 函数，但在网络设备驱动程序中最好用 dev_kfree_skb( )、dev_kfree_skb_irq( ) 或 
dev_kfree_skb_any( )函数进行套接字缓冲区的释放。
dev_kfree_skb( )用于非中断上下文， dev_kfree_skb_irq( )用于中断上下文，
dev_kfree_skb_any( )在中断和非中断上下文中皆可使用，它其实是做一个简单地上下文判断，
 
  
# references
[e100 NAPI](https://blog.csdn.net/Rong_Toa/article/details/109401935)