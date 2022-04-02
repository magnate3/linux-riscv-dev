 # skb_reset_network_header to obtain arp header
**skb_reset_mac_header**<br>
**skb_reset_network_header**<br>
**skb_reset_transport_header**<br>

## arp_hdr

```
static inline struct arphdr *arp_hdr(const struct sk_buff *skb)
{
        return (struct arphdr *)skb_network_header(skb);
}

include/linux/skbuff.h


static inline unsigned char *skb_network_header(const struct sk_buff *skb)
{
        return skb->head + skb->network_header;
}
```
  
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.14/drivers/net/ethernet/cadence/by_pass_arp/pic/arp.png)  
  
  ` the vaule of  kb->network_header changes ` <br>
```
[   36.619699] macb: before skb_reset_network_header, the  skb->network_header 0 
[   36.619714] macb: after skb_reset_network_header, the  skb->network_header 80 
```  
  
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

## netif_receive_skb do  skb_reset_network_header  and skb_reset_transport_header

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

## arp_create call skb_reserve 、skb_reset_network_header

```
struct sk_buff *arp_create(int type, int ptype, __be32 dest_ip,
                           struct net_device *dev, __be32 src_ip,
                           const unsigned char *dest_hw,
                           const unsigned char *src_hw,
                           const unsigned char *target_hw)
{
        struct sk_buff *skb;
        struct arphdr *arp;
        unsigned char *arp_ptr;
        int hlen = LL_RESERVED_SPACE(dev);
        int tlen = dev->needed_tailroom;

        /*
         *      Allocate a buffer
         */

        skb = alloc_skb(arp_hdr_len(dev) + hlen + tlen, GFP_ATOMIC);
        if (!skb)
                return NULL;

        skb_reserve(skb, hlen);
        skb_reset_network_header(skb);
        arp = skb_put(skb, arp_hdr_len(dev));
        skb->dev = dev;
        skb->protocol = htons(ETH_P_ARP);
        if (!src_hw)
                src_hw = dev->dev_addr;
        if (!dest_hw)
                dest_hw = dev->broadcast;

        /*
         *      Fill the device header for the ARP frame
         */
        if (dev_hard_header(skb, dev, ptype, dest_hw, src_hw, skb->len) < 0)
                goto out;

        /*
         * Fill out the arp protocol part.
         *
         * The arp hardware type should match the device type, except for FDDI,
         * which (according to RFC 1390) should always equal 1 (Ethernet).
         */
        /*
         *      Exceptions everywhere. AX.25 uses the AX.25 PID value not the
         *      DIX code for the protocol. Make these device structure fields.
         */
        switch (dev->type) {
        default:
                arp->ar_hrd = htons(dev->type);
                arp->ar_pro = htons(ETH_P_IP);
                break;

#if IS_ENABLED(CONFIG_AX25)
        case ARPHRD_AX25:
                arp->ar_hrd = htons(ARPHRD_AX25);
                arp->ar_pro = htons(AX25_P_IP);
                break;

#if IS_ENABLED(CONFIG_NETROM)
        case ARPHRD_NETROM:
                arp->ar_hrd = htons(ARPHRD_NETROM);
                arp->ar_pro = htons(AX25_P_IP);
                break;
#endif
#endif

#if IS_ENABLED(CONFIG_FDDI)
        case ARPHRD_FDDI:
                arp->ar_hrd = htons(ARPHRD_ETHER);
                arp->ar_pro = htons(ETH_P_IP);
                break;
#endif
        }

        arp->ar_hln = dev->addr_len;
        arp->ar_pln = 4;
        arp->ar_op = htons(type);

        arp_ptr = (unsigned char *)(arp + 1);

        memcpy(arp_ptr, src_hw, dev->addr_len);
        arp_ptr += dev->addr_len;
        memcpy(arp_ptr, &src_ip, 4);
        arp_ptr += 4;

        switch (dev->type) {
#if IS_ENABLED(CONFIG_FIREWIRE_NET)
        case ARPHRD_IEEE1394:
                break;
#endif
        default:
                if (target_hw)
                        memcpy(arp_ptr, target_hw, dev->addr_len);
                else
                        memset(arp_ptr, 0, dev->addr_len);
                arp_ptr += dev->addr_len;
        }
        memcpy(arp_ptr, &dest_ip, 4);

        return skb;

out:
        kfree_skb(skb);
        return NULL;
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