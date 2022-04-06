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



# NOHZ tick-stop error: Non-RCU local softirq work is pending, handler #08!!!

```
}
static int macb_poll_task(void * p)
{
        struct net_device *dev = (struct net_device *)p;
        struct macb *bp = netdev_priv(dev);
        struct macb_queue *queue;
        unsigned long flags;
        unsigned int q;
        while(!kthread_should_stop() && netif_running(dev))
        {
            unsigned long time_limit = jiffies + 2;
            local_irq_save(flags);
            for (q = 0, queue = bp->queues; q < bp->num_queues; ++q, ++queue)
            {
                macb_interrupt(dev->irq, queue);
                if (time_after_eq(jiffies, time_limit))
                {
                     udelay(10);
                }
            }
            local_irq_restore(flags);
        }
        return 0;
}
```
## reference net_rx_action

[Linux网络协议源码分析(六)：网卡收包流程](https://pzh2386034.github.io/Black-Jack/linux-net/2020/01/16/Linux%E7%BD%91%E7%BB%9C%E5%8D%8F%E8%AE%AE%E6%BA%90%E7%A0%81%E5%88%86%E6%9E%90(%E5%85%AD)-%E7%BD%91%E5%8D%A1%E6%94%B6%E5%8C%85%E6%B5%81%E7%A8%8B/)

```
/*
 * 遍历中断对应CPU的 softnet_data.poll_list 上的设备结构体，将设备上的数据包发到网络协议栈处理
 * 1. 设置软中断一次最大处理数据量、时间
 * 2. napi_poll: 逐个处理设备结构体，其中会使用驱动初始化时注册的回调收包 poll 函数，将数据包送到网络协议栈
 * 3. 如果最后 poll_list 上还有设备没处理，则退出前再次触发软中断
 */
static void net_rx_action(struct softirq_action *h)
{
	struct softnet_data *sd = this_cpu_ptr(&softnet_data);
	/* 设置软中断一次允许的最大执行时间为2个jiffies */
	unsigned long time_limit = jiffies + 2;
	/* 设置软中断接收函数一次最多处理的报文个数为300 */
	int budget = netdev_budget;
	LIST_HEAD(list);
	LIST_HEAD(repoll);
	/* 关闭本地cpu的中断，下面判断list是否为空时防止硬中断抢占 */
	local_irq_disable();
	/* 将要轮询的设备链表转移到临时链表上 */
	list_splice_init(&sd->poll_list, &list);
	local_irq_enable();
	/* 循环处理poll_list链表上的等待处理的napi */
	for (;;) {
		struct napi_struct *n;
		/* 如果遍历完链表，则停止 */
		if (list_empty(&list)) {
			if (!sd_has_rps_ipi_waiting(sd) && list_empty(&repoll))
				return;
			break;
		}
		/* 获取链表中首个设备 */
		n = list_first_entry(&list, struct napi_struct, poll_list);
		/* 调用驱动初始化时通过 netif_napi_add 注册的回调收包 poll 函数；非NAPI为固定 process_backlog()
		 * 处理完一个设备上的报文则要记录处理数量
		 */
		budget -= napi_poll(n, &repoll);

		/* 如果超出预设时间或者达到处理报文最大个数则停止处理 */
		if (unlikely(budget <= 0 ||
			     time_after_eq(jiffies, time_limit))) {
			sd->time_squeeze++;
			break;
		}
	}

	local_irq_disable();

	list_splice_tail_init(&sd->poll_list, &list);
	list_splice_tail(&repoll, &list);
	list_splice(&list, &sd->poll_list);
	/* 如果softnet_data.poll_list上还有未处理设备，则继续触发软中断 */
	if (!list_empty(&sd->poll_list))
		__raise_softirq_irqoff(NET_RX_SOFTIRQ);

	net_rps_action_and_irq_enable(sd);
}
```



# references
[e100 NAPI](https://blog.csdn.net/Rong_Toa/article/details/109401935)