

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/linux-5.15.rt/drivers/net/ethernet/cadence/no_napi/noapi2.png)

# napi
##  napi_disable
```
#ifndef TEST_POLL_NO_USE_NAPI
	for (q = 0, queue = bp->queues; q < bp->num_queues; ++q, ++queue)
		napi_disable(&queue->napi);
#endif
```

## napi_enable
```
#ifndef TEST_POLL_NO_USE_NAPI
	for (q = 0, queue = bp->queues; q < bp->num_queues;
	     ++q, ++queue)
		napi_enable(&queue->napi);
#endif
```

##  napi_schedule

```
#ifdef TEST_POLL
		if (status & bp->rx_intr_mask) {
			/* There's no point taking any more interrupts
			 * until we have processed the buffers. The
			 * scheduling call may fail if the poll routine
			 * is already scheduled, so disable interrupts
			 * now.
			 */
			queue_writel(queue, IDR, bp->rx_intr_mask);
			if (bp->caps & MACB_CAPS_ISR_CLEAR_ON_WRITE)
				queue_writel(queue, ISR, MACB_BIT(RCOMP));

			//int budget = netdev_budget;//300
            		gem_rx(queue, NULL, 300);
		}
#else
		if (status & bp->rx_intr_mask) {
			/* There's no point taking any more interrupts
			 * until we have processed the buffers. The
			 * scheduling call may fail if the poll routine
			 * is already scheduled, so disable interrupts
			 * now.
			 */
			queue_writel(queue, IDR, bp->rx_intr_mask);
			if (bp->caps & MACB_CAPS_ISR_CLEAR_ON_WRITE)
				queue_writel(queue, ISR, MACB_BIT(RCOMP));

			if (napi_schedule_prep(&queue->napi)) {
				netdev_vdbg(bp->dev, "scheduling RX softirq\n");
				__napi_schedule(&queue->napi);
			}
		}
#endif
```

## netif_napi_add

```
#ifdef TEST_POLL_NO_USE_NAPI
	        dev_err(&pdev->dev, "not need to use napi \n");
#else 
		netif_napi_add(dev, &queue->napi, macb_poll, NAPI_POLL_WEIGHT);
#endif
```

#  napi_gro_receive coredump

```
root@Ubuntu-riscv64:~# ping 10.11.11.81
PING 10.11.11.81 (10.11.11.81) 56(84) bytes of data.
[  667.063699] Unable to handle kernel NULL pointer dereference at virtual address 0000000000000038
[  667.063740] Oops [#1]
[  667.063747] Modules linked in: macb(O) phylink
[  667.063773] CPU: 2 PID: 1370 Comm: macb poll-#0 Tainted: G        W  O      5.15.24-rt31 #3
[  667.063785] Hardware name: SiFive HiFive Unmatched A00 (DT)
[  667.063791] epc : dev_gro_receive+0xf6/0x4e4
[  667.063816]  ra : napi_gro_receive+0x54/0x1fc
[  667.063828] epc : ffffffff809ba920 ra : ffffffff809bad62 sp : ffffffd00436bc80
[  667.063843]  gp : ffffffff81a27358 tp : ffffffe0867f8ac0 t0 : ffffffd00409bedc
[  667.063857]  t1 : 00000000000092d5 t2 : 000000000000fa2f s0 : ffffffd00436bd20
[  667.063870]  s1 : ffffffe083cc3300 a0 : 0000000000000000 a1 : ffffffe083cc3300
[  667.063884]  a2 : 0000000000000640 a3 : 9012000000000000 a4 : 0000000000000000
[  667.063898]  a5 : 000000000000000e a6 : 000000000000b370 a7 : 0000000000003cfa
[  667.063911]  s2 : ffffffe083b307c0 s3 : 0000000000000000 s4 : 0000000000000000
[  667.063924]  s5 : 0000000000000000 s6 : 0000000000000003 s7 : 0000000000000002
[  667.063937]  s8 : 000000000800c03c s9 : 0000000000000000 s10: 0000000000000000
[  667.063950]  s11: 0000000000000000 t3 : 0000000000000001 t4 : 00000000000003f9
[  667.063963]  t5 : 000000000001a1fb t6 : 0000000000000000
[  667.063973] status: 0000000200000100 badaddr: 0000000000000038 cause: 000000000000000d
[  667.063990] [<ffffffff809bad62>] napi_gro_receive+0x54/0x1fc
[  667.064003] [<ffffffff01acf324>] gem_rx+0x178/0x21e [macb]
[  667.065940] [<ffffffff01acf4ee>] macb_poll_task+0x124/0x510 [macb]
[  667.066634] [<ffffffff8002f8ba>] kthread+0x156/0x18e
[  667.066654] [<ffffffff8000386c>] ret_from_exception+0x0/0x
```

because gem_rx(queue, NULL, 300), 

```
static int gem_rx(struct macb_queue *queue, struct napi_struct *napi,
		  int budget)
napi=NULL
```

## napi->gro_hash

```
void netif_napi_add(struct net_device *dev, struct napi_struct *napi,
		    int (*poll)(struct napi_struct *, int), int weight)
{
	int i;

	INIT_LIST_HEAD(&napi->poll_list);
	hrtimer_init(&napi->timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL_PINNED);
	napi->timer.function = napi_watchdog;
	napi->gro_bitmask = 0;
	for (i = 0; i < GRO_HASH_BUCKETS; i++) {
		INIT_LIST_HEAD(&napi->gro_hash[i].list);
		napi->gro_hash[i].count = 0;
	}
	napi->skb = NULL;
	napi->poll = poll;
	if (weight > NAPI_POLL_WEIGHT)
		pr_err_once("netif_napi_add() called with weight %d on device %s\n",
			    weight, dev->name);
	napi->weight = weight;
	napi->dev = dev;
#ifdef CONFIG_NETPOLL
	napi->poll_owner = -1;
#endif
	set_bit(NAPI_STATE_SCHED, &napi->state);
	set_bit(NAPI_STATE_NPSVC, &napi->state);
	list_add_rcu(&napi->dev_list, &dev->napi_list);
	napi_hash_add(napi);
	/* Create kthread for this napi if dev->threaded is set.
	 * Clear dev->threaded if kthread creation failed so that
	 * threaded mode will not be enabled in napi_enable().
	 */
	if (dev->threaded && napi_kthread_create(napi))
		dev->threaded = 0;
}
```


# process_backlog

```
static int process_backlog(struct napi_struct *napi, int quota)
{
    int work = 0;
                                        
    /*取得本地CPU上的softnet_data  数据*/
    struct softnet_data *queue = &__get_cpu_var(softnet_data);
　　
    /*开始计时，一旦允许时间到，就退出轮询*/
    unsigned long start_time = jiffies;
    napi->weight = weight_p;
　　
    /*循环从softnet_data 的输入队列取报文并处理，直到队列中没有报文了,
     或处理的报文数大于了允许的上限值了，
     或轮询函数执行时间大于一个jiffies 了
　　*/
    do
    {
        struct sk_buff *skb;
        /*禁用本地中断，要存队列中取skb,防止抢占*/
        local_irq_disable();
　　
        /*从softnet_data 的输入队列中取得一个skb*/
        skb = __skb_dequeue(&queue->input_pkt_queue);
　　
        /*如果队列中没有skb,则使能中断并退出轮询*/
        if (!skb)
        {
            /*把napi 从 softnet_data 的 pool_list 链表上摘除*/
            __napi_complete(napi);
            /*使能本地CPU的中断*/
            local_irq_enable();
            break;
        }
        /*skb 已经摘下来了，使能中断*/
        local_irq_enable();
　　
        /*把skb送到协议栈相关协议模块进行处理,详细处理见后续章节*/
        netif_receive_skb(skb);
    } while (++work < quota && jiffies == start_time);
    /*返回处理报文个数*/
    return work;
}

```