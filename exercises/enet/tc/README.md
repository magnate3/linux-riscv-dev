

# 内核版本

[varigit/linux-imx](https://github.com/varigit/linux-imx/tree/5.15-2.0.x-imx_var01)

***linux-imx/net/tsn/***

# "Oper" pk  Admin

The IEEE 802.1Q-2018 defines two "types" of schedules, the "Oper" (from
operational?) and "Admin" ones. Up until now, 'taprio' only had
support for the "Oper" one, added when the qdisc is created. This adds
support for the "Admin" one, which allows the .change() operation to
be supported.

Just for clarification, some quick (and dirty) definitions, the "Oper"
schedule is the currently (as in this instant) running one, and it's
read-only. The "Admin" one is the one that the system configurator has
installed, it can be changed, and it will be "promoted" to "Oper" when
it's 'base-time' is reached.

The idea behing this patch is that calling something like the below,
(after taprio is already configured with an initial schedule):

$ tc qdisc change taprio dev IFACE parent root 	     \
     	   base-time X 	     	   	       	     \
     	   sched-entry <CMD> <GATES> <INTERVAL>	     \
	   ...

Will cause a new admin schedule to be created and programmed to be
"promoted" to "Oper" at instant X. If an "Admin" schedule already
exists, it will be overwritten with the new parameters.

Up until now, there was some code that was added to ease the support
of changing a single entry of a schedule, but was ultimately unused.
Now, that we have support for "change" with more well thought
semantics, updating a single entry seems to be less useful.

So we remove what is in practice dead code, and return a "not
supported" error if the user tries to use it. If changing a single
entry would make the user's life easier we may ressurrect this idea,
but at this point, removing it simplifies the code.

For now, only the schedule specific bits are allowed to be added for a
new schedule, that means that 'clockid', 'num_tc', 'map' and 'queues'
cannot be modified.

Example:

$ tc qdisc change dev IFACE parent root handle 100 taprio \
      base-time $BASE_TIME \
      sched-entry S 00 500000 \
      sched-entry S 0f 500000 \
      clockid CLOCK_TAI

The only change in the netlink API introduced by this change is the
introduction of an "admin" type in the response to a dump request,
that type allows userspace to separate the "oper" schedule from the
"admin" schedule. If userspace doesn't support the "admin" type, it
will only display the "oper" schedule.

# txtime mode

1) a new txtime aware qdisc, tbs, to be used per queue. Its cli will look like:
$ tc qdisc add (...) tbs clockid CLOCK_REALTIME delta 150000 offload sorting

2) a new cmsg-interface for setting a per-packet timestamp that will be used
either as a txtime or as deadline by tbs (and further the NIC driver for the
offlaod case): SCM_TXTIME.

3) a new socket option: SO_TXTIME. It will be used to enable the feature for a
socket, and will have as parameters a clockid and a txtime mode (deadline or
explicit), that defines the semantics of the timestamp set on packets using
SCM_TXTIME.

4) a new #define DYNAMIC_CLOCKID 15 added to include/uapi/linux/time.h .

5) a new schedule-aware qdisc, 'tas' or 'taprio', to be used per port. Its cli
will look like what was proposed for taprio (base time being an absolute timestamp).

# skb->queue_mapping
In multi-queue nic driver, it is used to indicate which queue is used to xmit packet.
It is set by skb_set_queue_mapping in netdev_pick_tx, __dev_queue_xmit.


# Qdisc

![image](https://github.com/magnate3/linux-riscv-dev/tree/main/exercises/enet/tc/tc.png)

## qdisc_priv(sch)

***初始化struct taprio_sched***

```
.priv_size	= sizeof(struct taprio_sched),

static int taprio_init(struct Qdisc *sch, struct nlattr *opt,
		       struct netlink_ext_ack *extack)
{
	struct taprio_sched *q = qdisc_priv(sch);
	struct net_device *dev = qdisc_dev(sch);
	int i;

	spin_lock_init(&q->current_entry_lock);

	hrtimer_init(&q->advance_timer, CLOCK_TAI, HRTIMER_MODE_ABS);
	q->advance_timer.function = advance_sched;

	q->root = sch
	
	
}
```

## child Qdsic

![image](https://github.com/magnate3/linux-riscv-dev/tree/main/exercises/enet/tc/child.png)


## 设备Qdisc   attach

```
dev_open
|-- __dev_open
    |-- dev_active
        |-- attach_default_qdiscs
		|-- qdisc_create_dflt

static void attach_default_qdiscs(struct net_device *dev)
{
    struct netdev_queue *txq;
    struct Qdisc *qdisc;

    /* 获得设备的第 0 个 queue */
    txq = netdev_get_tx_queue(dev, 0);

    /* 如果发送队列个数 <= 1 || 发送队列长度 = 0 */
    if (!netif_is_multiqueue(dev) || dev->tx_queue_len == 0) {
        /* 单队列的流量控制 */
        netdev_for_each_tx_queue(dev, attach_one_default_qdisc, NULL);
        dev->qdisc = txq->qdisc_sleeping;
        atomic_inc(&dev->qdisc->refcnt);
    } else {
        /* 多队列的流量控制; 此处 mq 指 multiqueue*/
        qdisc = qdisc_create_dflt(txq, &mq_qdisc_ops, TC_H_ROOT);
        if (qdisc) {
            qdisc->ops->attach(qdisc);  //////////// attach
            dev->qdisc = qdisc;
        }
    }
}

为多队列的设备创建mq_qdisc, 创建完mq_qdisc， 接着调用mq_qdisc_ops->mq_init函数为每个队列创建pfifo_fast_ops的qdisc

struct Qdisc *qdisc_create_dflt(struct netdev_queue *dev_queue,
			struct Qdisc_ops *ops, unsigned int parentid)
{
	struct Qdisc *sch;

	sch = qdisc_alloc(dev_queue, ops);
	if (IS_ERR(sch))
		goto errout;
	sch->parent = parentid;

	if (!ops->init || ops->init(sch, NULL) == 0)  //init操作
		return sch;

	qdisc_destroy(sch);
errout:
	return NULL;
}
EXPORT_SYMBOL(qdisc_create_dflt);
```

### taprio_attach

```
static void taprio_attach(struct Qdisc *sch)
{
	struct taprio_sched *q = qdisc_priv(sch);
	struct net_device *dev = qdisc_dev(sch);
	unsigned int ntx;

	/* Attach underlying qdisc */
	for (ntx = 0; ntx < dev->num_tx_queues; ntx++) {
		struct Qdisc *qdisc = q->qdiscs[ntx];
		struct Qdisc *old;

		if (FULL_OFFLOAD_IS_ENABLED(q->flags)) {
			qdisc->flags |= TCQ_F_ONETXQUEUE | TCQ_F_NOPARENT;  //////////TCQ_F_ONETXQUEUE 
			old = dev_graft_qdisc(qdisc->dev_queue, qdisc);
		} else {
			old = dev_graft_qdisc(qdisc->dev_queue, sch); //////////////////////
			qdisc_refcount_inc(sch);
		}
		if (old)
			qdisc_put(old);
	}

	/* access to the child qdiscs is not needed in offload mode */
	if (FULL_OFFLOAD_IS_ENABLED(q->flags)) {
		kfree(q->qdiscs);
		q->qdiscs = NULL;
	}
}
```



#  How to schedlule Qdisc

![image](https://github.com/magnate3/linux-riscv-dev/tree/main/exercises/enet/tc/stack.png)

***case 1***: empty qdisc and qdisc could be bypass
If the qdisc could be bypass, such as fifo qdisc,
and it is a empty qdisc,
and the qdisc is not running,

set the qdisc as running,
then send the packet directly by sch_direct_xmit.
If send success, clear the running flag by qdisc_run_end,
or(send failed), put the skb to qdisc queue by dev_requeue_skb.

***case 2***: enqueue and then send
In this case, skb must firstly enqueue.
Check and confirm qdisc is running,
if it is not running before check,
call __qdisc_run.

## How __qdisc_run works


__qdisc_run must be embraced by qdisc_run_begin and qdisc_run_end.
Before __qdisc_run, set flag __QDISC___STATE_RUNNING. after run, remove it.
The flag and two functions ensure a qdisc will run only on a CPU at the smae time.


## taprio 硬件流量控制


![image](https://github.com/magnate3/linux-riscv-dev/tree/main/exercises/enet/tc/taprio.png)

```
static void taprio_sched_to_offload(struct net_device *dev,
				    struct sched_gate_list *sched,
				    struct tc_taprio_qopt_offload *offload)
{
	struct sched_entry *entry;
	int i = 0;

	offload->base_time = sched->base_time;
	offload->cycle_time = sched->cycle_time;
	offload->cycle_time_extension = sched->cycle_time_extension;

	list_for_each_entry(entry, &sched->entries, list) {
		struct tc_taprio_sched_entry *e = &offload->entries[i];

		e->command = entry->command;
		e->interval = entry->interval;
		e->gate_mask = tc_map_to_queue_mask(dev, entry->gate_mask);

		i++;
	}

	offload->num_entries = i;
}
```


![image](https://github.com/magnate3/linux-riscv-dev/tree/main/exercises/enet/tc/entry.png)



```
1565         if (FULL_OFFLOAD_IS_ENABLED(q->flags)) {
1566                 q->dequeue = taprio_dequeue_offload;
1567                 q->peek = taprio_peek_offload;
1568         } else {
1569                 /* Be sure to always keep the function pointers
1570                  * in a consistent state.
1571                  */
1572                 q->dequeue = taprio_dequeue_soft;
1573                 q->peek = taprio_peek_soft;
1574         }
1575 
```

## enqueue

```
 __dev_xmit_skb(struct sk_buff *skb, struct Qdisc *q
 {
 struct sk_buff *to_free = NULL;
 rc = q->enqueue(skb, q, &to_free) & NET_XMIT_MASK
 }
 static int taprio_enqueue(struct sk_buff *skb, struct Qdisc *sch,
			  struct sk_buff **to_free)
{
	struct taprio_sched *q = qdisc_priv(sch);
	struct Qdisc *child;
	int queue;

	if (unlikely(FULL_OFFLOAD_IS_ENABLED(q->flags))) {
		WARN_ONCE(1, "Trying to enqueue skb into the root of a taprio qdisc configured with full offload\n");
		return qdisc_drop(skb, sch, to_free);
	}
}
```

```
 get_packet_txtime net/sched/sch_taprio.c:508 [inline]
 taprio_enqueue_one+0x881/0x1640 net/sched/sch_taprio.c:577
 taprio_enqueue+0x239/0x7e0 net/sched/sch_taprio.c:658
 dev_qdisc_enqueue+0x3f/0x230 net/core/dev.c:3732
 __dev_xmit_skb net/core/dev.c:3821 [inline]
 __dev_queue_xmit+0x2202/0x3f20 net/core/dev.c:4169
 dev_queue_xmit include/linux/netdevice.h:3088 [inline]
 neigh_hh_output include/net/neighbour.h:528 [inline]
 neigh_output include/net/neighbour.h:542 [inline]
 ip6_finish_output2+0x1083/0x1b20 net/ipv6/ip6_output.c:135
 __ip6_finish_output net/ipv6/ip6_output.c:196 [inline]
 ip6_finish_output+0x485/0x11d0 net/ipv6/ip6_output.c:207
 NF_HOOK_COND include/linux/netfilter.h:292 [inline]
 ip6_output+0x243/0x890 net/ipv6/ip6_output.c:228
 dst_output include/net/dst.h:458 [inline]
 NF_HOOK.constprop.0+0xfd/0x540 include/linux/netfilter.h:303
 mld_sendpack+0x715/0xd60 net/ipv6/mcast.c:1820
 mld_send_cr net/ipv6/mcast.c:2121 [inline]
 mld_ifc_work+0x756/0xcd0 net/ipv6/mcast.c:2653
 process_one_work+0xaa2/0x16f0 kernel/workqueue.c:2597
 worker_thread+0x687/0x1110 kernel/workqueue.c:2748
 kthread+0x33a/0x430 kernel/kthread.c:389
 ret_from_fork+0x2c/0x70 arch/x86/kernel/process.c:145
 ret_from_fork_asm+0x11/0x20 arch/x86/entry/entry_64.S:296
```

 
***linux-5.19 ndo_setup_tc  

```
static struct sk_buff *taprio_dequeue_offload(struct Qdisc *sch)
{
        WARN_ONCE(1, "Trying to dequeue from the root of a taprio qdisc configured with full offload\n");

        return NULL;
}
 static struct sk_buff *taprio_dequeue_offload(struct Qdisc *sch)
 {
 	struct taprio_sched *q = qdisc_priv(sch);
 	struct net_device *dev = qdisc_dev(sch);
 	struct sk_buff *skb;
 	int i;
 
 	for (i = 0; i < dev->num_tx_queues; i  ) {
 		struct Qdisc *child = q->qdiscs[i];
 
 		if (unlikely(!child))
 			continue;
 
 		skb = child->ops->dequeue(child);
 		if (unlikely(!skb))
 			continue;
 
 		qdisc_bstats_update(sch, skb);
 		qdisc_qstats_backlog_dec(sch, skb);
 		sch->q.qlen--;
 
 		return skb;
 	}
 
 	return NULL;
 }
 
 static struct sk_buff *taprio_dequeue(struct Qdisc *sch)
 {
 	struct taprio_sched *q = qdisc_priv(sch);
 
 	return q->dequeue(sch);
 }
 
```

***1）***  在taprio_sched_to_offload根据struct sched_entry生成struct tc_taprio_qopt_offload->(struct tc_taprio_sched_entry）

***2）*** 调用ops->ndo_setup_tc（offload）enetc_setup_taprio

***3）*** enetc_setup_taprio通过dma将tc_taprio_sched_entry传送给网卡硬件

***4）***  q->dequeue = taprio_dequeue_offload返回skb,不需要参考current_entry


 

![image](https://github.com/magnate3/linux-riscv-dev/tree/main/exercises/enet/tc/offload.png)







![image](https://github.com/magnate3/linux-riscv-dev/tree/main/exercises/enet/tc/xmit.png)


> ###  ndo_start_xmit = dpaa2_eth_tx

```
dpaa2_eth_tx  --> __dpaa2_eth_tx  --> priv->enqueue --> dpaa2_io_service_enqueue_multiple_fq
```


## __qdisc_run

```
void __qdisc_run(struct Qdisc *q)
{
        int quota = weight_p;
        while (qdisc_restart(q)) {
                /*
                 * Ordered by possible occurrence: Postpone processing if
                 * 1. we've exceeded packet quota
                 * 2. another process needs the CPU;
                 */
                if (--quota <= 0 || need_resched()) {
                        __netif_schedule(q);
                        break;
                }
        }
        qdisc_run_end(q);
}
```

函数首先获取 weight_p，这个变量通常是通过 sysctl 设置的，收包路径也会用到。我们稍后会看到如何调整此值。这个循环做两件事：

在 while 循环中调用 qdisc_restart，直到它返回 false（或触发下面的中断）
判断 quota 是否小于等于零，或 need_resched()是否返回 true。其中任何一个为真，将调用__netif_schedule 然后跳出循环
注意：用户程序调用 sendmsg 系统调用之后，内核便接管了执行过程，一路执行到这里;用户程序一直在累积系统时间（system time）。如果用户程序在内核中用完其时间 quota
，need_resched 将返回 true。 如果仍有可用 quota，且用户程序的时间片尚未使用，则将再次调用 qdisc_restart。


在上述代码中，我们看到 while 循环不断地从队列中取出 skb 并进行发送。注意，这个时候其实都占用的是用户进程的系统态时间(sy)。只有当 quota 用尽或者其它进程需要 CPU 的时候才触发软中断进行发送。

****所以这就是为什么一般服务器上查看 /proc/softirqs，一般 NET_RX 都要比 NET_TX 大的多的第二个原因。对于读来说，都是要经过 NET_RX 软中断，而对于发送来说，只有系统态配额用尽才让软中断上****

让我们先来看看 qdisc_restart(q)是如何工作的，然后将深入研究__netif_schedule(q)。

## qdisc_restart

```
static inline int qdisc_restart(struct Qdisc *q)
{
        struct netdev_queue *txq;
        struct net_device *dev;
        spinlock_t *root_lock;
        struct sk_buff *skb;
        /* Dequeue packet */
        skb = dequeue_skb(q);  //  taprio_dequeue_offload返回 
        if (unlikely(!skb))
                return 0;
        WARN_ON_ONCE(skb_dst_is_noref(skb));
        root_lock = qdisc_lock(q);
        dev = qdisc_dev(q);
        txq = netdev_get_tx_queue(dev, skb_get_queue_mapping(skb));
        return sch_direct_xmit(skb, q, dev, txq, root_lock);
}
```

qdisc_restart 从队列中取出一个 skb，并调用 sch_direct_xmit 继续发送。

```
//file: net/sched/sch_generic.c
int sch_direct_xmit(struct sk_buff *skb, struct Qdisc *q,
struct net_device *dev, struct netdev_queue *txq,
spinlock_t *root_lock)
{
//调用驱动程序来发送数据
ret = dev_hard_start_xmit(skb, dev, txq);
}
```



## 软中断调度

在 4.5 咱们看到了如果系统态 CPU 发送网络包不够用的时候，会调用 __netif_schedule 触发一个软中断。该函数会进入到 __netif_reschedule，由它来实际发出 NET_TX_SOFTIRQ 类型软中断。

软中断是由内核线程来运行的，该线程会进入到 net_tx_action 函数，在该函数中能获取到发送队列，并也最终调用到驱动程序里的入口函数 dev_hard_start_xmit。


![image](https://github.com/magnate3/linux-riscv-dev/tree/main/exercises/enet/tc/soft.png)

# 内核定时机制API之ns_to_timespec64 和 ns_to_timeval

```
struct timespec64 ns_to_timespec64(const s64 nsec)用于将纳秒转成timespec64格式返回给用户
其源码分析如下：
struct timespec64 ns_to_timespec64(const s64 nsec)
{
	struct timespec64 ts;
	s32 rem;
	#如果形参nsec为null，则让timespec64的两个成员变量都为零
	if (!nsec)
		return (struct timespec64) {0, 0};
	#当纳秒除以NSEC_PER_SEC 得到秒，rem为剩余的纳秒
	ts.tv_sec = div_s64_rem(nsec, NSEC_PER_SEC, &rem);
	if (unlikely(rem < 0)) {
		ts.tv_sec--;
		rem += NSEC_PER_SEC;
	}
	#给timespec64结构体的纳秒赋值
	ts.tv_nsec = rem;
	#返回timespec64 结构体给用户使用
	return ts;
}



struct timeval ns_to_timeval(const s64 nsec)用于将纳秒转成timeval格式返回给用户
其源码分析如下：
struct timeval ns_to_timeval(const s64 nsec)
{
	#首先将形参的纳秒转成timespec
	struct timespec ts = ns_to_timespec(nsec);
	struct timeval tv;
	#然后通过timespec的结构体成员变量赋值给timeval成员变量.
	tv.tv_sec = ts.tv_sec;
	tv.tv_usec = (suseconds_t) ts.tv_nsec / 1000;
	#返回timeval 给用户使用
	return tv;
}
```

#    get_tcp_tstamp

```
 <TASK>
 __dump_stack lib/dump_stack.c:88 [inline]
 dump_stack_lvl+0xcd/0x134 lib/dump_stack.c:106
 ubsan_epilogue+0xb/0x5a lib/ubsan.c:151
 __ubsan_handle_out_of_bounds.cold+0x62/0x6c lib/ubsan.c:291
 ktime_mono_to_any+0x1d4/0x1e0 kernel/time/timekeeping.c:908
 get_tcp_tstamp net/sched/sch_taprio.c:322 [inline]
 get_packet_txtime net/sched/sch_taprio.c:353 [inline]
 taprio_enqueue_one+0x5b0/0x1460 net/sched/sch_taprio.c:420
 taprio_enqueue+0x3b1/0x730 net/sched/sch_taprio.c:485
 dev_qdisc_enqueue+0x40/0x300 net/core/dev.c:3785
 __dev_xmit_skb net/core/dev.c:3869 [inline]
 __dev_queue_xmit+0x1f6e/0x3630 net/core/dev.c:4194
 batadv_send_skb_packet+0x4a9/0x5f0 net/batman-adv/send.c:108
 batadv_iv_ogm_send_to_if net/batman-adv/bat_iv_ogm.c:393 [inline]
 batadv_iv_ogm_emit net/batman-adv/bat_iv_ogm.c:421 [inline]
 batadv_iv_send_outstanding_bat_ogm_packet+0x6d7/0x8e0 net/batman-adv/bat_iv_ogm.c:1701
 process_one_work+0x9b2/0x1690 kernel/workqueue.c:2298
 worker_thread+0x658/0x11f0 kernel/workqueue.c:2445
 kthread+0x405/0x4f0 kernel/kthread.c:327
 ret_from_fork+0x1f/0x30 arch/x86/entry/entry_64.S:295
```

# TXTIME_ASSIST_IS_ENABLED


# CPSW2g Ethernet

## ADMIN and OPER 

#   sja1105_setup_tc_taprio


#  igc_tsn

igc_tsn_enable_offload
drivers/net/ethernet/intel/igc/igc_tsn.c

# references

[CPSW2g Ethernet](https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-jacinto7/latest/exports/docs/linux/Foundational_Components/Kernel/Kernel_Drivers/Network/CPSW2g.html)

[预定流量(EST)卸载的改进](https://blog.csdn.net/chocolate2018/article/details/113937676)

[tc流控attach](https://blog.csdn.net/Megahertz66/article/details/118094425)

[实时机器人应用对Linux通信堆栈的评估](https://www.361shipin.com/blog/1507229726408802306)

[谈一谈Linux让实时/高性能任务独占CPU的事](https://cloud.tencent.com/developer/article/1792712)

[jeez/ Scheduled Tx Tools](https://gist.github.com/jeez/bd3afeff081ba64a695008dd8215866f)


[【干货】25 张图一万字，拆解 Linux 网络包发送过程（下）](https://zhuanlan.zhihu.com/p/397983142)


[Linux 网络栈监控和调优：发送数据（2017）](https://colobu.com/2019/12/09/monitoring-tuning-linux-networking-stack-sending-data/)