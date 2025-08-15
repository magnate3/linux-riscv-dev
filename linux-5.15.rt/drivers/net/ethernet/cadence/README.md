# insmod  macb.ko 

```
root@Ubuntu-riscv64:~# insmod phylink.ko 
root@Ubuntu-riscv64:~# insmod  macb.ko 
```

# rcu_check_gp_kthread_starvation

```
root@Ubuntu-riscv64:~# ping 10.11.11.81
PING 10.11.11.81 (10.11.11.81) 56(84) bytes of data.
[ 1834.503748] rcu: INFO: rcu_preempt detected stalls on CPUs/tasks:
[ 1834.503775] rcu:     3-...!: (4333 GPs behind) idle=536/0/0x0 softirq=0/0 fqs=0  (false positive?)
[ 1834.503865] rcu: rcu_preempt kthread timer wakeup didn't happen for 5251 jiffies! g16165 f0x0 RCU_GP_WAIT_FQS(5) ->state=0x402
[ 1834.503887] rcu:     Possible timer handling issue on cpu=1 timer-softirq=2930
[ 1834.503900] rcu: rcu_preempt kthread starved for 5252 jiffies! g16165 f0x0 RCU_GP_WAIT_FQS(5) ->state=0x402 ->cpu=1
[ 1834.503921] rcu:     Unless rcu_preempt kthread gets sufficient CPU time, OOM is now expected behavior.
[ 1834.503935] rcu: RCU grace-period kthread stack dump:
[ 1834.504098] rcu: Stack dump where RCU GP kthread last ran:
[ 1855.511748] rcu: INFO: rcu_preempt detected stalls on CPUs/tasks:
[ 1855.511769] rcu:     3-...!: (4334 GPs behind) idle=536/0/0x0 softirq=0/0 fqs=0  (false positive?)
[ 1855.511851] rcu: rcu_preempt kthread timer wakeup didn't happen for 5251 jiffies! g16169 f0x0 RCU_GP_WAIT_FQS(5) ->state=0x402
[ 1855.511873] rcu:     Possible timer handling issue on cpu=1 timer-softirq=2931
[ 1855.511885] rcu: rcu_preempt kthread starved for 5252 jiffies! g16169 f0x0 RCU_GP_WAIT_FQS(5) ->state=0x402 ->cpu=1
[ 1855.511907] rcu:     Unless rcu_preempt kthread gets sufficient CPU time, OOM is now expected behavior.
[ 1855.511921] rcu: RCU grace-period kthread stack dump:
[ 1855.512069] rcu: Stack dump where RCU GP kthread last ran:

root@Ubuntu-riscv64:~# dmesg | tail -n 20
[ 1855.512069] rcu: Stack dump where RCU GP kthread last ran:
[ 1855.512077] Task dump for CPU 1:
[ 1855.512083] task:macb poll-#0    state:R  running task     stack:    0 pid: 1272 ppid:     2 flags:0x00000010
[ 1855.512106] Call Trace:
[ 1855.512110] [<ffffffff800059e8>] dump_backtrace+0x30/0x38
[ 1855.512128] [<ffffffff80b1590a>] show_stack+0x40/0x4c
[ 1855.512154] [<ffffffff80037e0e>] sched_show_task+0x196/0x1c0
[ 1855.512172] [<ffffffff80b1619c>] dump_cpu_task+0x56/0x60
[ 1855.512189] [<ffffffff80073cbe>] rcu_check_gp_kthread_starvation+0x120/0x1b4
[ 1855.512208] [<ffffffff8007a560>] rcu_sched_clock_irq+0x770/0xd9e
[ 1855.512227] [<ffffffff80084ca8>] update_process_times+0xca/0x100
[ 1855.512243] [<ffffffff8009516e>] tick_sched_handle.isra.19+0x44/0x56
[ 1855.512266] [<ffffffff8009554e>] tick_sched_timer+0x7a/0xc2
[ 1855.512281] [<ffffffff80085b0c>] __hrtimer_run_queues+0x104/0x2a8
[ 1855.512298] [<ffffffff8008699c>] hrtimer_interrupt+0xf2/0x20e
[ 1855.512313] [<ffffffff80931692>] riscv_timer_interrupt+0x4a/0x56
[ 1855.512331] [<ffffffff8006ac1c>] handle_percpu_devid_irq+0xc0/0x242
[ 1855.512355] [<ffffffff80064d52>] handle_domain_irq+0xa4/0xfc
[ 1855.512370] [<ffffffff80645a6c>] riscv_intc_irq+0x48/0x70
[ 1855.512392] [<ffffffff8000386c>] ret_from_exception+0x0/0xc
root@Ubuntu-riscv64:~#
```

# ping

```

root@Ubuntu-riscv64:~# ip a
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: sit0@NONE: <NOARP> mtu 1480 qdisc noop state DOWN group default qlen 1000
    link/sit 0.0.0.0 brd 0.0.0.0
3: enx00e04c3600d7: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN group default qlen 1000
    link/ether 00:e0:4c:36:00:d7 brd ff:ff:ff:ff:ff:ff
    inet6 fe80::2e0:4cff:fe36:d7/64 scope link 
       valid_lft forever preferred_lft forever
4: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP group default qlen 1000
    link/ether 70:b3:d5:92:fa:3c brd ff:ff:ff:ff:ff:ff
    inet 10.11.11.80/24 brd 10.11.11.255 scope global eth0
       valid_lft forever preferred_lft forever
    inet 169.254.116.29/16 brd 169.254.255.255 scope global noprefixroute eth0
       valid_lft forever preferred_lft forever
    inet6 fe80::72b3:d5ff:fe92:fa3c/64 scope link 
       valid_lft forever preferred_lft forever
root@Ubuntu-riscv64:~# ping 10.11.11.81  
PING 10.11.11.81 (10.11.11.81) 56(84) bytes of data.
64 bytes from 10.11.11.81: icmp_seq=1 ttl=64 time=7.46 ms
^C
--- 10.11.11.81 ping statistics ---
7 packets transmitted, 1 received, 85.7143% packet loss, time 6104ms
rtt min/avg/max/mdev = 7.459/7.459/7.459/0.000 ms
root@Ubuntu-riscv64:~# 
```

# macb_is_gem
```
     /* setup appropriated routines according to adapter type */
        if (macb_is_gem(bp)) {
                dev_err(&pdev->dev, "macb is gem \n");
                bp->max_tx_length = GEM_MAX_TX_LEN;
                bp->macbgem_ops.mog_alloc_rx_buffers = gem_alloc_rx_buffers;
                bp->macbgem_ops.mog_free_rx_buffers = gem_free_rx_buffers;
                bp->macbgem_ops.mog_init_rings = gem_init_rings;
                bp->macbgem_ops.mog_rx = gem_rx;
                dev->ethtool_ops = &gem_ethtool_ops;
        } else {
                dev_err(&pdev->dev, "macb is not  gem \n");
                bp->max_tx_length = MACB_MAX_TX_LEN;
                bp->macbgem_ops.mog_alloc_rx_buffers = macb_alloc_rx_buffers;
                bp->macbgem_ops.mog_free_rx_buffers = macb_free_rx_buffers;
                bp->macbgem_ops.mog_init_rings = macb_init_rings;
                bp->macbgem_ops.mog_rx = macb_rx;
                dev->ethtool_ops = &macb_ethtool_ops;
        }
```
```
[ 3898.170213] macb 10090000.ethernet: not need to register interrupt 
[ 3898.170221] macb 10090000.ethernet: macb is gem 
```

# napi

```
netif_napi_add(dev, &queue->napi, macb_poll, NAPI_POLL_WEIGHT);
```
## macb_poll
```
macb_poll
```

##  napi_disable   napi_enable



# 

```
# ./tcpdrop.py
TIME     PID    IP SADDR:SPORT          > DADDR:DPORT          STATE (FLAGS)
20:49:06 0      4  10.32.119.56:443     > 10.66.65.252:22912   CLOSE (ACK)
	tcp_drop+0x1
	tcp_v4_do_rcv+0x135
	tcp_v4_rcv+0x9c7
	ip_local_deliver_finish+0x62
	ip_local_deliver+0x6f
	ip_rcv_finish+0x129
	ip_rcv+0x28f
	__netif_receive_skb_core+0x432
	__netif_receive_skb+0x18
	netif_receive_skb_internal+0x37
	napi_gro_receive+0xc5
	ena_clean_rx_irq+0x3c3
	ena_io_poll+0x33f
	net_rx_action+0x140
	__softirqentry_text_start+0xdf
	irq_exit+0xb6
	do_IRQ+0x82
	ret_from_intr+0x0
	native_safe_halt+0x6
	default_idle+0x20
	arch_cpu_idle+0x15
	default_idle_call+0x23
	do_idle+0x17f
	cpu_startup_entry+0x73
	rest_init+0xae
	start_kernel+0x4dc
	x86_64_start_reservations+0x24
	x86_64_start_kernel+0x74
	secondary_startup_64+0xa5
```

# interrupt

##   bp->rx_intr_mask | MACB_TX_INT_FLAGS

```
#define MACB_RX_INT_FLAGS	(MACB_BIT(RCOMP) | MACB_BIT(RXUBR)	\
				 | MACB_BIT(ISR_ROVR))

#define MACB_TX_INT_FLAGS	(MACB_TX_ERR_FLAGS | MACB_BIT(TCOMP))
```

```

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
```
