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