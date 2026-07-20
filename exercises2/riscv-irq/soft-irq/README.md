
# 从硬中断到软中断

关于do_IRQ的实现有很多，不同硬件对中断的处理都会有所不同，但一个基本的执行思路就是：
```
void __irq_entry do_IRQ(unsigned int irq)                                                      |do_IRQ[98]                     void do_IRQ(struct pt_regs *regs, int irq)
{                                                                                              |
        irq_enter();                                                                           |*** arch/sh/kernel/irq.c:                                                             |do_IRQ[185]                    asmlinkage __irq_entry int do_IRQ(unsigned int irq, struct pt_r\
        generic_handle_irq(irq);                                                               |egs *regs)
        irq_exit();                                                                            |
}
```
我们没必要都展开，让我们专注我们的问题。do_IRQ会执行上面e1000_intr这个中断处理函数，这个中断处理是属于上半部的处理，在do_IRQ的结尾会调用irq_exit()，这是软中断和中断衔接的一个地方。我们重点说一下这里。

```
void irq_exit(void)
{
        __irq_exit_rcu();
        rcu_irq_exit();
         /* must be last! */
        lockdep_hardirq_exit();
}

static inline void __irq_exit_rcu(void)
{
#ifndef __ARCH_IRQ_EXIT_IRQS_DISABLED
        local_irq_disable();
#else
        lockdep_assert_irqs_disabled();
#endif
        account_irq_exit_time(current);
        preempt_count_sub(HARDIRQ_OFFSET);
        if (!in_interrupt() && local_softirq_pending())
                invoke_softirq();

        tick_irq_exit();
}
```
在irq_exit()的第一步就是一个local_irq_disable()，也就是说禁止了中断，不再响应中断。因为下面要处理所有标记为要处理的软中断，关中断是因为后面要清除这些软中断，将CPU软中断的位图中置位的位清零，这需要关中断，防止其它进程对位图的修改造成干扰。      

然后preempt_count_sub(HARDIRQ_OFFSET)，硬中断的计数减1，表示当前的硬中断到这里就结束了。但是如果当前的中断是嵌套在其它中断里的话，这次减1后不会计数清0，如果当前只有这一个中断的话，这次减1后计数会清0。注意这很重要。           

因为接下来一步判断!in_interrupt() && local_softirq_pending()，第一个!in_interrupt()就是通过计数来判断当前是否还处于中断上下文中，如果当前还有为完成的中断，则直接退出当前中断。后半部的执行在后续适当的时机再进行，这个“适当的时机”比如ksoftirqd守护进程的调度，或者下次中断到此正好不在中断上下文的时候等情况。      

我们现在假设当前中断结束后没有其它中断了，也就是不在中断上下文了，且当前CPU有等待处理的软中断，即local_softirq_pending()也为真。那么执行invoke_softirq()。   

```
static inline void invoke_softirq(void)
{
        if (ksoftirqd_running(local_softirq_pending()))
                return;

        if (!force_irqthreads) {
#ifdef CONFIG_HAVE_IRQ_EXIT_ON_IRQ_STACK
                /*                                                                                                                                                                             
                 * We can safely execute softirq on the current stack if                                                                                                                       
                 * it is the irq stack, because it should be near empty                                                                                                                        
                 * at this stage.                                                                                                                                                              
                 */
                __do_softirq();
#else
                /*                                                                                                                                                                             
                 * Otherwise, irq_exit() is called on the task stack that can                                                                                                                  
                 * be potentially deep already. So call softirq in its own stack                                                                                                               
                 * to prevent from any overrun.                                                                                                                                                
                 */
                do_softirq_own_stack();
#endif
        } else {
                wakeup_softirqd();
        }
}
```
这个函数的逻辑很简单，首先如果ksoftirqd正在被执行，那么我们不想处理被pending的软中断，交给ksoftirqd线程来处理，这里直接退出。   

如果ksoftirqd没有正在运行，那么判断force_irqthreads，也就是判断是否配置了CONFIG_IRQ_FORCED_THREADING，是否要求强制将软中断处理都交给ksoftirqd线程。因为这里明显要在中断处理退出的最后阶段处理软中断，但是也可以让ksoftirqd来后续处理。如果设置了force_irqthreads，则不再执行__do_softirq()，转而执行wakeup_softirqd()来唤醒ksoftirqd线程，将其加入可运行队列，然后退出。      

如果没有设置force_irqthreads，那么就执行__do_softirq():    
```
asmlinkage __visible void __softirq_entry __do_softirq(void)
{
...
...
        pending = local_softirq_pending();
        account_irq_enter_time(current);

        __local_bh_disable_ip(_RET_IP_, SOFTIRQ_OFFSET);
        in_hardirq = lockdep_softirq_start();

restart:
        /* Reset the pending bitmask before enabling irqs */
        set_softirq_pending(0);

        local_irq_enable();

        h = softirq_vec;

        while ((softirq_bit = ffs(pending))) {
...
...
        }

        if (__this_cpu_read(ksoftirqd) == current)
                rcu_softirq_qs();
        local_irq_disable();

        pending = local_softirq_pending();
        if (pending) {
                if (time_before(jiffies, end) && !need_resched() &&
                    --max_restart)
                        goto restart;

                wakeup_softirqd();
        }

        lockdep_softirq_end(in_hardirq);
        account_irq_exit_time(current);
        __local_bh_enable(SOFTIRQ_OFFSET);
        WARN_ON_ONCE(in_interrupt());
        current_restore_flags(old_flags, PF_MEMALLOC);
}
```
注意在函数开始时就先执行了一个__local_bh_disable_ip(_RET_IP_, SOFTIRQ_OFFSET)，表示当前要处理软中断了，在这种情况下是不允许睡眠的，也就是不能进程调度。这点很重要，也很容易混淆，加上前面我们说的irq_exit()开头的local_irq_disable()，所以当前处在一个既禁止硬中断，又禁止软中断，不能睡眠不能调度的状态。很多人就容易将这种状态归类为“中断上下文”，我个人认为是不对的。从上面in_interrupt函数的定义来看，是否处于中断上下文和preempt_count对于中断的计数有关：
```
#define irq_count()     (preempt_count() & (HARDIRQ_MASK | SOFTIRQ_MASK \
                                 | NMI_MASK))

#define in_interrupt()          (irq_count())
```
和是否禁止了中断没有直接的关系。虽然中断上下文应该不允许睡眠和调度，但是不能睡眠和调度的时候不等于in_interrupt，比如spin_lock的时候也是不能睡眠的（这是目前我个人观点）。但是很多程序员之所以容易一概而论，是因为对于内核程序员来讲，判断自己所编程的位置是否可以睡眠和调度是最被关心的，所以禁用了中断后不能调度和睡眠就很容易被归类为在中断上下文，实际上我个人认为这应该算一个误解，或者说是“变相扩展”后的说辞。一切还要看我们对中断上下文这个概念的界定，如果像in_interrupt那样界定，那关不关中断和是否处于中断上下文就没有直接的关系。   

下面在__do_softirq开始处理软中断（执行每一个待处理的软中断的action）前还有一个很关键的地方，就是local_irq_enable()，这就打开了硬件中断，然后后面的软中断处理可以在允许中断的情况下执行。注意这时候__local_bh_disable_ip(_RET_IP_, SOFTIRQ_OFFSET)仍然有效，睡眠仍然是不允许的。    

到这里我们可以看到，内核是尽量做到能允许中断就尽量允许，能允许调度就尽量允许，因为无情的禁止是对CPU资源最大的浪费，也是对外设中断的不负责。否则长期处于禁止中断的情况下，网卡大量丢包将是难免的，而这也将是制约成网卡实际速率的瓶颈。   

#   sys_call_softirq    
 include/linux/hardirq.h    
```

void sys_call_softirq(void);
```


kernel/softirq.c    
```
void sys_call_softirq(void)
{
#ifndef __ARCH_IRQ_EXIT_IRQS_DISABLED
        local_irq_disable();
#else
        lockdep_assert_irqs_disabled();
#endif

        account_hardirq_exit(current);
        preempt_count_sub(HARDIRQ_OFFSET);
        if (!in_interrupt() && local_softirq_pending())
                invoke_softirq();

}
EXPORT_SYMBOL_GPL(sys_call_softirq);
```
***__do_softirq(void) 会调用local_irq_enable()开中断;***       

##  preempt_count_sub   coredump

```
[  111.518388][ T1111] DEBUG_LOCKS_WARN_ON(val > preempt_count())
[  111.518451][ T1111] WARNING: CPU: 1 PID: 1111 at kernel/sched/core.c:5195 preempt_count_sub+0x8a/0xca
[  111.550517][ T1111] Modules linked in: ioctl_example(OE)
[  111.561599][ T1111] CPU: 1 PID: 1111 Comm: test Tainted: G           OE     5.14.12 #25 80ec36e3b44e109532c1e117c470953f45fbafd7
[  111.585644][ T1111] Hardware name: sifive,hifive-unleashed-a00 (DT)
[  111.598682][ T1111] epc : preempt_count_sub+0x8a/0xca
[  111.609209][ T1111]  ra : preempt_count_sub+0x8a/0xca
[  111.619732][ T1111] epc : ffffffff8003bb2c ra : ffffffff8003bb2c sp : ffffffd0041bbd90
[  111.636227][ T1111]  gp : ffffffff8243ea90 tp : ffffffe083a9d340 t0 : ffffffff82236c90
[  111.652724][ T1111]  t1 : 0000000000000000 t2 : 0000000000000000 s0 : ffffffd0041bbda0
[  111.669220][ T1111]  s1 : ffffffff824420e8 a0 : 000000000000002a a1 : ffffffff81e4d258
[  111.685716][ T1111]  a2 : 0000000000000003 a3 : 0000000000000001 a4 : b18b774e0ff1c400
[  111.702209][ T1111]  a5 : b18b774e0ff1c400 a6 : c0000000ffffefff a7 : 00000000028f5c29
[  111.718705][ T1111]  s2 : ffffffe0801df400 s3 : 0000000000000000 s4 : ffffffff801df400
[  111.735201][ T1111]  s5 : ffffffff824420e8 s6 : 0000003fff910b90 s7 : 0000000000000003
[  111.751695][ T1111]  s8 : ffffffe085afb0e0 s9 : 0000000000000000 s10: 00000000000bb468
[  111.768189][ T1111]  s11: 00000000000e1745 t3 : 00000000000f0000 t4 : ffffffffffffffff
[  111.784684][ T1111]  t5 : ffffffffffffffff t6 : ffffffd0041bbac8
[  111.797184][ T1111] status: 0000000200000100 badaddr: 0000000000000000 cause: 0000000000000003
[  111.815131][ T1111] [<ffffffff8003bb2c>] preempt_count_sub+0x8a/0xca
[  111.828357][ T1111] [<ffffffff80017d9e>] sys_call_softirq+0x2a/0xca
[  111.841426][ T1111] [<ffffffff024ec0a4>] my_ioctl+0xa4/0x130 [ioctl_example]
[  111.856150][ T1111] [<ffffffff80246d64>] sys_ioctl+0x10c/0x8b0
[  111.868307][ T1111] [<ffffffff80003a36>] ret_from_syscall+0x0/0x2
[  111.880994][ T1111] ---[ end trace 914d5c1dfa5098f0 ]---
```



```
 */
#define __irq_enter()                                   \
        do {                                            \
                preempt_count_add(HARDIRQ_OFFSET);      \
                lockdep_hardirq_enter();                \
                account_hardirq_enter(current);         \
        } while (0)
```

## successful test

+ os   (SMP PREEMPT)
```
# uname -a
Linux buildroot 5.14.12 #26 SMP PREEMPT Wed Sep 4 14:28:45 HKT 2024 riscv64 GNU/Linux
# 
```




```
void sys_call_softirq(void)
{
#ifndef __ARCH_IRQ_EXIT_IRQS_DISABLED
        local_irq_disable();
#else
        lockdep_assert_irqs_disabled();
#endif

        //account_hardirq_exit(current);
        //preempt_count_sub(HARDIRQ_OFFSET);
        if (!in_interrupt() && local_softirq_pending())
                invoke_softirq();

}
EXPORT_SYMBOL_GPL(sys_call_softirq);
```

```
ioctl_example.ko
# tftp -g  -r  test 10.11.11.81
# insmod  ioctl_example.ko 
[   65.165408][ T1096] ioctl_example: loading out-of-tree module taints kernel.
[   65.182454][ T1096] ioctl_example: module verification failed: signature and/or required key missing - tainting kernel
[   65.208944][ T1096] Hello, Linux kernel!
[   65.217199][ T1096] ioctl_example - registered Device numver Major: 90, Minor: 0
# mknod /dev/my_device c 90 0
# ./test 
-sh: ./test: Permission denied
# chmod +x ./test 
# ./test 
[  105.560325][ T1101] ioctl_example open was called
[  105.570269][ T1101] ioctl_example - the answer copied: 123
RD_VALUE - answe[  105.582419][ T1101] ioctl_example - updated to the answer: 456
r: 123
[  105.597460][ T1101] ioctl_example - the answer copied: 456
WR_VALUE and RD_[  105.610399][ T1101] ioctl_example - 7 greets to LKM
VALUE - answer: [  105.623529][ T1101] ioctl_example close was called
456
GREETER
succeed to open
# 
```

# test2    

软中断判断   
```
pending = local_softirq_pending();
```

软中断reset   
```
        /* Reset the pending bitmask before enabling irqs */
        set_softirq_pending(0);
```

```
#ifndef local_softirq_pending

#ifndef local_softirq_pending_ref
#define local_softirq_pending_ref irq_stat.__softirq_pending
#endif

#define local_softirq_pending() (__this_cpu_read(local_softirq_pending_ref))
#define set_softirq_pending(x)  (__this_cpu_write(local_softirq_pending_ref, (x)))
#define or_softirq_pending(x)   (__this_cpu_or(local_softirq_pending_ref, (x)))
```


```
 while ((softirq_bit = ffs(pending))) {
                unsigned int vec_nr;
                int prev_count;
                if(NET_RX_SOFTIRQ == softirq_bit)
                {
                    pr_info("NET_RX_SOFTIRQ action trigger , pending %x, %x \n", pending, pending & ~(1 << (softirq_bit - 1)));
                }
                h += softirq_bit - 1;
```


# 网卡软中断


```
static int __init net_dev_init(void)
{
...
...
        open_softirq(NET_TX_SOFTIRQ, net_tx_action);
        open_softirq(NET_RX_SOFTIRQ, net_rx_action);
...
...
}
```

