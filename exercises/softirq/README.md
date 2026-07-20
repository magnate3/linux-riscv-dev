
#  禁止软中断
```
__do_softirq
    /* 取出软中断pending */
    pending = local_softirq_pending();
    /*
     * 禁止软中断 
     * 软中断必须以串行的方式在cpu运行，如果该cpu此时有硬件中断，那么硬中断退出时也会调用软中断处理函数
     * 禁止软中断后，in_interrupt()函数会返回1
    */
    __local_bh_disable_ip(_RET_IP_, SOFTIRQ_OFFSET);
        preempt_count_add(cnt);
```

```
void __local_bh_disable_ip(unsigned long ip, unsigned int cnt)
{
        unsigned long flags;
        int newcnt;

        WARN_ON_ONCE(in_hardirq());

        /* First entry of a task into a BH disabled section? */
        if (!current->softirq_disable_cnt) {
                if (preemptible()) {
                        local_lock(&softirq_ctrl.lock);
                        /* Required to meet the RCU bottomhalf requirements. */
                        rcu_read_lock();
                } else {
                        DEBUG_LOCKS_WARN_ON(this_cpu_read(softirq_ctrl.cnt));
                }
        }

        /*
         * Track the per CPU softirq disabled state. On RT this is per CPU
         * state to allow preemption of bottom half disabled sections.
         */
        newcnt = __this_cpu_add_return(softirq_ctrl.cnt, cnt);
        /*
         * Reflect the result in the task state to prevent recursion on the
         * local lock and to make softirq_count() & al work.
         */
        current->softirq_disable_cnt = newcnt;

        if (IS_ENABLED(CONFIG_TRACE_IRQFLAGS) && newcnt == cnt) {
                raw_local_irq_save(flags);
                lockdep_softirqs_off(ip);
                raw_local_irq_restore(flags);
        }
}
```

#  软中断使能函数local_bh_enable()里面运行软中断处理函数

```
local_bh_enable
    __local_bh_enable_ip(_THIS_IP_, SOFTIRQ_DISABLE_OFFSET);
        /*
         * 禁止本地cpu中断
        */
        #ifdef CONFIG_TRACE_IRQFLAGS
            local_irq_disable();
        #endif
        /*
         * 如果preempt_count里面的软中断字段为SOFTIRQ_DISABLE_OFFSET
        */
        if (softirq_count() == SOFTIRQ_DISABLE_OFFSET)
            trace_softirqs_on(ip)  --- 会使能软中断，但不清楚具体干嘛
        /*
         * 调用软中断回调函数前，
         * 把preempt_count中软中断字段减去SOFTIRQ_DISABLE_OFFSET - 1
        */
        preempt_count_sub(cnt - 1);
        /* 
         * 执行软中断回调函数
        */
        if (unlikely(!in_interrupt() && local_softirq_pending()))
            do_softirq();
                /* 在硬件中断上下文，或者禁止软中断，直接返回 */
                if (in_interrupt())
                    return;
                /*
                 * 又一次禁止cpu本地中断
                */
                local_irq_save(flags);
                pending = local_softirq_pending();
                /*
                 * 如果有待处理的软中断，并且软中断线程没有在执行，那么调用软中断处理函数执行
                */
                if (pending && !ksoftirqd_running())
                    do_softirq_own_stack();
                        __do_softirq();
                /*
                 * 使能cpu本地硬件中断
                */
                local_irq_restore(flags);
        /*
         * 处理完软中断回调函数后，再把preempt_count中软中断字段减去1
        */
        preempt_count_dec();
        /*
         * 使能cpu本地硬件中断
        */
        #ifdef CONFIG_TRACE_IRQFLAGS
            local_irq_enable();
        #endif
        /*
         * 按条件是否执行调度
        */
        preempt_check_resched();
            if (should_resched(0))
                __preempt_schedule();
 
```