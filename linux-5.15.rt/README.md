# schedule(void)
```
asmlinkage __visible void __sched schedule(void)
{
        struct task_struct *tsk = current;

        sched_submit_work(tsk);
        do {
                preempt_disable();
                __schedule(SM_NONE);
                sched_preempt_enable_no_resched();
        } while (need_resched());
        sched_update_worker(tsk);
}
EXPORT_SYMBOL(schedule);
```
# rcu

```
kernel/rcu/tree.c
#define  TEST_ETHCAT_3 1
```

# core.c

```
kernel/sched/core.c
#define TEST_ETHCAT_1 1
```
# rt.c

```
kernel/sched/rt.c
#define TEST_ETHCAT_2 1
```


# softirq.c

```
#define TEST_ETHCAT_4 1
kernel/softirq.c
//CONFIG_PREEMPT_RT
static inline void invoke_softirq(void)
{
	if (should_wake_ksoftirqd())
		wakeup_softirqd();
}
```
## force_irqthreads
```
在硬件中断退出时会调用irq_exit
void irq_exit(void)
{
	if (!in_interrupt() && local_softirq_pending())
		invoke_softirq();
}
在这个函数中我们看到会调用invoke_softirq()来触发软件中断，但是这里有个条件是in_interrupt（）
#define in_interrupt()		(irq_count())
#define irq_count()	(preempt_count() & (HARDIRQ_MASK | SOFTIRQ_MASK \
				 | NMI_MASK))
这里很清楚的可以看到这里的中断上下文包含硬件中断和软件中断
不能处于中断上下文，这里的中断中断上下文分为硬件中断和软件中断
一般在调用irq_exit的时候且没有中断嵌套的时候就不会再硬件中断上下文中
同样如果这次的中断不是发生在一个软件中断中，则也符合不在软件中断上下文中
符合不在中断上下文的条件后还要检测当前是否有软件中断在等待运行即
#define local_softirq_pending() \
	__IRQ_STAT(smp_processor_id(), __softirq_pending)

如果都符合则调用invoke_softirq来触发软件中断

static inline void invoke_softirq(void)
{
	#如果软件中断的守护进程已经开始处理软件中断，则就没有必要再触发软件中断了。直接退出
	if (ksoftirqd_running())
		return;
	#是否强制开机中断中断线程化
	if (!force_irqthreads) {
#ifdef CONFIG_HAVE_IRQ_EXIT_ON_IRQ_STACK
		/*
		 * We can safely execute softirq on the current stack if
		 * it is the irq stack, because it should be near empty
		 * at this stage.
		 */
		 #在irq的自己的stack上处理软件中断，从这里可以知道如果打开CONFIG_HAVE_IRQ_EXIT_ON_IRQ_STACK的话，irq就有自己的stack
		__do_softirq();
#else
		/*
		 * Otherwise, irq_exit() is called on the task stack that can
		 * be potentially deep already. So call softirq in its own stack
		 * to prevent from any overrun.
		 */
		 #在task的stack上开始处理软件中断
		do_softirq_own_stack();
#endif
	} else {
	#使用这个函数来唤醒守护进程来处理软件中断
		wakeup_softirqd();
	}
}
```