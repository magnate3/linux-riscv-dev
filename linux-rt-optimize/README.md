

# 网卡优化 

![images](./pic/irq.png)   



## poll（更优）


![images](./pic/poll.png)


#  进程调度API之preempt_count_add/preempt_count_sub



void preempt_count_add(int val) 用于增加当前进程的引用计数，这样可以避免当前进程被抢占
与之对应的是void preempt_count_sub(int val)
用来当前进程的引用计数，这样当引用计数为0时，当前进程就可以被抢占.
这两个函数是一对的，一般一起使用
其使用的例程如下：
``` 
#define __irq_enter()					\
	do {						\
		account_irq_enter_time(current);	\
		preempt_count_add(HARDIRQ_OFFSET);	\
		trace_hardirq_enter();			\
	} while (0)
 
/*
 * Exit irq context without processing softirqs:
 */
#define __irq_exit()					\
	do {						\
		trace_hardirq_exit();			\
		account_irq_exit_time(current);		\
		preempt_count_sub(HARDIRQ_OFFSET);	\
	} while (0)
```
	
可以看到在进入irq是调用preempt_count_add 来增加引用计数避免被抢占，离开irq是调用preempt_count_sub 来减少引用计数使能抢占
其源码分析如下：
```
void preempt_count_add(int val)
{
 
#ifdef CONFIG_DEBUG_PREEMPT
	/*
	 * Underflow?
	 */
	if (DEBUG_LOCKS_WARN_ON((preempt_count() < 0)))
		return;
#endif
	__preempt_count_add(val);
#ifdef CONFIG_DEBUG_PREEMPT
	/*
	 * Spinlock count overflowing soon?
	 */
	DEBUG_LOCKS_WARN_ON((preempt_count() & PREEMPT_MASK) >=
				PREEMPT_MASK - 10);
#endif
	preempt_latency_start(val);
}
```
假定不打开CONFIG_DEBUG_PREEMPT的话，则preempt_count_add 中首先调用__preempt_count_add 来增加引用计数，然后调用preempt_latency_start 来开始
Start timing the latency.这个has如果没有定义CONFIG_DEBUG_PREEMPT 和 CONFIG_PREEMPT_TRACER的话，也等同于空函数.    
```
void preempt_count_sub(int val)
{
#ifdef CONFIG_DEBUG_PREEMPT
	/*
	 * Underflow?
	 */
	if (DEBUG_LOCKS_WARN_ON(val > preempt_count()))
		return;
	/*
	 * Is the spinlock portion underflowing?
	 */
	if (DEBUG_LOCKS_WARN_ON((val < PREEMPT_MASK) &&
			!(preempt_count() & PREEMPT_MASK)))
		return;
#endif
 
	preempt_latency_stop(val);
	__preempt_count_sub(val);
}
```
假定不打开CONFIG_DEBUG_PREEMPT的话，则ppreempt_count_sub 中首先调用preempt_latency_stop 来Stop timing the latency来增加引用计数，然后调用preempt_latency_start 来开始   
 