
# grep CONFIG_DEBUG_ATOMIC_SLEEP .config
```
CONFIG_DEBUG_ATOMIC_SLEEP=y
CONFIG_PREEMPT_VOLUNTARY .config
#CONFIG_PREEMPT_VOLUNTARY is not set
```

```
BUG: sleeping function called from invalid context at kernel/locking/mutex.c:281
[   72.609925][  T841] in_atomic(): 1, irqs_disabled(): 0, non_block: 0, pid: 841, name: cat
[   72.618720][  T841] CPU: 3 PID: 841 Comm: cat Tainted: G           O      5.14.12-g15c9a49d91df-dirty #196 c12a546bfa86a3a583779ff0b4db20624ac3a1b9
[   72.632454][  T841] Hardware name: sifive,hifive-unleashed-a00 (DT)
[   72.638757][  T841] Call Trace:
[   72.641941][  T841] [<ffffffff800057fe>] walk_stackframe+0x0/0x8a
[   72.648100][  T841] [<ffffffff80766af2>] dump_stack_lvl+0x40/0x5c
[   72.654251][  T841] [<ffffffff800385c6>] ___might_sleep+0xf6/0x10c
[   72.660478][  T841] [<ffffffff807725c0>] mutex_lock+0x2e/0x62
```

此bug发生的原因spin_lock后，调用了mutex_lock。mutex_lock底层调用了___might_sleep，___might_sleep会进行上下文检测

```
1)   spin_lock
2)    mutex_lock-->___might_sleep
3)   spin_unlock
```


**未开启  CONFIG_DEBUG_ATOMIC_SLEEP,不会出现这个bug**

## 未开启  CONFIG_DEBUG_ATOMIC_SLEEP
```
[root@centos7 boot]# grep CONFIG_DEBUG_ATOMIC_SLEEP  config-4.14.0-115.el7a.0.1.aarch64
# CONFIG_DEBUG_ATOMIC_SLEEP is not set
[root@centos7 boot]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
```

在spin lock之前 和 spin lock之后输出如下内容
```
pr_info("%d,irqs_disabled %d,  preempt_count: %d, in atomic %d , pid %d, name %s\n ", preempt_count_equals(0), irqs_disabled(), preempt_count(), in_atomic(), current->pid, current->comm);
```
spin lock之前
```
[13434.004171] 1,irqs_disabled 0,  preempt_count: 0, in atomic 0 , pid 12396, name cat

```
spin lock之后

```
[13434.004172] 1,irqs_disabled 0,  preempt_count: 0, in atomic 0 , pid 12396, name cat
```
in_atomic的定义如下
```
#define in_atomic()  (preempt_count() != 0)
```

##   开启  CONFIG_DEBUG_ATOMIC_SLEEP

# grep CONFIG_DEBUG_ATOMIC_SLEEP .config
```
CONFIG_DEBUG_ATOMIC_SLEEP=y
CONFIG_PREEMPT_VOLUNTARY .config
#CONFIG_PREEMPT_VOLUNTARY is not set
```

spin lock之前
```
[11622.215806][  T863] 1,irqs_disabled 0,  preempt_count: 0, in atomic 0 , pid 863, name cat
```


spin lock之后
```
[11622.215849][  T863] 0,irqs_disabled 0,  preempt_count: 1, in atomic 1 , pid 863, name cat
```
***preempt_count_equals 和  preempt_count  和in_atomic的值发生变化了***

1）只要进程获得了spin_lock的任一个变种形式的lock，那么无论是单处理器系统还是多处理器系统，都会导致preempt_count发生变化。所以preempt_count_equals用来判断是否在spinlock保护的区间里；

2） irqs_disabled判断当前中断是否开启，即是否在中断上下文；


# 内核常用的might_sleep函数


## 1. 前言
　内核版本：linux 4.9.225。内核版本：linux 4.9.225。对于内核常用的might_sleep函数，如果没有调试的需要(没有定义CONFIG_DEBUG_ATOMIC_SLEEP)，这个宏/函数什么事情都不，might_sleep就是一个空函数，所以平常看code的时候可以忽略。内核只是用它来提醒开发人员，调用该函数的函数可能会sleep。

```
[root@centos7 boot]# grep CONFIG_DEBUG_ATOMIC_SLEEP  config-4.14.0-115.el7a.0.1.aarch64
# CONFIG_DEBUG_ATOMIC_SLEEP is not set
[root@centos7 boot]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 boot]# 
```
## 2. might_sleep的定义
```
# include/linux/kernel.
#ifdef CONFIG_PREEMPT_VOLUNTARY
extern int _cond_resched(void);
# define might_resched() _cond_resched()
#else
# define might_resched() do { } while (0)
#endif
 
#ifdef CONFIG_DEBUG_ATOMIC_SLEEP
  void ___might_sleep(const char *file, int line, int preempt_offset);
  void __might_sleep(const char *file, int line, int preempt_offset);
/**
 * might_sleep - annotation for functions that can sleep
 *
 * this macro will print a stack trace if it is executed in an atomic
 * context (spinlock, irq-handler, ...).
 *
 * This is a useful debugging help to be able to catch problems early and not
 * be bitten later when the calling function happens to sleep when it is not
 * supposed to.
 */
# define might_sleep() \
	do { __might_sleep(__FILE__, __LINE__, 0); might_resched(); } while (0)
#else
# define might_sleep() do { might_resched(); } while (0)
#endif
```
## 2.1 未开启CONFIG_DEBUG_ATOMIC_SLEEP选项
在没有调试需求，即在选项 CONFIG_DEBUG_ATOMIC_SLEEP和 CONFIG_PREEMPT_VOLUNTARY(自愿抢占，代码中增加抢占点，在中断退出后遇到抢占点时进行抢占切换)未打开的情况下：
```
# define might_resched() do { } while (0)
# define might_sleep() do { might_resched(); } while (0)
```
可以看到，这里什么事情都没做。其中内核源码对此也有明确的注释：might_sleep - annotation for functions that can sleep。所以对于release版的kernel 而言，might_sleep函数仅仅是一个annotation，用来提醒开发人员，一个使用might_sleep的函数在其后的代码执行中可能会sleep。

## 2.2 开启CONFIG_DEBUG_ATOMIC_SLEEP选项
如果有调试需求的话，就必须打开内核的 CONFIG_DEBUG_ATOMIC_SLEEP选项。 此时might_sleep定义如下：
```
# define might_resched() _cond_resched()
# define might_sleep() \
	do { __might_sleep(__FILE__, __LINE__, 0); might_resched(); } while (0)
```
```
#define in_atomic()     (preempt_count() != 0)
```
CONFIG_DEBUG_ATOMIC_SLEEP选项主要用来排查是否在一个ATOMIC操作的上下文中有函数发生sleep行为，关于什么是 ATOMIC操作，内核源码在might_sleep函数前也有一段注释：this macro will print a stack trace if it is executed in an atomic context ***(spinlock, irq-handler, ...)***

所以，一个进程获得了spinlock之后(进入所谓的atomic context)，或者是在一个irq-handle中(也就是一个中断上下文中)。这两种情况下理论上不应该让当前的execution path进入sleep状态(虽然不是强制规定，换句话说，一个拥有spinlock的进程进入sleep并不必然意味着系统就一定会deadlock 等，但是对内核编程而言，还是应该尽力避开这个雷区)。

在CONFIG_DEBUG_ATOMIC_SLEEP选项打开的情形下，might_sleep具体实现的功能分析如下：
```
# kernel/sched/core.c
void ___might_sleep(const char *file, int line, int preempt_offset)
{
	static unsigned long prev_jiffy;	/* ratelimiting */
	unsigned long preempt_disable_ip;
 
	rcu_sleep_check(); /* WARN_ON_ONCE() by default, no rate limit reqd. */
	if ((preempt_count_equals(preempt_offset) && !irqs_disabled() &&!is_idle_task(current)) ||
	    system_state != SYSTEM_RUNNING || oops_in_progress)  /* 核心判断 */
		return;
	if (time_before(jiffies, prev_jiffy + HZ) && prev_jiffy)
		return;
	prev_jiffy = jiffies;
 
	/* Save this before calling printk(), since that will clobber it */
	preempt_disable_ip = get_preempt_disable_ip(current);
 
	printk(KERN_ERR
		"BUG: sleeping function called from invalid context at %s:%d\n",
			file, line);
	printk(KERN_ERR
		"in_atomic(): %d, irqs_disabled(): %d, pid: %d, name: %s\n",
			in_atomic(), irqs_disabled(),
			current->pid, current->comm);
 
	if (task_stack_end_corrupted(current))
		printk(KERN_EMERG "Thread overran stack, or stack corrupted\n");
 
	debug_show_held_locks(current);
	if (irqs_disabled())
		print_irqtrace_events(current);
	if (IS_ENABLED(CONFIG_DEBUG_PREEMPT)
	    && !preempt_count_equals(preempt_offset)) {
		pr_err("Preemption disabled at:");
		print_ip_sym(preempt_disable_ip);
		pr_cont("\n");
	}
	dump_stack();
	add_taint(TAINT_WARN, LOCKDEP_STILL_OK);
}
EXPORT_SYMBOL(___might_sleep);
```
在当前CONFIG_DEBUG_ATOMIC_SLEEP选项使能的前提下， 可以看到__might_sleep还是干了不少事情的，最主要的工作是在第一个if语句那里，尤其是preempt_count_equals和 irqs_disabled，都是用来判断当前的上下文是否是一个atomic context，
1) 因为我们知道，只要进程获得了spin_lock的任一个变种形式的lock，那么无论是单处理器系统还是多处理器系统，***都会导致 preempt_count发生变化***
2) irq_disabled则是用来判断当前中断是否开启。
__might_sleep正是根据这些信息来判断当前正在执行的代码上下文是否是个atomic，如果不是，那么函数就直接返回了，因为一切正常。。

   所以让CONFIG_DEBUG_ATOMIC_SLEEP选项打开，可以捕捉到在一个atomic context中是否发生了sleep，如果你的代码不小心在某处的确出现了这种情形，那么might_sleep会通过后续的printk以及dump_stack来协助你发现这种情形。

至于__might_sleep函数中的system_state,它是一个全局性的enum型变量，在 /init/main.c中声明，主要用来记录当前系统的状态：

```
# init/main.c
enum system_states system_state __read_mostly;
EXPORT_SYMBOL(system_state);
注意system_state已经被export出来，所以内核模块可以直接读该值来判断当前系统的运行状态，具体定义及常见的状态如下：
```

```
# include\linux\kernel.h
/* Values used for system_state */
extern enum system_states {
	SYSTEM_BOOTING,
	SYSTEM_RUNNING,
	SYSTEM_HALT,
	SYSTEM_POWER_OFF,
	SYSTEM_RESTART,
} system_state;
```
最常见的状态是SYSTEM_RUNNING了，当系统正常起来之后就处于这个状态。

#  spin_lock

```
static __always_inline void spin_lock(spinlock_t *lock)
{
    raw_spin_lock(&lock->rlock);
}
```

```
#define raw_spin_lock(lock)    _raw_spin_lock(lock)
```
Where _raw_spin_lock is defined depends on ***whether CONFIG_SMP option is set and CONFIG_INLINE_SPIN_LOCK option*** is set. ***If the SMP is disabled***, _raw_spin_lock is defined in the include/linux/spinlock_api_up.h header file as a macro and looks like:

```
#define _raw_spin_lock(lock)    __LOCK(lock)
```
If the SMP is enabled and CONFIG_INLINE_SPIN_LOCK is set, it is defined in include/linux/spinlock_api_smp.h header file as the following:

```
#define _raw_spin_lock(lock) __raw_spin_lock(lock)
```

If the SMP is enabled and CONFIG_INLINE_SPIN_LOCK is not set, it is defined in kernel/locking/spinlock.c source code file as the following:

```
void __lockfunc _raw_spin_lock(raw_spinlock_t *lock)
{
    __raw_spin_lock(lock);
}
```
Here we will consider the latter form of _raw_spin_lock. The __raw_spin_lock function looks:


***include/linux/spinlock_api_smp.h***
```
static inline void __raw_spin_lock(raw_spinlock_t *lock)
{
        preempt_disable();
        spin_acquire(&lock->dep_map, 0, 0, _RET_IP_);
        LOCK_CONTENDED(lock, do_raw_spin_trylock, do_raw_spin_lock);
}
```



As you may see, first of all we disable preemption by the call of the preempt_disable macro from the include/linux/preempt.h (more about this you may read in the ninth part of the Linux kernel initialization process chapter). When we unlock the given spinlock, preemption will be enabled again:
```
static inline void __raw_spin_unlock(raw_spinlock_t *lock)
{
        ...
        ...
        ...
        preempt_enable();
}
```

We need to do this to prevent the process from other processes to preempt it while it is spinning on a lock. The spin_acquire macro which through a chain of other macros expands to the call of the:

```
#define spin_acquire(l, s, t, i)                lock_acquire_exclusive(l, s, t, NULL, i)
#define lock_acquire_exclusive(l, s, t, n, i)           lock_acquire(l, s, t, 0, 1, n, i)
```

# references

[Synchronization primitives in the Linux kernel. Part 1.](https://0xax.gitbooks.io/linux-insides/content/SyncPrim/linux-sync-1.html)


