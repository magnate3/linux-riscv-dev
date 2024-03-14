
# clocksource & timekeeper

```
static void __init mct_init_dt(struct device_node *np, unsigned int int_type)
{
	exynos4_timer_resources(np, of_iomap(np, 0)); //(1)初始化localtimer，并将其注册成clockevent
	exynos4_clocksource_init(); //(2)初始化globaltimer，并将其注册成clocksource
	exynos4_clockevent_init(); //(3)将globaltimer的comparator 0注册成一个clockevent，一般不会使用
}
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hrtimer/clock.png)

上图描述的是clocksource和timekeeper的关系：
一个global timer对应注册一个clocksource。

一个系统中可以有多个clocksource，timekeeper选择精度最高的那个来使用。

用户使用timekeeper提供的接口来获取系统的时间戳。

为了避免无人主动获取时间clocksource定时器的溢出，timekeeper需要定期的去获取clocksource的值来更新系统时间，一般是在tick处理中更新。

##  timekeeper选择  clocksource

```
int __clocksource_register_scale(struct clocksource *cs, u32 scale, u32 freq)
{

	/* Initialize mult/shift and max_idle_ns */
	/* (1.1) 根据timer的频率freq，计算cs->mult、cs->shift
	    这两个字段是用来把timer的计数转换成实际时间单位ns
	    ns = (count * cs->mult) >> cs->shift */
	__clocksource_update_freq_scale(cs, scale, freq);

	/* Add clocksource to the clocksource list */
	mutex_lock(&clocksource_mutex);
	/* (1.2) 将新的clocksource加入全局链表 */
	clocksource_enqueue(cs);
	clocksource_enqueue_watchdog(cs);
	/* (1.3) 从全局链表中重新选择一个best
	    clocksource给timekeeper使用 */
	clocksource_select();
	clocksource_select_watchdog(false);
	mutex_unlock(&clocksource_mutex);
	return 0;
}
```


# Tickless 和 CPUIdle 的关系

Tickless 是指动态时钟，即系统的周期 Tick 可动态地关闭和打开。这个功能可通过内核配置项 CONFIG_NO_HZ 打开，
***而 Idle 正是使用了这项技术，使系统尽量长时间处于空闲状态，从而尽可能地节省功耗.***
打开内核配置项 CONFIG_NO_HZ_IDLE，即可让系统在 Idle 前关闭周期 Tick，退出 Idle 时重新打开周期 Tick。
那么在关闭了周期 Tick 之后，系统何时被唤醒呢？
在关闭周期 Tick 时，同时会根据时钟子系统计算下一个时钟中断到来的时间，以这个时间为基准来设置一个 hrtimer 用于唤醒系统（高精度时钟框架），而这个时间的计算方法也很简单，即在所有注册到时钟框架的定时器中找到离此时最近的那一个的时间点作为这个时间。当然，用什么定时器来唤醒系统还要根据 CPU Idle 的深度来决定，后面会介绍。
不同层级的 CPU Idle 对唤醒时钟源的处理
前面提到了，系统关闭周期 Tick 的同时，会计算出下一个时钟中断到来的时间，以这个时间为基准来设置一个 hrtimer 用于唤醒系统。
那么，如果有些 CPU 进入的层级比较深，关闭了 CPU 中的 hrtimer，系统将无法再次被唤醒。


针对这种情况，则需要低功耗 Timer 去唤醒系统，这里先以 MTK 平台为例，
在 CPU 进入 dpidle 和 soidle （两种 Idle 模式）时都会关闭 hrtimer ，
另外起用一个 GPT Timer，而这个 GPT Timer 的超时时间直接从被关闭的 hrtimer 中的寄存器获取。
这样就保证时间的延续性。因为 GPT Timer 是以 32K 晶振作为时钟源，
所以在 CPU 进入 dpidle 时可以把 26M 的主时钟源给关闭，从而达到最大程度的省电。

***GPT Timer是以 32K 晶振,频率比26M 的主时钟源小***

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hrtimer/tick1.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hrtimer/tick2.png)


# hrtimer

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hrtimer/htimer.png)

## hrtimer_switch_to_hres

```
 static void hrtimer_switch_to_hres(void)
{
	struct hrtimer_cpu_base *base = this_cpu_ptr(&hrtimer_bases);
 
        /* 切换到高精度模式 */
	if (tick_init_highres()) {
		pr_warn("Could not switch to high resolution mode on CPU %u\n",
			base->cpu);
		return;
	}
        /* 设置本CPU对应的hrtimer_cpu_base结构体的hres_active字段表明进入高精度模式 */
	base->hres_active = 1;
	hrtimer_resolution = HIGH_RES_NSEC;
 
        /* 设置Tick模拟层 */
	tick_setup_sched_timer();
	/* 对定时事件设备进行重编程 */
	retrigger_next_event(NULL);
}
```

该函数调用 tick_init_highres 函数，切换到高精度模式：

```
 int tick_init_highres(void)
{
        /* 将定时事件设备的中断处理程序设置成hrtimer_interrupt */
	return tick_switch_to_oneshot(hrtimer_interrupt);
}
```
该函数调用 tick_switch_to_oneshot 函数，将定时事件设备切换到单次触发模式，并将中断到期处理函数设置成 hrtimer_interrupt：

```
 int tick_switch_to_oneshot(void (*handler)(struct clock_event_device *))
{
	struct tick_device *td = this_cpu_ptr(&tick_cpu_device);
	struct clock_event_device *dev = td->evtdev;
 
	if (!dev || !(dev->features & CLOCK_EVT_FEAT_ONESHOT) ||
		    !tick_device_is_functional(dev)) {
 
		pr_info("Clockevents: could not switch to one-shot mode:");
		if (!dev) {
			pr_cont(" no tick device\n");
		} else {
			if (!tick_device_is_functional(dev))
				pr_cont(" %s is not functional.\n", dev->name);
			else
				pr_cont(" %s does not support one-shot mode.\n",
					dev->name);
		}
		return -EINVAL;
	}
 
        /* 将当前CPU对应Tick设备的模式切换成TICKDEV_MODE_ONESHOT */
	td->mode = TICKDEV_MODE_ONESHOT;
        /* 设置新的事件处理函数 */
	dev->event_handler = handler;
        /* 将定时事件设备切换到单次触发模式 */
	clockevents_switch_state(dev, CLOCK_EVT_STATE_ONESHOT);
        /* 通知Tick广播层切换到单次触发模式 */
	tick_broadcast_switch_to_oneshot();
	return 0;
}
```

一旦切换到了高精度模式，那底层的定时事件设备就一定会被切换到单次触发模式，而且到期后中断处理程序不再会调用Tick层的 tick_handle_periodic，而是换成了高分辨率定时器层的hrtimer_interrupt 函数。一旦完成了切换也就意味着从此周期触发的 Tick 将不复存在了，如果此时不对底层的定时事件设备进行重编程，那么它就永远不会再次被触发。因此，在切换成功后，还必须要找到最近到期的定时器，并用它的到期事件对定时事件设备进行重编程：


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/hrtimer/timer.png)

***低精度***
```
(gdb) bt
#0  hrtimer_run_queues () at kernel/time/hrtimer.c:1895
#1  0xffffffff811c0f94 in run_local_timers () at kernel/time/timer.c:2047
#2  update_process_times (user_tick=0) at kernel/time/timer.c:2070
#3  0xffffffff811d7aa5 in tick_sched_handle (ts=ts@entry=0xffff888237d23900, regs=<optimized out>) at ./arch/x86/include/asm/ptrace.h:136
#4  0xffffffff811d7b61 in tick_sched_timer (timer=0xffff888237d23900) at kernel/time/tick-sched.c:1490
#5  0xffffffff811c2008 in __run_hrtimer (flags=6, now=0xffffc9000014cf30, timer=0xffff888237d23900, base=0xffff888237d233c0, cpu_base=0xffff888237d23380)
    at kernel/time/hrtimer.c:1685
#6  __hrtimer_run_queues (cpu_base=cpu_base@entry=0xffff888237d23380, now=2432552332858, flags=flags@entry=6, active_mask=active_mask@entry=15)
    at kernel/time/hrtimer.c:1749
#7  0xffffffff811c2b89 in hrtimer_interrupt (dev=<optimized out>) at kernel/time/hrtimer.c:1811
#8  0xffffffff8109272e in local_apic_timer_interrupt () at arch/x86/kernel/apic/apic.c:1095
#9  __sysvec_apic_timer_interrupt (regs=<optimized out>) at arch/x86/kernel/apic/apic.c:1112
#10 0xffffffff820261eb in sysvec_apic_timer_interrupt (regs=0xffffc900000bbdc8) at arch/x86/kernel/apic/apic.c:1106
Backtrace stopped: previous frame inner to this frame (corrupt stack?)
```



***高精度***
```
(gdb) bt
#0  __hrtimer_run_queues (cpu_base=cpu_base@entry=0xffff888237c23380, now=2358938282822, flags=flags@entry=6, active_mask=active_mask@entry=15)
    at kernel/time/hrtimer.c:1719
#1  0xffffffff811c2b89 in hrtimer_interrupt (dev=<optimized out>) at kernel/time/hrtimer.c:1811
#2  0xffffffff8109272e in local_apic_timer_interrupt () at arch/x86/kernel/apic/apic.c:1095
#3  __sysvec_apic_timer_interrupt (regs=<optimized out>) at arch/x86/kernel/apic/apic.c:1112
#4  0xffffffff820261eb in sysvec_apic_timer_interrupt (regs=0xffffffff82c03d28) at arch/x86/kernel/apic/apic.c:1106
#5  0xffffffff82200ecb in asm_sysvec_apic_timer_interrupt () at ./arch/x86/include/asm/idtentry.h:645
#6  0x0000000000000000 in ?? ()
```

## tick_sched_timer

**1)**  tick_setup_sched_timer   

```
static int hrtimer_switch_to_hres(void)
   |----......
   |---->tick_init_highres()
        |---->tick_switch_to_oneshot(hrtimer_interrupt) 
        |     设置clock_event_device的event_handler为
      |    hrtimer_interrupt
  |----......
   |---->tick_setup_sched_timer();
   |     这个函数使用tick_cpu_sched这个per-CPU变量来模拟原来
   |    tick device的功能。tick_cpu_sched本身绑定了
   |    一个hrtimer，这个hrtimer的超时值为下一个tick，
   |     回调函数为tick_sched_timer。因此，每过一个
   |     tick，tick_sched_timer就会被调用一次，在这个回调函数中首先
   |     完成原来tick device的工作，然后设置下一次的超时值为再下一个
   |     tick，从而达到了模拟周期运行的tick device的功能。如果所有的
   |     CPU在同一时间点被唤醒，并发执行tick时可能会出现。
```
**1)**  tick_sched_timer    

```
tick_sched_timer -->  tick_sched_handle -->  update_process_times
```

```

static enum hrtimer_restart tick_sched_timer(struct hrtimer *timer)
{

    /*tick_sched_do_timer主要的职责是根据当前时间来更新系统jiffies*/
	tick_sched_do_timer(ts, now);
 
	/* 是否在中断上下文中 */
	if (regs)
		tick_sched_handle(ts, regs);

}
```

# 如何检测你的定时器系统是否支持高精度定时器
有许多种方式可以判定你的系统是否支持高精度定时器

*1)* 检查内核的启动信息
看内核的启动信息或者使用 dmesg。 如果内核成功打开了高精度定时器，在启动的时候会打印信息： Switched to high resolution mode on CPU0 (或者相似的信息)。

*2)*查看 /proc/timer_list
也可以查看 timer_list，可以看到列出的时钟是否支持高精度。下面列出了一份在 OSK(ARM系列开发板)的 /proc/timer_list，显示了时钟被配置成高精度




```
cpu: 0
 clock 0:
  .base:       ffff803ff79d8bc0
  .index:      0
  .resolution: 1 nsecs
  .get_time:   ktime_get
  .offset:     0 nsecs
active timers:
 #0: <ffff803ff79d9010>, tick_sched_timer, S:01
 # expires at 99372980000000-99372980000000 nsecs [in 195164698 to 195164698 nsecs]
 #1: root_task_group, sched_rt_period_timer, S:01
 # expires at 99373020006507-99373020006507 nsecs [in 235171205 to 235171205 nsecs]
 #2: <ffff803ff79d9168>, watchdog_timer_fn, S:01
 # expires at 99376020000000-99376020000000 nsecs [in 3235164698 to 3235164698 nsecs]
 #3: <ffff00002de8fce0>, hrtimer_wakeup, S:01
 # expires at 99424485772807-99424545515806 nsecs [in 51700937505 to 51760680504 nsecs]
 #4: sched_clock_timer, sched_clock_poll, S:01
 # expires at 101155069755300-101155069755300 nsecs [in 1782284919998 to 1782284919998 nsecs]
 clock 1:
  .base:       ffff803ff79d8c00
  .index:      1
  .resolution: 1 nsecs
  .get_time:   ktime_get_real
  .offset:     1665542767218829404 nsecs
active timers:
 clock 2:
  .base:       ffff803ff79d8c40
  .index:      2
  .resolution: 1 nsecs
  .get_time:   ktime_get_boottime
  .offset:     0 nsecs
active timers:
 clock 3:
  .base:       ffff803ff79d8c80
  .index:      3
  .resolution: 1 nsecs
  .get_time:   ktime_get_clocktai
  .offset:     1665542767218829404 nsecs
active timers:
  .expires_next   : 99372980000000 nsecs
  .hres_active    : 1
  .nr_events      : 1399643
  .nr_retries     : 25327
  .nr_hangs       : 0
  .max_hang_time  : 0
  .nohz_mode      : 2
  .last_tick      : 99372780000000 nsecs
  .tick_stopped   : 1
  .idle_jiffies   : 4304874574
  .idle_calls     : 11274818
  .idle_sleeps    : 11134814
  .idle_entrytime : 99372780049512 nsecs
  .idle_waketime  : 99372780004031 nsecs
  .idle_exittime  : 99372770050195 nsecs
  .idle_sleeptime : 98281267294200 nsecs
  .iowait_sleeptime: 245900673 nsecs
  .last_jiffies   : 4304874575
  .next_timer     : 99372980000000
  .idle_expires   : 99372980000000 nsecs
jiffies: 4304874575
```


下面有一些需要检查的事项:

检查你的时钟分辨率的报告。 如果你的时钟支持高精度，那么 .resolution 值将是多少个 ns。如果不支持的话，.resolution 值将等于 1 个 tick 对应的纳秒数（在嵌入式平台通常都是 10000ns）。

检查时钟设备的 event_handler。 如果事件处理例程是 hrtimer_interrupt 时钟被设置成高精度。如果事件处理例程是 tick_handle_periodic， 那么时钟设备会设置成有规律的滴答。

检查 timers 的列表，看属性 .hres_active 的值是否为 1。 如果是 1，高精度定时器的特性已经被激活了




# references

[Linux时间子系统之（十三）：Tick Device layer综述](http://www.wowotech.net/timer_subsystem/tick-device-layer.html)


[浅析linux 内核 高精度定时器（hrtimer）实现机制（二）](https://zhuanlan.zhihu.com/p/544513145)


[FreeRTOS笔记——低功耗 Tickless 模式](https://codeantenna.com/a/ZQkOxERV3i)