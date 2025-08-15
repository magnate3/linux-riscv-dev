
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/hrtimer.h>
#include <linux/jiffies.h>
#include <linux/timekeeper_internal.h>
 
 
static struct hrtimer timer;
ktime_t kt;
static enum hrtimer_restart hrtimer_handler(struct hrtimer *timer)
{

 //ktime_t basenow;
 //ktime_t now;
 //struct hrtimer_cpu_base *cpu_base = this_cpu_ptr(&hrtimer_bases);
 //now = hrtimer_update_base(cpu_base);
 //basenow = ktime_add(now, timer->base->offset);
 u64 now = ktime_to_ns(ktime_get());
 u64 soft = ktime_to_ns(timer->_softexpires);
 u64 expires = ktime_to_ns(timer->node.expires);
 printk("softexpires %llu, expires %llu\n, now %llu , now - expires=  %llu", soft, expires, now, now - expires);
 if(soft != expires)
 {
      pr_info("softexpires %llu, expires %llu\n not equals \n", soft, expires);  
 }
 hrtimer_forward(timer, timer->base->get_time(), kt);
 return HRTIMER_RESTART;
 }
 
static int __init test_init(void)
{
 
 pr_info("timer resolution: %lu\n", TICK_NSEC);
 //kt = ktime_set(1, 10); /* 1 sec, 10 nsec */
 kt = ktime_set(0, 5000000); /* 1 sec, 10 nsec */
 hrtimer_init(&timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
 //hrtimer_set_expires(&timer, kt);
 hrtimer_start(&timer, kt, HRTIMER_MODE_REL);//中断触发周期为:1sec + 10 nsec
 timer.function = hrtimer_handler;
 
 printk("\n hrtimer test start is_soft %x and is_hard %x ---------\n", timer.is_soft, timer.is_hard);
 return 0;
}
 
static void __exit test_exit(void)
{
 hrtimer_cancel(&timer);
 printk("-------- test over ----------\n");
 return;
}
 
MODULE_LICENSE("GPL");
module_init(test_init);
module_exit(test_exit);
