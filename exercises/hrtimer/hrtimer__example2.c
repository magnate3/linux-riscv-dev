#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/hrtimer.h>
#include <linux/ktime.h>
#include <linux/sched.h> 
#include <linux/time.h>
#include <linux/hrtimer.h>
#include <linux/types.h>

#define KTIME_ARR_SIZE 16
#define MILLION 1000000

ktime_t ktime_arr[KTIME_ARR_SIZE];
struct hrtimer timer;
struct hrtimer kthread_timer;
struct task_struct * kthread;
struct task_struct * internal_kthread;
atomic_t barrier;

struct sched_param{
	int sched_priority;
};


static int internal_kthread_fn(void *data) {

    int i = 0; 
    printk(KERN_INFO "internal_kthread_fn run!\n");
    for (i = 0; i < MILLION; i++) {

        ktime_get();
    }

    return 0;
}


static enum hrtimer_restart hrtimer_fn(struct hrtimer * timer) {
    printk(KERN_INFO "hrtimer_restart hrtimer_fn run!\n");
    ktime_arr[3] = ktime_get();

    /* wake up kthread
    */
    wake_up_process(kthread);

    printk(KERN_INFO "This is in hrtimer_fn!\n");

    return HRTIMER_NORESTART;
}

static enum hrtimer_restart kthread_hrtimer_fn(struct hrtimer * kthread_timer){
	printk(KERN_INFO "hrtimer_restart kthread_hrtimer_fn run!\n");

    ktime_arr[6] = ktime_get();

    wake_up_process(kthread);

    return HRTIMER_NORESTART;
}


static int kthread_fn(void *data) {

    struct sched_param param;
    param.sched_priority = 2;

    ktime_arr[4] = ktime_get();

    internal_kthread = kthread_create(internal_kthread_fn, NULL, "internal_kthread");
    kthread_bind(internal_kthread, 0);
    sched_setscheduler(internal_kthread, SCHED_FIFO, &param);
    wake_up_process(internal_kthread);

    hrtimer_init(&kthread_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    kthread_timer.function = &kthread_hrtimer_fn;
    hrtimer_start(&kthread_timer, ktime_set(1, 0), HRTIMER_MODE_REL);

    printk(KERN_INFO "This is kthread: %s function!\n", current->comm);

    set_current_state(TASK_INTERRUPTIBLE);

    ktime_arr[5] = ktime_get();

    schedule();

    ktime_arr[7] = ktime_get();

    return 0;
}

static int timing_init (void) {

    int i = 0; 
    
    struct sched_param param;
    param.sched_priority = 5;

    printk(KERN_ALERT "timing module is being loaded!\n");

    for (i = 0; i < KTIME_ARR_SIZE; i++) {

        ktime_arr[i] = ktime_set(0, 0);
    }

    /* kthrread creation 
    */
    ktime_arr[0] = ktime_get();
    kthread = kthread_create(kthread_fn, NULL, "kthread_timing");

    ktime_arr[1] = ktime_get();
    sched_setscheduler(kthread, SCHED_FIFO, &param);
    ktime_arr[2] = ktime_get();

    /*pin kthread to a specific core
    */
    kthread_bind(kthread, 0);

    /* hrtimer init and start
    */
    hrtimer_init(&timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    timer.function = &hrtimer_fn;
    hrtimer_start(&timer, ktime_set(1, 0), HRTIMER_MODE_REL);

    return 0;

}

static void timing_exit (void) {

    int i = 0;
    struct timespec ts;
    struct timespec zero_one_diff;
    struct timespec one_two_diff;
    struct timespec zero_four_diff;
    struct timespec three_five_diff;
    struct timespec six_seven_diff;

    hrtimer_cancel(&kthread_timer);
    hrtimer_cancel(&timer);
    printk(KERN_INFO "Two timers have been cancelled!\n");

    for (i = 0; i < KTIME_ARR_SIZE; i++) {

        ts = ktime_to_timespec(ktime_arr[i]);
        printk(KERN_INFO "ktime[%d] = %lld.%.9ld\n", i, (long long)ts.tv_sec, ts.tv_nsec);
    }

    zero_one_diff = ktime_to_timespec(ktime_sub(ktime_arr[1], ktime_arr[0]));
    one_two_diff = ktime_to_timespec(ktime_sub(ktime_arr[2], ktime_arr[1]));
    zero_four_diff = ktime_to_timespec(ktime_sub(ktime_arr[4], ktime_arr[0]));
    three_five_diff = ktime_to_timespec(ktime_sub(ktime_arr[5], ktime_arr[3]));
    six_seven_diff = ktime_to_timespec(ktime_sub(ktime_arr[7], ktime_arr[6]));

    printk(KERN_INFO "latency for creating kthread: %lld.%.9ld\n",  (long long)zero_one_diff.tv_sec, zero_one_diff.tv_nsec);
    printk(KERN_INFO "latency for set kthread policy: %lld.%.9ld\n", (long long)one_two_diff.tv_sec, one_two_diff.tv_nsec);
    printk(KERN_INFO "time interval for thread creation and run: %lld.%.9ld\n", (long long)zero_four_diff.tv_sec, zero_four_diff.tv_nsec);
    printk(KERN_INFO "latency to wake up a sleeping thread: %lld.%.9ld\n", (long long)three_five_diff.tv_sec, three_five_diff.tv_nsec);
    printk(KERN_INFO "time of kthread prempting internal_kthread: %lld.%.9ld\n", (long long)six_seven_diff.tv_sec, six_seven_diff.tv_nsec);
	
    printk(KERN_ALERT "timing module is being unloaded\n");
}

module_init (timing_init);
module_exit (timing_exit);

MODULE_LICENSE ("GPL");
MODULE_AUTHOR ("Jiangnan Liu, Qitao Xu, Zhe Wang");
MODULE_DESCRIPTION ("Observing Timing Events");