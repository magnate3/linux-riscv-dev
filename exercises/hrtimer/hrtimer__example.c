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
#include <linux/fs.h>
#include <linux/device.h>

#define HRTIMER_TEST_PIN 7

#define HRTIMER_TEST_CYCLE   0, (100000 / 2)

#define DEVICE_NAME    "HRTIMER_TEST"
#define CLASS_NAME    "HRTIMER_TEST"

int major_number;
struct device *device;
struct class *class;
static struct hrtimer kthread_timer;
int value = 0;

enum hrtimer_restart hrtimer_cb_func(struct hrtimer *timer) {
    value = !value;

    hrtimer_forward(timer, timer->base->get_time(), ktime_set(HRTIMER_TEST_CYCLE));
    return HRTIMER_RESTART;
}

void kthread_hrtimer_init(void) {
    hrtimer_init(&kthread_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    kthread_timer.function = hrtimer_cb_func;
    hrtimer_start(&kthread_timer, ktime_set(HRTIMER_TEST_CYCLE), HRTIMER_MODE_REL);
}

static int __init hrtimer_test_init(void) {
    printk(KERN_ALERT "hrtimer_test : Init !!\n");

    major_number = register_chrdev(0, DEVICE_NAME, NULL);

    if (major_number < 0) {
        printk(KERN_ALERT "hrtimer_test: Register fail!\n");
        return major_number;
    }

    printk(KERN_ALERT "Registe success, major number is %d\n", major_number);

    class = class_create(THIS_MODULE, CLASS_NAME);

    if (IS_ERR(class)) {
        unregister_chrdev(major_number, DEVICE_NAME);
        return PTR_ERR(class);
    }

    device = device_create(class, NULL, MKDEV(major_number, 0), NULL, DEVICE_NAME);

    if (IS_ERR(device)) {
        class_destroy(class);
        unregister_chrdev(major_number, DEVICE_NAME);
        return PTR_ERR(device);
    }

    printk(KERN_ALERT "hrtimer_test: init success!!\n");

    kthread_hrtimer_init();

    return 0;
}

static void __exit hrtimer_test_exit(void) {

    hrtimer_cancel(&kthread_timer);

    device_destroy(class, MKDEV(major_number, 0));
    class_unregister(class);
    class_destroy(class);
    unregister_chrdev(major_number, DEVICE_NAME);

    printk(KERN_ALERT "hrtimer_test: exit success!!\n");
}

module_init(hrtimer_test_init);
module_exit(hrtimer_test_exit);

MODULE_AUTHOR("RieChen");
MODULE_LICENSE("GPL");
