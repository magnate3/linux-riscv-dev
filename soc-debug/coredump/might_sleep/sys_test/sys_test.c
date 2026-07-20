#include <linux/device.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sysfs.h>
#include <linux/spinlock.h> 

static struct device *dev;
static struct kobject *root, *phy;
#define  CONFIG_DEBUG_ATOMIC_SLEEP
#ifdef CONFIG_DEBUG_ATOMIC_SLEEP

static inline int preempt_count_equals(int preempt_offset)
{
        int nested = preempt_count() + rcu_preempt_depth();

        return (nested == preempt_offset);
}
#endif
#define SPIN_LOCK_UNLOCKED 1
spinlock_t              lock;
static DEFINE_MUTEX(test_phy_mutex);
static ssize_t dev1_show(struct device *dev, struct device_attribute *attr, char *buf) {
    pr_info("before spinlock, equals %d,irqs_disabled %d,  preempt_count: %d, in atomic %d , pid %d, name %s", preempt_count_equals(0), irqs_disabled(), preempt_count(), in_atomic(), current->pid, current->comm);
    spin_lock(&lock);
    pr_info("spin lock \n");
    pr_info("after  spinlock, equals %d,irqs_disabled %d,  preempt_count: %d, in atomic %d , pid %d, name %s", preempt_count_equals(0), irqs_disabled(), preempt_count(), in_atomic(), current->pid, current->comm);
    mutex_lock(&test_phy_mutex);
    mutex_unlock(&test_phy_mutex);
    pr_info("spin unlock \n ");
    
    spin_unlock(&lock);
    return sprintf(buf, "show over \n");
}

static ssize_t dev1_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count) {
    pr_info("preempt_count_equals 0 : %d,  irqs_disabled %d,  preempt_count: %d, in atomic %d , pid %d, name %s\n ", preempt_count_equals(0), irqs_disabled(), preempt_count(), in_atomic(), current->pid, current->comm);

}

static struct device_attribute dev_attr = __ATTR(dev1, 0660, dev1_show, dev1_store);

static int sysfs_demo_init(void) {
    spin_lock_init(&lock);
    dev = root_device_register("sysfs_mar_phy");
    root = &dev->kobj;
    phy = kobject_create_and_add("phyDbg", root);
    sysfs_create_file(phy, &dev_attr.attr);
    return 0;
}

static void sysfs_demo_exit(void) {
    printk(KERN_INFO "sysfs marvel exit\n");
    sysfs_remove_file(root, &dev_attr.attr);
    kobject_put(phy);
    root_device_unregister(dev);
}
module_init(sysfs_demo_init);
module_exit(sysfs_demo_exit);
MODULE_LICENSE("GPL");
