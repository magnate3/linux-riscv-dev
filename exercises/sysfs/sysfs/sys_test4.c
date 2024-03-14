#include <linux/device.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sysfs.h>

static int dev_int;
static struct device *dev;
static struct kobject *root, *s1, *s2, *s3;

static ssize_t dev1_show(struct device *dev, struct device_attribute *attr, char *buf) {
    return sprintf(buf, "dev_int: %d\n", dev_int);
}

static ssize_t dev1_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count) {
    sscanf(buf, "%d", &dev_int);
    if (dev_int == 0) {
        kobject_del(s3);
        sysfs_remove_link(s2, "symlink_demo");
    }
    return count;
}

static struct device_attribute dev_attr = __ATTR(dev1, 0660, dev1_show, dev1_store);

static int sysfs_demo_init(void) {
    printk(KERN_INFO "sysfs demo init\n");

    dev = root_device_register("sysfs_demo");
    root = &dev->kobj;
    s1 = kobject_create_and_add("subdir1", root);
    s2 = kobject_create_and_add("subdir2", s1);
    s3 = kobject_create_and_add("subdir3", s2);
    sysfs_create_file(s2, &dev_attr.attr);
    sysfs_create_link(s2, s1, "symlink_demo");

    return 0;
}

static void sysfs_demo_exit(void) {
    printk(KERN_INFO "sysfs demo exit\n");

    sysfs_remove_file(root, &dev_attr.attr);
    kobject_put(s2);
    kobject_put(s1);
    root_device_unregister(dev);
}

module_init(sysfs_demo_init);
module_exit(sysfs_demo_exit);

MODULE_LICENSE("GPL");