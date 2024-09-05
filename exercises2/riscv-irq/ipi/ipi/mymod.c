// SPDX-License-Identifier: GPL-2.0

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/ioctl.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/printk.h>
#include <linux/smp.h>
#include <linux/spinlock.h>
#include <linux/types.h>

#include "mymod.h"

MODULE_LICENSE("GPL");

#define DEV_COUNT 1

static dev_t dev;
static struct cdev cdev;
static struct class *class;
static struct device *device;
static DEFINE_MUTEX(mymutex);
static DEFINE_SPINLOCK(myspinlock);

static int mymod_cdev_open(struct inode *inode, struct file *file)
{
	pr_info("%s\n", __func__);
	pr_info("%s end\n", __func__);
	return 0;
}

static int mymod_cdev_release(struct inode *inode, struct file *file)
{
	pr_info("%s start\n", __func__);
	pr_info("%s end\n", __func__);
	return 0;
}

static void ipi_good(void *data)
{
	pr_info("%s start\n", __func__);
	pr_info("%s end\n", __func__);
}

static void ipi_bad_blocking(void *data)
{
	pr_info("%s start\n", __func__);
	mutex_lock(&mymutex);
	pr_info("%s end\n", __func__);
}

static void ipi_bad_nonblocking(void *data)
{
	pr_info("%s start\n", __func__);
	spin_lock(&myspinlock);
	pr_info("%s end\n", __func__);
}

static long mymod_cdev_ioctl(struct file *file, unsigned int cmd,
	unsigned long arg)
{
	const int cpu = 3;
	void *data = NULL;
	const int wait = 1;

	pr_info("%s start\n", __func__);

	switch (cmd) {
		case MYMOD_MUTEX_LOCK:
			pr_info("%s acquiring mutex\n", __func__);
			mutex_lock(&mymutex);
			break;
		case MYMOD_MUTEX_UNLOCK:
			pr_info("%s releasing mutex\n", __func__);
			mutex_unlock(&mymutex);
			break;
		case MYMOD_SPIN_LOCK:
			pr_info("%s acquiring spinlock\n", __func__);
			spin_lock(&myspinlock);
			break;
		case MYMOD_SPIN_UNLOCK:
			pr_info("%s releasing spinlock\n", __func__);
			spin_unlock(&myspinlock);
			break;
		case MYMOD_IPI_GOOD:
			pr_info("%s smp call good\n", __func__);

			if (smp_call_function_single(cpu, ipi_good, data, wait)) {
				pr_err("smp call fail\n");
			}

			break;
		case MYMOD_IPI_BAD_BLOCKING:
			pr_info("%s smp call blocking\n", __func__);

			if (smp_call_function_single(cpu, ipi_bad_blocking, data, wait)) {
				pr_err("smp call fail\n");
			}

			break;
		case MYMOD_IPI_BAD_NONBLOCKING:
			pr_info("%s smp call non-blocking\n", __func__);

			if (smp_call_function_single(cpu, ipi_bad_nonblocking, data, wait)) {
				pr_err("smp call fail\n");
			}

			break;
		default:
			pr_err("invalid ioctl cmd\n");
			return -EINVAL;
			break;
	}

	pr_info("%s end\n", __func__);
	return 0;
}

static struct file_operations cdev_fops = {
	.owner = THIS_MODULE,
	.open = mymod_cdev_open,
	.release = mymod_cdev_release,
	.unlocked_ioctl = mymod_cdev_ioctl,
};

static int __init mymod_init(void)
{
	int rc;

	pr_info("%s start\n", __func__);

	rc = alloc_chrdev_region(&dev, 0, DEV_COUNT, "mymod");
	if (rc) {
		pr_err("alloc_chrdev_region failed: %d\n", rc);
		return rc;
	}
	pr_info("%s alloc'd chrdev region, major:%u, minor:%u\n",
		__func__, MAJOR(dev), MINOR(dev));

	cdev_init(&cdev, &cdev_fops);
	cdev.owner = THIS_MODULE;
	rc = cdev_add(&cdev, dev, DEV_COUNT);
	if (rc) {
		pr_err("cdev_add fail: %d\n", rc);
		unregister_chrdev_region(dev, DEV_COUNT);
		return rc;
	}

	class = class_create(THIS_MODULE,"mymod");
	if (!class) {
		pr_err("class_create fail\n");
		unregister_chrdev_region(dev, DEV_COUNT);
		return -EFAULT;
	}

	device = device_create(class, NULL, dev, NULL, "mymod%d", MINOR(dev));
	if (!device) {
		pr_err("class_create fail\n");
		class_destroy(class);
		unregister_chrdev_region(dev, DEV_COUNT);
		return -EFAULT;
	}

	pr_info("%s end\n", __func__);
	return 0;
}

static void __exit mymod_exit(void)
{
	pr_info("%s start\n", __func__);

	cdev_del(&cdev);

	unregister_chrdev_region(dev, DEV_COUNT);

	device_destroy(class, dev);
	class_destroy(class);

	pr_info("%s end\n", __func__);
}

module_init(mymod_init);
module_exit(mymod_exit);
