#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/types.h>
#include <linux/cdev.h>
#include <linux/string.h>
#include <linux/sysfs.h>

#define DEMO_NAME "demo_dev"

struct hwmon_attr {
        struct device_attribute dev_attr;
        char name[12];
        };

struct demo_dev {
	struct cdev chr_dev;
	struct device *demo_device;
	dev_t dev;
        struct hwmon_attr igb_attr;
};

static dev_t devno;
static unsigned int major;
static struct demo_dev *cur;
static struct class *demo_class;

static ssize_t albert_show(struct device *dev, 
		struct device_attribute *attr, char *buf)
{
        struct hwmon_attr *igb_attr = container_of(attr, struct hwmon_attr,
                                                     dev_attr);
	printk("albert:%s, name %s \n",__func__, igb_attr->name);
	return 0;
}

static ssize_t albert_store(struct device *dev, 
		struct device_attribute *attr, const char *buf, size_t size)
{
	printk("albert:%s\n",__func__);
	return 0;	
}

static int chr_open(struct inode *nd, struct file *filp)
{
	int major = MAJOR(nd->i_rdev);
	int minor = MINOR(nd->i_rdev);

	printk("chr_open, major = %d, minor = %d\n",major, minor);
	return 0;	
}

static ssize_t chr_read(struct file *f, char __user *u, size_t sz, loff_t *off)
{
	printk("in the chr_read() function!\n");
	return 0;
}

struct file_operations chr_ops = {
	.owner = THIS_MODULE,
	.open  = chr_open,
	.read  = chr_read,
};

static int demo_init(void)
{
        int ret;
	cur = kzalloc(sizeof(*cur), GFP_KERNEL);
	if (!cur) {
		printk("albert:Unable to alloc albert dev\n");
		return -ENOMEM;
	}

	demo_class = class_create(THIS_MODULE, "demo_class");
	if (IS_ERR(demo_class)) {
		pr_err("couldn't register class demo!\n");
		ret = PTR_ERR(demo_class);
		goto out_destroy_class;
	}

//	demo_class->dev_groups = demo_groups;

	ret = alloc_chrdev_region(&devno, 0, 1, DEMO_NAME);
	if (ret < 0)
		return ret;

	printk("albert:major = %d,minor = %d\n", MAJOR(devno), MINOR(devno));
	major = MAJOR(devno);
	cur->dev = MKDEV(major, 0);

	cdev_init(&cur->chr_dev, &chr_ops);
	ret = cdev_add(&cur->chr_dev, cur->dev, 1);
	if (ret < 0)
		return ret;

	cur->demo_device = device_create(demo_class, NULL, cur->dev, NULL, "demo_device");
	if (IS_ERR(cur->demo_device)) {
		pr_err("couldn't create demo device!\n");
		ret = PTR_ERR(cur->demo_device);
		goto out_destroy_device;
	}
        cur->igb_attr.dev_attr.store = albert_store;
        cur->igb_attr.dev_attr.show = albert_show;
        cur->igb_attr.dev_attr.attr.mode = 0444;
        cur->igb_attr.dev_attr.attr.name = "igb_test"; 
        strncpy(cur->igb_attr.name,"igb_demo", sizeof("igb_demo")); 
	ret = device_create_file(cur->demo_device, & cur->igb_attr.dev_attr);
	return 0;

out_destroy_device:
	device_destroy(demo_class,cur->dev);
out_destroy_class:
	class_destroy(demo_class);
	return ret;
}

static void demo_exit(void)
{
	printk("removing chr_dev module!\n");
	cdev_del(&cur->chr_dev);
	unregister_chrdev_region(devno, 1);
	device_destroy(demo_class,cur->dev);
	class_destroy(demo_class);
}

module_init(demo_init);
module_exit(demo_exit);

MODULE_LICENSE("GPL");
 
