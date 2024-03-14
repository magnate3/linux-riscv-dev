/*************************************************************************

 ************************************************************************/
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include <asm/io.h>
#include <linux/fs.h>
#include <linux/cdev.h>

#include <linux/device.h>

#define DEV_MAJOR   500
#define DEV_MINOR   0


dev_t devno;
struct class* ldm_cls;
struct device* ldm_device;
static int led_init(void)
{
    printk("led configration sucess \n");
    return 0;
}


static struct cdev ldm_dev;

static int ldm_open(struct inode *inod, struct file * filp)
{
    printk("open sucess \n");
    led_init();
    return 0;
}


static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open  = ldm_open,
};


static int __init ldm_init(void)
{
    int ret;
    int i = 0;
    int major;
//    devno = MKDEV(DEV_MAJOR, DEV_MINOR);
//    ret = register_chrdev_region(devno, 1, "demo");
    ret = alloc_chrdev_region(&devno, 0, 1, "ldm_led");     //自动分配设备号 这个名字无所谓
    if (ret < 0)
    {
        printk("alloc_chrdev_region failed \n");
        goto err_register_chrdev;
    }

    cdev_init(&ldm_dev, &fops);

    cdev_add(&ldm_dev, devno, 1);
    
    //获取主设备号
    major = MAJOR(devno);
    pr_info("************** major : %d \n", major);
    ldm_cls = class_create(THIS_MODULE, "ldm_cls");
    for (i = 0; i < 3; i++)
        ldm_device = device_create(ldm_cls, NULL,MKDEV(major, i), NULL, "ldm_led_%d", i);  //名字默认载/dev目录下

printk("ldm_init sucess \n");
    return 0;
err_register_chrdev:
    return ret;
}

static void __exit ldm_exit(void)
{
    int i = 0;
    int major;
    major = MAJOR(devno);
    for (i = 0; i < 3; i++)
        device_destroy(ldm_cls, MKDEV(major, i));
    class_destroy(ldm_cls);
    cdev_del(&ldm_dev);
    unregister_chrdev_region(devno, 1);
    printk("ldm_exit sucess \n");    
}


module_init(ldm_init);
module_exit(ldm_exit);
MODULE_LICENSE("GPL");
