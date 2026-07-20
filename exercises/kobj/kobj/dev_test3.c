/*
 * Sample kobject implementation
 *
 * Copyright (C) 2004-2007 Greg Kroah-Hartman <greg@kroah.com>
 * Copyright (C) 2007 Novell Inc.
 *
 * Released under the GPL version 2 only.
 *
 */
#include <linux/kobject.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
static  char mybuf[100]="123";
static ssize_t show_my_device(struct device *dev,
                  struct device_attribute *attr, char *buf)        //cat命令时,将会调用该函数
{
    return sprintf(buf, "%s\n", mybuf);
}

static ssize_t set_my_device(struct device *dev,
                 struct device_attribute *attr,
                 const char *buf, size_t len)        //echo命令时,将会调用该函数
{
    sprintf(mybuf, "%s", buf);
    return len;
}
static DEVICE_ATTR(my_device_test, S_IWUSR|S_IRUSR, show_my_device, set_my_device);
                //定义一个名字为my_device_test的设备属性文件
struct file_operations mytest_ops={
         .owner  = THIS_MODULE,
};

static int major;
static struct class *cls;
static struct device_attribute *dev_attr;
static int mytest_init(void)
{
         struct device *mydev;   
         dev_attr = kmalloc(sizeof(struct device_attribute), GFP_KERNEL);
        dev_attr->show = show_my_device;
        dev_attr->store = set_my_device;
        dev_attr->attr.name = "my_kobj";
        dev_attr->attr.mode = S_IRUGO;
         major=register_chrdev(0,"mytest", &mytest_ops);
         cls=class_create(THIS_MODULE, "mytest_class");
         mydev = device_create(cls, 0, MKDEV(major,0),NULL,"mytest_device");    //创建mytest_device设备    
    
    if(sysfs_create_file(&(mydev->kobj), &(dev_attr->attr))){    //在mytest_device设备目录下创建一个my_device_test属性文件
            return -1;}
            
         return 0;
}

static void mytest_exit(void)
{
         kfree(dev_attr);
         //sys_remove_file(&mydev->kobj, dev_attr_my_device_test);
         device_destroy(cls, MKDEV(major,0));
         class_destroy(cls);
         unregister_chrdev(major, "mytest");
}

module_init(mytest_init);
module_exit(mytest_exit);
MODULE_LICENSE("GPL");
