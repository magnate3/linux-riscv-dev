#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <asm/uaccess.h>
#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/mm.h>

#define DEV_NAME "hugepage-driver"
#define DEV_MAJOR 42

static struct class *dev_class = NULL;

static int dev_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return 0;
}

static int dev_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return 0;
}

static ssize_t dev_write(struct file *file, const char __user *buf, size_t count, loff_t *off)
{
        char str[100];
        unsigned long long paddr;
        int i, length;
        char *vaddr;

        if (copy_from_user(str, buf, sizeof(str)) != 0) {
                printk(KERN_INFO "Copy failes");
                return 0;
        }

        sscanf(str, "%llu %d", &paddr, &length);

        printk(KERN_INFO "The physical address is %llu\n", paddr);
        vaddr = phys_to_virt(paddr);
        printk(KERN_INFO "The virtual address is %p\n", vaddr);        
        printk(KERN_INFO "The memory length is %d\n", length);

        // Increase the value of each byte by 1
        for (i = 0; i < length; i++) {
                *vaddr = *vaddr + 1;
                vaddr++;
        }

        return 0;
}

static struct   file_operations dev_ops = {
        .owner  = THIS_MODULE,
        .open   = dev_open,
        .release = dev_release,
        .write  = dev_write
};

int init_module(void)
{
        int ret;

        printk(KERN_INFO "Install %s\n", DEV_NAME);

        ret = register_chrdev(DEV_MAJOR, DEV_NAME, &dev_ops);
        dev_class = class_create(THIS_MODULE, DEV_NAME);
        device_create(dev_class, NULL, MKDEV(DEV_MAJOR, 0), NULL, DEV_NAME);

        return (ret);
}


void cleanup_module(void)
{       
        device_destroy(dev_class, MKDEV(DEV_MAJOR, 0));
        class_destroy(dev_class);
        unregister_chrdev(DEV_MAJOR, DEV_NAME);

        printk(KERN_INFO "Remove %s\n", DEV_NAME);
}

MODULE_LICENSE("GPL");


