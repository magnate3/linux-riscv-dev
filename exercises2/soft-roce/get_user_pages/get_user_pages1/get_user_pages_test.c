#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <asm/uaccess.h>
#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/mm.h>

static  struct  class *sample_class;

static int sample_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return (0);
}

static int sample_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return (0);

}

static ssize_t  sample_write(struct file *file, const char __user *buf, size_t count, loff_t *off)
{
        int     res;
        struct  page *page;
        char    *myaddr;
        printk(KERN_INFO "%s\n", __FUNCTION__);
        down_read(&current->mm->mmap_sem);
#if 0
        res = get_user_pages(current, current->mm,
                (unsigned long)buf,
                1,
                1,              
                1,
                &page,
                NULL);
#else
        res = get_user_pages(
                (unsigned long)buf,
                1,
                1,
                &page,
                NULL);
#endif
        if (res) {
                printk(KERN_INFO "Got mmaped.\n");
                myaddr = kmap(page);
                printk(KERN_INFO "%s\n", myaddr);
                strcpy(myaddr, "Mohan");
                kunmap(page);
                /* Clean up */
                //if (!PageReserved(page))
                //     SetPageDirty(page);
                set_page_dirty_lock(page); 
                release_pages(&page, 1, 0);
                //put_page(page);
                
        }
        up_read(&current->mm->mmap_sem);
        return (0);
}

static struct   file_operations sample_ops = {
        .owner  = THIS_MODULE,
        .open   = sample_open,
        .release = sample_release,
        .write  = sample_write
};

static int __init sample_init(void)
{
        int ret;
        ret = register_chrdev(42, "Sample", &sample_ops);
        sample_class = class_create(THIS_MODULE, "Sample");
        device_create(sample_class, NULL, MKDEV(42, 0), NULL, "Sample");
        return (ret);
}

static void __exit sample_exit(void)
{
        device_destroy(sample_class, MKDEV(42, 0));
        class_destroy(sample_class);
        unregister_chrdev(42, "Sample");
}

module_init(sample_init);
module_exit(sample_exit);
MODULE_LICENSE("GPL");
