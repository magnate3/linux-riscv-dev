#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <asm/uaccess.h>
#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/version.h>
#include <asm/io.h>
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
        int     ret;
        struct  page *page;
        char    *page_addr;
        char    *myaddr;
        int locked = 0;
        int user_pages = 0;
	int nr_pages = count>> PAGE_SHIFT;
	//int nr_pages = 1;
	struct page **pages;
        unsigned long vaddr = (unsigned long)buf;
        unsigned long page_offset = vaddr & ~PAGE_MASK;
	unsigned long phys_addr;
        //FOLL_PIN | FOLL_LONGTERM|FOLL_WRITE | FOLL_FORCE,
	unsigned int flags = FOLL_FORCE | FOLL_WRITE | FOLL_LONGTERM;
        printk(KERN_INFO "%s\n", __FUNCTION__);
        pages = kcalloc(nr_pages, sizeof(*pages), GFP_KERNEL);
	if (!pages) {
		ret = -ENOMEM;
		goto free_page_table;
	}
#if  LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
        down_read(&current->mm->mmap_sem);
#else
	down_read(&current->mm->context.ldt_usr_sem);
#endif
#if 1
       user_pages = pin_user_pages(vaddr & PAGE_MASK, nr_pages, flags, pages, NULL);
#else
       locked = 1;
       user_pages = get_user_pages_locked(vaddr , nr_pages, flags,
			pages, &locked);
#endif
#if  LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
        up_read(&current->mm->mmap_sem);
#else
	up_read(&current->mm->context.ldt_usr_sem);
#endif
#if 1
        if (user_pages != nr_pages) {
		ret = user_pages < 0 ? user_pages : -ENOMEM;
                printk(KERN_INFO "get_user_pages_locked %d pages,need %d pages \n", user_pages,nr_pages);
		goto free_pages;
	}
        printk(KERN_INFO "Got mmaped.\n");
        page = pages[0];
        page_addr = kmap(page);
	phys_addr = page_to_phys(page);

        printk(KERN_INFO "kernel phy addr %lx\n", phys_addr);
        myaddr = (char *)((unsigned long)page_addr+ page_offset) ;
        printk(KERN_INFO "%s\n", myaddr);
        strcpy(myaddr, "Mohan from kernel");
        kunmap(page);
        /* Clean up */
        //if (!PageReserved(page))
        //     SetPageDirty(page);
        set_page_dirty_lock(page); 
        //release_pages(&page, 1, 0);
	//return ret;
#endif    
free_pages:
	while (--user_pages >= 0) {
		put_page(pages[user_pages]);
	}
	kfree(pages);
free_page_table:
        return ret;
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
