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
// In Linux 5.0, dma_alloc_coherent always zeroes memory and dma_zalloc_coherent
// was removed.
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 0, 0)
#define dma_alloc_coherent dma_zalloc_coherent
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 11, 0)
static int pin_user_pages_fast_longterm(unsigned long start, int nr_pages, unsigned int gup_flags, struct page **pages)
{
        pr_info("kernel >= 5.11.0 call pin_user_pages_fast_longterm \n");
	// vma array allocation removed in 52650c8b466bac399aec213c61d74bfe6f7af1a4.
	return pin_user_pages_fast(start, nr_pages, gup_flags | FOLL_LONGTERM, pages);
}
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
static int pin_user_pages_fast_longterm(unsigned long start, int nr_pages, unsigned int gup_flags, struct page **pages)
{
	// Can't use pin_user_pages_fast(FOLL_LONGTERM) because it calls __gup_longterm_locked with vmas = NULL
	// which allocates a contiguous vmas array and that fails often.

	int ret;

	struct vm_area_struct **vmas = kvmalloc_array(nr_pages, sizeof(struct vm_area_struct *), GFP_KERNEL);
	if (vmas == NULL)
		return -ENOMEM;

	ret = pin_user_pages(start, nr_pages, gup_flags | FOLL_LONGTERM, pages, vmas);

	kvfree(vmas);
	return ret;
}
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(5, 2, 0)
static int pin_user_pages_fast_longterm(unsigned long start, int nr_pages, unsigned int gup_flags, struct page **pages)
{
	// Can't use get_user_pages_fast(FOLL_LONGTERM) because it calls __gup_longterm_locked with vmas = NULL
	// which allocates a contiguous vmas array and that fails often.

	int ret;

	struct vm_area_struct **vmas = kvmalloc_array(nr_pages, sizeof(struct vm_area_struct *), GFP_KERNEL);
	if (vmas == NULL)
		return -ENOMEM;

	ret = get_user_pages(start, nr_pages, gup_flags | FOLL_LONGTERM, pages, vmas);

	kvfree(vmas);
	return ret;
}
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(4, 14, 4)
static int pin_user_pages_fast_longterm(unsigned long start, int nr_pages, unsigned int gup_flags, struct page **pages)
{
	int ret;

        pr_info("kernel > 4.14.4 call pin_user_pages_fast_longterm \n");
	// If we don't pass in vmas, get_user_pages_longterm will allocate it in contiguous memory and that fails often.
	struct vm_area_struct **vmas = kvmalloc_array(nr_pages, sizeof(struct vm_area_struct *), GFP_KERNEL);
	if (vmas == NULL)
		return -ENOMEM;

	down_read(&current->mm->mmap_sem);
	ret = get_user_pages_longterm(start, nr_pages, gup_flags, pages, vmas);
	up_read(&current->mm->mmap_sem);

	kvfree(vmas);
	return ret;
}
#else
static int pin_user_pages_fast_longterm(unsigned long start, int nr_pages, unsigned int gup_flags, struct page **pages)
{
        pr_info("kernel < 4.14.4 call pin_user_pages_fast_longterm \n");
	// Kernels this old don't know about long-term pinning, so they don't allocate the vmas array.
	return get_user_pages_fast(start, nr_pages, gup_flags, pages);
}
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
// unpin_user_pages_dirty_lock is provided by the kernel.
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(5, 4, 0)
static void unpin_user_pages_dirty_lock(struct page **pages, unsigned long npages, bool make_dirty)
{
	put_user_pages_dirty_lock(pages, npages, make_dirty);
}
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(5, 2, 0)
static void unpin_user_pages_dirty_lock(struct page **pages, unsigned long npages, bool make_dirty)
{
	if (make_dirty)
		put_user_pages_dirty_lock(pages, npages);
	else
		put_user_pages(pages, npages);
}
#else
static void unpin_user_pages_dirty_lock(struct page **pages, unsigned long npages, bool make_dirty)
{
	struct page **end = pages + npages;
	for (; pages != end; pages++) {
		if (make_dirty)
			set_page_dirty_lock(*pages);
		put_page(*pages);
	}
}
#endif
static ssize_t  sample_write(struct file *file, const char __user *buf, size_t count, loff_t *off)
{
        int     res;
        struct page *pages[1];
        struct  page *page;
        char    *myaddr;
        unsigned long arg = (unsigned long)buf;
        struct vm_area_struct *vma = NULL;
        printk(KERN_INFO "%s\n", __FUNCTION__);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 14, 4)
	mmap_read_lock(current->mm);
#else
        down_read(&current->mm->mmap_sem);
#endif
        vma = find_vma(current->mm, arg);
        if (!vma)
           return -EIO;

#if 1
    res = pin_user_pages_fast_longterm(arg, 1, FOLL_WRITE, pages);
    page = pages[0];
    if (res < 1) {
        printk(KERN_INFO "GUP error: %d\n", res);
        free_page((unsigned long) page);
        return -EFAULT;
    }

#else
        res = get_user_pages(
                arg ,
                1,
                1,
                &page,
                NULL);
#endif
        if (res) {
                printk(KERN_INFO "Got mmaped.\n");
                myaddr = kmap(page);
                printk(KERN_INFO "%s\n", myaddr);
                strcpy(myaddr, "from kernel is  Mohan");
                //page_cache_release(page);
                //put_page(page);
                unpin_user_pages_dirty_lock(pages, 1, false);
        }
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 14, 4)
	mmap_read_lock(current->mm);
#else
        up_read(&current->mm->mmap_sem);
#endif
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
