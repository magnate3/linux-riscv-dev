/*
 * MMU notifier on myMMU
 *
 * (C) 2020.10.06 BuddyZhang1 <buddy.zhang@aliyun.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/sched/mm.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/mmu_notifier.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/version.h>
#include <asm-generic/io.h>

/* MMU notifier */
#include <linux/mmu_notifier.h>
/* Current */
#include <linux/sched.h>
#include <linux/hmm.h>

#define DPT_XA_TAG_WRITE 3UL
#define DEV_NAME                "my_dev"
#define TEST_INTERVAL_NOTIFIER 1
static struct xarray                   pt;
static void *kaddr;
static  struct mmu_interval_notifier   my_interval_notifier;
static struct mmu_notifier my_notifier;
static struct mmu_notifier_range my_range;


static void my_mmu_release(struct mmu_notifier *mn,
                                                struct mm_struct *mm)
{
        printk("myMMU notifier: release\n");
}

static int my_mmu_clear_flush_young(struct mmu_notifier *mn,
                struct mm_struct *mm, unsigned long start, unsigned long end)
{
        printk("myMMU notifier: clear_flush_young\n");
        return 0;
}

static int my_mmu_clear_young(struct mmu_notifier *mn,
                struct mm_struct *mm, unsigned long start, unsigned long end)
{
        printk("myMMU notifier: clear_young\n");
        return 0;
}

static int my_mmu_test_young(struct mmu_notifier *mn,
                        struct mm_struct *mm, unsigned long address)
{   
        printk("myMMU notifier: test_young\n");
        return 0;
}

static void my_mmu_change_pte(struct mmu_notifier *mn,
                struct mm_struct *mm, unsigned long address, pte_t pte)
{
	//dump_stack();
        printk("myMMU notifier: change_pte\n");
}

static int my_mmu_invalidate_range_start(struct mmu_notifier *mn,
                                const struct mmu_notifier_range *range)
{
        printk("myMMU notifier: invalidate_range_start.\n");
        return 0;
}

static void my_mmu_invalidate_range_end(struct mmu_notifier *mn,
                                const struct mmu_notifier_range *range)
{
        printk("myMMU notifier: invalidate_range_end.\n");
}

static void my_mmu_invalidate_range(struct mmu_notifier *mn,
                struct mm_struct *mm, unsigned long start, unsigned long end)
{
	//dump_stack();
        printk("myMMU notifier: invalidate_range.\n");
}

static const struct mmu_notifier_ops my_mmu_notifer_ops = {
        .release     = my_mmu_release,
        .clear_young = my_mmu_clear_young,
        .test_young  = my_mmu_test_young,
        .change_pte  = my_mmu_change_pte,
        .clear_flush_young = my_mmu_clear_flush_young,
        .invalidate_range  = my_mmu_invalidate_range,
        .invalidate_range_start = my_mmu_invalidate_range_start,
        .invalidate_range_end   = my_mmu_invalidate_range_end,
};
// static bool nouveau_svm_range_invalidate
static bool mlx5_ib_invalidate_range(struct mmu_interval_notifier *mni, const struct mmu_notifier_range *range, unsigned long cur_seq)
{
        printk("myMMU notifier: mlx5 invalidate_range.\n");
	mmu_interval_set_seq(mni, cur_seq);
	return true;
}
const struct mmu_interval_notifier_ops mlx5_mn_ops = {
	        .invalidate = mlx5_ib_invalidate_range,
};
static int dmirror_do_read(unsigned long start, unsigned long end)
{
	unsigned long pfn;
	//void *ptr;
	for (pfn = start >> PAGE_SHIFT; pfn < (end >> PAGE_SHIFT); pfn++) {
		void *entry;
		struct page *page;
		void *tmp;

		entry = xa_load(&pt, pfn);
		page = xa_untag_pointer(entry);
		if (!page)
			return -ENOENT;

		tmp = kmap(page);
		//memcpy(ptr, tmp, PAGE_SIZE);
		kunmap(page);

		//ptr += PAGE_SIZE;
	}

	return 0;
}

static int dmirror_do_fault(struct hmm_range *range)
{
	unsigned long *pfns = range->hmm_pfns;
	unsigned long pfn;
        char *addr = NULL;
	char buf[64] = {0};
	unsigned long page_offset;
	pr_info("do in  %s  func\n",__func__);
	for (pfn = (range->start >> PAGE_SHIFT);
	     pfn < (range->end >> PAGE_SHIFT);
	     pfn++, pfns++) {
		struct page *page;
		void *entry;

		/*
		 * Since we asked for hmm_range_fault() to populate pages,
		 * it shouldn't return an error entry on success.
		 */
		WARN_ON(*pfns & HMM_PFN_ERROR);
		WARN_ON(!(*pfns & HMM_PFN_VALID));

		page = hmm_pfn_to_page(*pfns);
		WARN_ON(!page);
		if(pfn == range->start >> PAGE_SHIFT)
		{
	           page_offset = range->start & ~PAGE_MASK;
                   addr =  page_address(page); 
		   addr += page_offset;
		   memcpy(buf,addr,32);
		   pr_info("buf is %s \n", buf);
		}
		entry = page;
		if (*pfns & HMM_PFN_WRITE)
			entry = xa_tag_pointer(entry, DPT_XA_TAG_WRITE);
		else if (WARN_ON(range->default_flags & HMM_PFN_WRITE))
			return -EFAULT;
		entry = xa_store(&pt, pfn, entry, GFP_ATOMIC);
		if (xa_is_err(entry))
			return xa_err(entry);
	}

	return 0;
}

#if 1
static int dmirror_range_fault(struct mm_struct *mm,
				struct hmm_range *range)
{
	//struct mm_struct *mm = dmirror->notifier.mm;
	unsigned long timeout =
		jiffies + msecs_to_jiffies(HMM_RANGE_DEFAULT_TIMEOUT);
	int ret = 0;

	while (true) {
		if (time_after(jiffies, timeout)) {
			ret = -EBUSY;
			goto out;
		}

		range->notifier_seq = mmu_interval_read_begin(range->notifier);
		mmap_read_lock(mm);
		ret = hmm_range_fault(range);
		mmap_read_unlock(mm);
		if (ret) {
			if (ret == -EBUSY)
				continue;
			goto out;
		}

		//mutex_lock(&dmirror->mutex);
		if (mmu_interval_read_retry(range->notifier,
					    range->notifier_seq)) {
			//mutex_unlock(&dmirror->mutex);
			continue;
		}
		break;
	}

	ret = dmirror_do_fault(range);

	//mutex_unlock(&dmirror->mutex);
out:
	return ret;
}
#else
static int dmirror_range_fault(struct mm_struct *mm,
				struct hmm_range *range)
{
	int ret = 0;
	mmap_read_lock(mm);
	ret = hmm_range_fault(range);
        mmap_read_unlock(mm);
	return ret;
}
#endif
static int dmirror_fault(struct mm_struct *mm, unsigned long start,
			 unsigned long end, bool write)
{
	unsigned long addr;
	unsigned long pfns[64];
	struct hmm_range range = {
		.notifier = &my_interval_notifier,
		.hmm_pfns = pfns,
		.pfn_flags_mask = 0,
		.default_flags =
			HMM_PFN_REQ_FAULT | (write ? HMM_PFN_REQ_WRITE : 0),
		//.dev_private_owner = dmirror->mdevice,
	};
	int ret = 0;

	/* Since the mm is for the mirrored process, get a reference first. */
	if (!mmget_not_zero(mm))
		return 0;

	for (addr = start; addr < end; addr = range.end) {
		range.start = addr;
		range.end = min(addr + (ARRAY_SIZE(pfns) << PAGE_SHIFT), end);

		ret = dmirror_range_fault(mm, &range);
		if (ret)
			break;
	}

	mmput(mm);
	return ret;
}
static int my_mmap(struct file *filp, struct vm_area_struct *vma)
{

	int ret = 0;
        ret = remap_pfn_range(vma, vma->vm_start, (virt_to_phys(kaddr) >> PAGE_SHIFT) + vma->vm_pgoff, vma->vm_end - vma->vm_start, vma->vm_page_prot);
	pr_info("remap ret %d \n",ret);
	return 0;
}

static int my_open(struct inode *inode, struct file *file)
{
        struct mm_struct *mm = get_task_mm(current);

        file->private_data = mm;
#if 0 
        /* mmu notifier initialize */
        my_notifier.ops = &my_mmu_notifer_ops;
        /* mmu notifier register */
        mmu_notifier_register(&my_notifier, mm);
#endif
#ifdef TEST_INTERVAL_NOTIFIER
	my_interval_notifier.ops = &mlx5_mn_ops;
        //mmu_interval_notifier_insert(&my_interval_notifier, current->mm, 0, ULONG_MAX & PAGE_MASK, &mlx5_mn_ops);
#endif
        return 0;
}

static int my_release(struct inode *inode, struct file *file)
{
        //mmu_notifier_unregister(&my_notifier, current->mm);
#ifdef TEST_INTERVAL_NOTIFIER
	mmu_interval_notifier_remove(&my_interval_notifier);
#endif
        return 0;
}
static ssize_t my_read(struct file *filp, /* see include/linux/fs.h   */
		                           char __user *buffer, /* buffer to fill with data */
					                              size_t length, /* length of the buffer     */
								                                 loff_t *offset)
{
	int ret = 0;
        struct mm_struct *mm = filp->private_data;
	unsigned long start = (unsigned long)buffer;
	unsigned long end = (unsigned long)buffer + length;
	//ret = dmirror_fault(mm,  vma->vm_start, vma->vm_end, true);
#ifdef TEST_INTERVAL_NOTIFIER
        mmu_interval_notifier_insert(&my_interval_notifier, current->mm, start, length, &mlx5_mn_ops);
#endif
	ret = dmirror_fault(mm,  start, end, true);
	pr_info("fault ret %d \n",ret);
	return 0;
}
/* file operations */
static struct file_operations my_fops = {
        .owner          = THIS_MODULE,
        .open           = my_open,
        .mmap           = my_mmap,
        .release        = my_release,
	.read = my_read,
};
/* Misc device driver */
static struct miscdevice my_drv = {
        .minor  = MISC_DYNAMIC_MINOR,
        .name   = DEV_NAME,
        .fops   = &my_fops,
};
/* Module initialize entry */
static int __init my_init(void)
{
	kaddr = kzalloc(PAGE_SIZE * 3, GFP_KERNEL);
        /* Register Misc device */
        misc_register(&my_drv);

        printk("Hello modules on myMMU\n");
	return 0;
}
static void __exit my_exit(void)
{
	kfree(kaddr);
        /* Register Misc device */
        misc_deregister(&my_drv);
         xa_destroy(&pt);
        printk("Hello modules on myMMU\n");
}
module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
