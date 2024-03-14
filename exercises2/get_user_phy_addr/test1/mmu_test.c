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
#define TEST_INTERVAL_NOTIFIER 0
#define TEST_ALLOC_PAGE 0
static struct xarray                   pt;
static void *kaddr;
struct page *start_page = NULL;
static  struct mmu_interval_notifier   my_interval_notifier;
static struct mmu_notifier my_notifier;
static struct mmu_notifier_range my_range;

static unsigned long vaddr2paddr(struct mm_struct *mm , unsigned long addr);
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

        struct mm_struct *mm = filp->private_data;
	int ret = 0;
#if TEST_ALLOC_PAGE 
        unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;
	unsigned long pfn_start = page_to_pfn(start_page) + vma->vm_pgoff;
	unsigned long virt_start = (unsigned long)page_address(start_page) + (vma->vm_pgoff << PAGE_SHIFT);
	    /* 映射大小不超过实际物理页 */
	unsigned long size = vma->vm_end - vma->vm_start;
	ret = remap_pfn_range(vma, vma->vm_start, pfn_start, size, vma->vm_page_prot);
        printk("kernel page  phy addr %lx\n",pfn_start);
#else
        ret = remap_pfn_range(vma, vma->vm_start, (virt_to_phys(kaddr) >> PAGE_SHIFT) + vma->vm_pgoff, vma->vm_end - vma->vm_start, vma->vm_page_prot);
	pr_info("remap ret %d \n",ret);
        printk("kaddr phy addr %lx\n",vaddr2paddr(mm,(unsigned long )kaddr));
#endif
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
#if TEST_INTERVAL_NOTIFIER
	my_interval_notifier.ops = &mlx5_mn_ops;
        //mmu_interval_notifier_insert(&my_interval_notifier, current->mm, 0, ULONG_MAX & PAGE_MASK, &mlx5_mn_ops);
#endif
        return 0;
}

static int my_release(struct inode *inode, struct file *file)
{
        //mmu_notifier_unregister(&my_notifier, current->mm);
#if TEST_INTERVAL_NOTIFIER
	mmu_interval_notifier_remove(&my_interval_notifier);
#endif
        return 0;
}
static ssize_t my_read(struct file *filp, /* see include/linux/fs.h   */
		                           char __user *buffer, /* buffer to fill with data */
					                              size_t length, /* length of the buffer     */
								                                 loff_t *offset)
{
#if TEST_INTERVAL_NOTIFIER
	int ret = 0;
        struct mm_struct *mm = filp->private_data;
	unsigned long start = (unsigned long)buffer;
	unsigned long end = (unsigned long)buffer + length;
	//ret = dmirror_fault(mm,  vma->vm_start, vma->vm_end, true);
        mmu_interval_notifier_insert(&my_interval_notifier, current->mm, start, length, &mlx5_mn_ops);
	ret = dmirror_fault(mm,  start, end, true);
	pr_info("fault ret %d \n",ret);
#else
        struct mm_struct *mm = filp->private_data;
        printk("buffer phy addr %lx\n",vaddr2paddr(mm,(unsigned long )buffer));
#endif
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
static unsigned long vaddr2paddr(struct mm_struct *mm , unsigned long addr)
{
    pgd_t *pgd;
    p4d_t *p4d;
    pte_t *ptep, pte;
    pud_t *pud;
    pmd_t *pmd;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0 ;
    struct page *page = NULL;
    //struct mm_struct *mm = current->mm;

    pgd = pgd_offset(mm, addr);
    if (pgd_none(*pgd) || pgd_bad(*pgd))
        goto out;
    //printk(KERN_NOTICE "Valid pgd");

    p4d = p4d_offset(pgd, addr);
    if (!p4d_present(*p4d))
        goto out;
    pud = pud_offset(p4d, addr);
    if (pud_none(*pud) || pud_bad(*pud))
        goto out;
    //printk(KERN_NOTICE "Valid pud");

    pmd = pmd_offset(pud, addr);
    if (pmd_none(*pmd) || pmd_bad(*pmd))
        goto out;
    //printk(KERN_NOTICE "Valid pmd");

    //ptep = pte_offset_kernel(pmd, addr);
    ptep = pte_offset_map(pmd, addr);
    if (!ptep)
        goto out;
    pte = *ptep;

    page = pte_page(pte);
    if (page)
    {
        //page_addr = pte_val(pte) & PAGE_MASK;
	page_addr = pte_pfn(pte) << PAGE_SHIFT;
        page_offset = addr & ~PAGE_MASK;
        //paddr = page_addr + page_offset;
        paddr = page_addr | page_offset;
        printk(KERN_INFO "page frame struct is @ %p, and user paddr %lx", page, paddr);
    }
    pte_unmap(ptep);
#if 0
    ptep = pte_offset_kernel(pmd, addr);
    if (!ptep)
        goto out;
    pte = *ptep;

    page = pte_page(pte);
    if (page)
    {
        //page_addr = pte_val(pte) & PAGE_MASK;
	page_addr = pte_pfn(pte) << PAGE_SHIFT;
        page_offset = addr & ~PAGE_MASK;
        //paddr = page_addr + page_offset;
        paddr = page_addr | page_offset;
        printk(KERN_INFO "page frame struct is @ %p, and kernel paddr %lx", page, paddr);
    }
#endif
 out:
    return paddr;

}
/* Module initialize entry */
static int __init my_init(void)
{
#if TEST_ALLOC_PAGE
	start_page = alloc_pages(GFP_HIGHUSER_MOVABLE | __GFP_ZERO, 2);
	//start_page = alloc_pages(GFP_USER, 2);
	//start_page = alloc_pages(GFP_KERNEL, 2);
#else
	kaddr = kzalloc(PAGE_SIZE * 3, GFP_KERNEL);
#endif
        /* Register Misc device */
        misc_register(&my_drv);

        printk("**** Hello modules on myMMU\n");
	return 0;
}
static void __exit my_exit(void)
{
#if TEST_ALLOC_PAGE
	__free_pages(start_page,2);
#else
	kfree(kaddr);
#endif
        /* Register Misc device */
        misc_deregister(&my_drv);
         xa_destroy(&pt);
        printk("Hello modules on myMMU\n");
}
module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
