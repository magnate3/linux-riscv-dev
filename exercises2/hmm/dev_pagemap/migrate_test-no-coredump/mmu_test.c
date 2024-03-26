/*
 * MMU notifier on myMMU
 *
 *  
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#if 0
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
#include <linux/migrate.h>
#include <asm-generic/io.h>

#include <linux/pagemap.h>
#include <linux/hmm.h>
#include <linux/vmalloc.h>
#include <linux/swap.h>
#include <linux/swapops.h>

/* MMU notifier */
#include <linux/mmu_notifier.h>
/* Current */
#include <linux/sched.h>
/* check whether a pte points to a swap entry */
static inline int is_swap_pte(pte_t pte)
{
	        return !pte_none(pte) && !pte_present(pte);
}
#else
#include <linux/miscdevice.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/mutex.h>
#include <linux/rwsem.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/highmem.h>
#include <linux/delay.h>
#include <linux/pagemap.h>
#include <linux/hmm.h>
#include <linux/vmalloc.h>
#include <linux/swap.h>
#include <linux/swapops.h>
#include <linux/sched/mm.h>
#include <linux/version.h>
#endif
/* DD Platform Name */
#define DEV_NAME                "my_dev"
static void *kaddr;
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
#if 0
static ssize_t  my_write(struct file *file, const char __user *buf, size_t count, loff_t *off)
{

     unsigned long vaddr = (unsigned long)buf;
     struct vm_area_struct *vma;
     vma = find_vma(current->mm, vaddr);
#if  LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
     down_read(&current->mm->mmap_sem);
#else
     down_read(&current->mm->context.ldt_usr_sem);
#endif
     if(vma)
     {
          zap_vma_ptes(vma, vma->vm_start, vma->vm_end - vma->vm_start);
     }
#if  LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
     up_read(&current->mm->mmap_sem);
#else
     up_read(&current->mm->context.ldt_usr_sem);
#endif
	return 0;
}
#else
static void cmp_page(struct mm_struct *mm , const struct page * src, const struct page *dst)
{
		char *addr1 = NULL, *addr2 = NULL;
		char buf[64] = {0};
		addr1 =  page_address(src); 
		addr2 =  page_address(dst); 
		if(0 == memcmp(addr1,addr2,PAGE_SIZE))
		{
			 pr_info("src and dts page are equal \n");
                         memcpy(buf,addr2,32);       
			 pr_info("buf is %s \n", buf);
		}
}
static void dmirror_migrate_alloc_and_copy(struct migrate_vma *args, unsigned long start,struct vm_area_struct *vma)
{
	const unsigned long *src = args->src;
	unsigned long *dst = args->dst;
	unsigned long addr;

	for (addr = args->start; addr < args->end; addr += PAGE_SIZE,
						   src++, dst++) {
		struct page *spage;
		struct page *dpage;
		//struct page *rpage;

		if (!(*src & MIGRATE_PFN_MIGRATE))
			continue;

		/*
		 * Note that spage might be NULL which is OK since it is an
		 * unallocated pte_none() or read-only zero page.
		 */
		spage = migrate_pfn_to_page(*src);

#if 0
		dpage = alloc_page(GFP_HIGHUSER);
#else
		dpage = alloc_page_vma(GFP_HIGHUSER, vma, start);
		//dpage = alloc_page_vma(GFP_HIGHUSER, vmf->vma, vmf->address);
#endif
		lock_page(dpage);
		if (spage)
		{
		        pr_info("%s call copy highpage  ,dpage @ %p, spage @ %p \n", __func__,dpage, spage);
			copy_highpage(dpage, spage);
			if(addr == args->start)
			{
			    cmp_page(current->mm, spage, dpage);
			}
		}

#if 1
		*dst = migrate_pfn(page_to_pfn(dpage)) |
			    MIGRATE_PFN_LOCKED;
		if ((*src & MIGRATE_PFN_WRITE) ||
		    (!spage && args->vma->vm_flags & VM_WRITE))
			*dst |= MIGRATE_PFN_WRITE;
#endif
		// need to free page
	}
}
static inline struct page *test_migration_entry_to_page(swp_entry_t entry)
{
     struct page *p = pfn_to_page(swp_offset(entry));
		        /*
			 *          * Any use of migration entries may only occur while the
			 *                   * corresponding page is locked
			 *                            */
     //BUG_ON(!PageLocked(compound_head(p)));
     return p;
}
static ssize_t my_read(struct file *file, char __user *u, size_t count,
		                             loff_t *ppos)
{

    pgd_t *pgd;
    p4d_t *p4d;
    pte_t *ptep, pte;
    pud_t *pud;
    pmd_t *pmd;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0 ;
    unsigned long pfn = 0 ;
    //unsigned long mpfn = 0;
    unsigned long addr = (unsigned long)u;
    struct page *page = NULL;
    struct mm_struct *mm = current->mm;

    //struct vm_area_struct *vma = find_vma(current->mm, addr);
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
    {
        goto out1;
    }
    //printk(KERN_NOTICE "Valid pte");
    pte = *ptep;
    if (pte_present(pte)) {
                printk(KERN_INFO "************* pte present \n");
    }
    if(is_swap_pte(pte)){
	swp_entry_t entry;
	entry = pte_to_swp_entry(pte);
        if (!pte_present(pte)) {
                printk(KERN_INFO "************* swap pte, but not present \n");
        }
        if (is_migration_entry(entry))
	{
	    page = test_migration_entry_to_page(entry);
	    printk(KERN_INFO "swap pte,migration  page is present \n");
	}
	if (is_device_private_entry(entry))
	{
            if (!pte_present(pte)) {
                printk(KERN_INFO "************* swap pte, but not present, and it's device private entry");
		goto out1;
	    }
#if 1
	    page = device_private_entry_to_page(entry);
	    pfn = device_private_entry_to_pfn(entry);
	    page_addr = pfn << PAGE_SHIFT;
	    page_offset = addr & ~PAGE_MASK;
	    //paddr = page_addr + page_offset;
	    paddr = page_addr | page_offset;
	    printk(KERN_INFO "swap pte,device page frame struct is @ %p, and user paddr %lu, virt addr %lu", page, paddr, addr);
#else
	    printk(KERN_INFO "swap pte,device page frame is present \n");
#endif
	}
    }
 else
    {
        page = pte_page(pte);
        if (page)
        {
            //page_addr = pte_val(pte) & PAGE_MASK;
            page_addr = pte_pfn(pte) << PAGE_SHIFT;
            page_offset = addr & ~PAGE_MASK;
            //paddr = page_addr + page_offset;
            paddr = page_addr | page_offset;
            printk(KERN_INFO " not swap pte,page frame struct is @ %p, and user paddr %lu, virt addr %lu", page, paddr, addr);
        }
    }

 out1:
        pte_unmap(ptep);
 out:
    return 0;
}
void putback_lru_page(struct page *page)
{
    lru_cache_add(page);
    put_page(page);         /* drop ref from isolate */
}
static ssize_t  my_write(struct file *file, const char __user *buf, size_t count, loff_t *off)
{

     unsigned long addr = (unsigned long)buf;
     struct vm_area_struct *vma;
     struct migrate_vma args;
     unsigned long size = count;
     unsigned long start, end;
     unsigned long next;
     int ret;
     unsigned long src_pfns[64];
     unsigned long dst_pfns[64];
     start = addr;
     end = start + size;
     if (end < start)
	 return -EINVAL;
#if  LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
     down_read(&current->mm->mmap_sem);
#else
     down_read(&current->mm->context.ldt_usr_sem);
#endif
     for (addr = start; addr < end; addr = next) {
         vma = find_vma(current->mm, addr);
        	if (!vma || addr < vma->vm_start || !(vma->vm_flags & VM_READ)) {
			ret = -EINVAL;
		goto out;
	  }
        next = addr + PAGE_SIZE;
	//next = min(vma->vm_end, end);
        if (next > vma->vm_end)
           next = vma->vm_end;

	//pr_info("vma page : %p \n",vma->page);
	args.vma = vma;
	args.src = src_pfns;
	args.dst = dst_pfns;
	args.start = addr;
	args.end = next;
	//args.pgmap_owner = dmirror->mdevice;
        args.flags = MIGRATE_VMA_SELECT_SYSTEM;
	ret = migrate_vma_setup(&args);
	if (ret)
	{

	    pr_info("migrate_vma_setup fail \n");
	    goto out;
       	}
	dmirror_migrate_alloc_and_copy(&args,start,vma);
	migrate_vma_pages(&args);
	migrate_vma_finalize(&args);
	//zap_vma_ptes(vma, vma->vm_start, vma->vm_end - vma->vm_start);
	//test_migrate_vma_finalize is a bad achievement of migrate_vma_finalize
	//test_migrate_vma_finalize(&args);
     }
out:

#if  LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
     up_read(&current->mm->mmap_sem);
#else
     up_read(&current->mm->context.ldt_usr_sem);
#endif
 
	return ret;
}
#endif
static int my_mmap(struct file *filp, struct vm_area_struct *vma)
{
#if 0
        struct mm_struct *mm = filp->private_data;
        pte_t pte;

        /* Trigger invalidate range [range, start, end] */
        mmu_notifier_range_init(&my_range,  MMU_NOTIFY_UNMAP, 0, vma,mm,
                vma->vm_start & PAGE_MASK, vma->vm_end & PAGE_MASK);
        mmu_notifier_invalidate_range_start(&my_range);
        mmu_notifier_invalidate_range_end(&my_range);

        /* Trigger clear_flush_young */
        mmu_notifier_clear_flush_young(mm, vma->vm_start, vma->vm_end);

        /* Trigger clear_young */
        mmu_notifier_clear_young(mm, vma->vm_start, vma->vm_end);

        /* Trigger test_young */
        mmu_notifier_test_young(mm, vma->vm_start);

        /* Trigger change pte */
        mmu_notifier_change_pte(mm, vma->vm_start, pte);
        /* Trigger realease */
        mmu_notifier_release(mm);
#else
		return remap_pfn_range(vma, vma->vm_start, (virt_to_phys(kaddr) >> PAGE_SHIFT) + vma->vm_pgoff, vma->vm_end - vma->vm_start, vma->vm_page_prot);
#endif
        return 0;
}

static int my_open(struct inode *inode, struct file *file)
{
        struct mm_struct *mm = get_task_mm(current);

        file->private_data = mm;
        /* mmu notifier initialize */
        my_notifier.ops = &my_mmu_notifer_ops;
        /* mmu notifier register */
        mmu_notifier_register(&my_notifier, mm);

        return 0;
}

static int my_release(struct inode *inode, struct file *file)
{
        mmu_notifier_unregister(&my_notifier, current->mm);
        return 0;
}

/* file operations */
static struct file_operations my_fops = {
        .owner          = THIS_MODULE,
        .open           = my_open,
        .mmap           = my_mmap,
        .release        = my_release,
	.read           = my_read,
	.write          = my_write
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

        printk("Hello modules on myMMU\n");
}
module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
