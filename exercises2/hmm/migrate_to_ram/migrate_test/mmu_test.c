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
struct dev_pagemap   pgmap;
struct page		*head=NULL;
#define DEVMEM_CHUNK_SIZE		(8U)
//#define DEVMEM_CHUNK_SIZE		(256 * 1024 * 1024U)
struct dmirror_device_t {
		struct cdev *		cdevice;
};
struct dmirror_device_t dmirror_device;
static void my_mmu_release(struct mmu_notifier *mn,
                                                struct mm_struct *mm)
{
        //printk("myMMU notifier: release\n");
}

static int my_mmu_clear_flush_young(struct mmu_notifier *mn,
                struct mm_struct *mm, unsigned long start, unsigned long end)
{
        //printk("myMMU notifier: clear_flush_young\n");
        return 0;
}

static int my_mmu_clear_young(struct mmu_notifier *mn,
                struct mm_struct *mm, unsigned long start, unsigned long end)
{
        //printk("myMMU notifier: clear_young\n");
        return 0;
}

static int my_mmu_test_young(struct mmu_notifier *mn,
                        struct mm_struct *mm, unsigned long address)
{   
        //printk("myMMU notifier: test_young\n");
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
        //printk("myMMU notifier: invalidate_range_start.\n");
        return 0;
}

static void my_mmu_invalidate_range_end(struct mmu_notifier *mn,
                                const struct mmu_notifier_range *range)
{
        //printk("myMMU notifier: invalidate_range_end.\n");
}

static void my_mmu_invalidate_range(struct mmu_notifier *mn,
                struct mm_struct *mm, unsigned long start, unsigned long end)
{
	//dump_stack();
        //printk("myMMU notifier: invalidate_range.\n");
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
struct page * get_dpage(void)
{
	struct page *dpage = head;
	if(NULL == head)
	{
	      return NULL;
	}
        head = dpage->zone_device_data;	
	return dpage;
}
bool  dpage_is_null(void)
{
      return NULL == head;
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
		struct page *rpage;

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
		rpage = alloc_page(GFP_HIGHUSER);
		//rpage = alloc_page_vma(GFP_HIGHUSER, vma, start);
		//dpage = alloc_page_vma(GFP_HIGHUSER, vmf->vma, vmf->address);

		dpage = get_dpage();
		dpage->zone_device_data = rpage;
#endif
		get_page(dpage);
		lock_page(dpage);
		if (spage)
		{
		        pr_info("%s call copy highpage  ,dpage @ %p, spage @ %p \n", __func__,rpage, spage);
			copy_highpage(rpage, spage);
			if(addr == args->start)
			{
			    cmp_page(current->mm, spage, rpage);
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
                printk(KERN_INFO "------------- pte present \n");
    }
    if(is_swap_pte(pte)){
	swp_entry_t entry;
	entry = pte_to_swp_entry(pte);
        printk(KERN_INFO "************* swap pte\n");
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
	    printk(KERN_INFO "swap pte,device page frame struct is @ %p, and user paddr 0x%lx, virt addr 0x%lx", page, paddr, addr);
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
            printk(KERN_INFO " not swap pte,page frame struct is @ %p, and user paddr 0x%lx, virt addr 0x%lx \n", page, paddr, addr);
	    //pte_clear_soft_dirty(pte);
        }
	else
	{
            printk(KERN_INFO " not swap pte,virt addr 0x%lx  , page frame not exist \n",addr);
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
	 if(dpage_is_null())
	 {
              return 0;
         }
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
	args.pgmap_owner = &dmirror_device;
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
static void dmirror_devmem_free(struct page *page)
{
	struct page *rpage = page->zone_device_data;

	if (rpage)
  	   __free_page(rpage);
	page->zone_device_data = head;
	head = page;
}
static vm_fault_t dmirror_devmem_fault_alloc_and_copy(struct migrate_vma *args)
{
	const unsigned long *src = args->src;
	unsigned long *dst = args->dst;
	unsigned long start = args->start;
	unsigned long end = args->end;
	unsigned long addr;

	for (addr = start; addr < end; addr += PAGE_SIZE,
				       src++, dst++) {
		struct page *dpage, *spage;

		spage = migrate_pfn_to_page(*src);
		if (!spage || !(*src & MIGRATE_PFN_MIGRATE))
			continue;
		spage = spage->zone_device_data;

		dpage = alloc_page_vma(GFP_HIGHUSER_MOVABLE, args->vma, addr);
		if (!dpage)
			continue;

		lock_page(dpage);
		copy_highpage(dpage, spage);
		*dst = migrate_pfn(page_to_pfn(dpage)) | MIGRATE_PFN_LOCKED;
		if (*src & MIGRATE_PFN_WRITE)
			*dst |= MIGRATE_PFN_WRITE;
	}
	return 0;
}

static vm_fault_t dmirror_devmem_fault(struct vm_fault *vmf)
{
	struct migrate_vma args;
	unsigned long src_pfns;
	unsigned long dst_pfns;
	struct page *rpage;
	vm_fault_t ret;
	//dump_stack();

	/*
	 * Normally, a device would use the page->zone_device_data to point to
	 * the mirror but here we use it to hold the page for the simulated
	 * device memory and that page holds the pointer to the mirror.
	 */
	rpage = vmf->page->zone_device_data;

	/* FIXME demonstrate how we can adjust migrate range */
	args.vma = vmf->vma;
	args.start = vmf->address;
	args.end = args.start + PAGE_SIZE;
	args.src = &src_pfns;
	args.dst = &dst_pfns;
	args.pgmap_owner = &dmirror_device;
	args.flags = MIGRATE_VMA_SELECT_DEVICE_PRIVATE;

	if (migrate_vma_setup(&args))
		return VM_FAULT_SIGBUS;

	ret = dmirror_devmem_fault_alloc_and_copy(&args);
	if (ret)
		return ret;
	migrate_vma_pages(&args);
	/*
	 * No device finalize step is needed since
	 * dmirror_devmem_fault_alloc_and_copy() will have already
	 * invalidated the device page table.
	 */
	migrate_vma_finalize(&args);
	pr_info("%s addr 0x%lx \n",__func__,vmf->address);
	return 0;
}
static const struct dev_pagemap_ops dmirror_devmem_ops = {
	.page_free	= dmirror_devmem_free,
	.migrate_to_ram	= dmirror_devmem_fault,
};
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

static int __init my_init(void)
{
	struct resource *res;
	void *r;
	int ret = 0;
	unsigned long pfn;
	unsigned long pfn_first;
	unsigned long pfn_last;
	res = request_free_mem_region(&iomem_resource, DEVMEM_CHUNK_SIZE, "hmm_dmirror");
	if (IS_ERR(res))
	   goto err1;
        pgmap.type = MEMORY_DEVICE_PRIVATE;
	pgmap.range.start = res->start;
	pgmap.range.end = res->end;
	pgmap.nr_range = 1;
	pgmap.ops = &dmirror_devmem_ops;
	pgmap.owner = &dmirror_device;
#if 0
	r = devm_memremap_pages(adev->dev, pgmap);
#else
	r = memremap_pages(&pgmap, numa_node_id());
	if (IS_ERR(r))
        	goto err_release;
#endif
	pfn_first = pgmap.range.start >> PAGE_SHIFT;
	pfn_last = pfn_first + (range_len(&pgmap.range) >> PAGE_SHIFT);
	for (pfn = pfn_first; pfn < pfn_last; pfn++) {
	    struct page *page = pfn_to_page(pfn);
	    page->zone_device_data = head;
	    head = page;
	}
	kaddr = kzalloc(PAGE_SIZE * 3, GFP_KERNEL);
        /* Register Misc device */
        ret = misc_register(&my_drv);
	if(ret)
	{
	     goto err2;
	}
        printk("Hello modules on myMMU\n");
	return 0;
err2:

	memunmap_pages(&pgmap);
err_release:
	release_mem_region(pgmap.range.start, range_len(&pgmap.range));
err1:
	return -1;
}
static void __exit my_exit(void)
{
	kfree(kaddr);
        /* Register Misc device */
        misc_deregister(&my_drv);
	memunmap_pages(&pgmap);
	release_mem_region(pgmap.range.start, range_len(&pgmap.range));
        printk("good bye modules on myMMU\n");
}
module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
