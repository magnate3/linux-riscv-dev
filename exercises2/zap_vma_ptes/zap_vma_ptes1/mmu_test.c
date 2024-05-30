/*
 * MMU notifier on myMMU
 *
 *  
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
