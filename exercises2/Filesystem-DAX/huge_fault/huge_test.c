// SPDX-License-Identifier: GPL-2.0-only
/*
 * PFNMAP: CUSTOMIZE MAPPED 1Gig
 *
 *   Add "memmap=2M$4G" into CMDLINE
 *
 * 
 */
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/huge_mm.h>
#include <linux/pfn_t.h>
#include <linux/version.h>

#define SPECIAL_DEV_NAME	"test-PFNMAP"
#define PFN_PHYSADDR		0x100000000 /* 128MiB */

#if LINUX_VERSION_CODE <= KERNEL_VERSION(4, 18, 0)
static int vm_huge_fault(struct vm_fault *vmf, 
				enum page_entry_size pe_size)
{
	pfn_t pfn = phys_to_pfn_t(PFN_PHYSADDR, PFN_DEV);
        unsigned long addr = vmf->address & PUD_MASK;
	return vmf_insert_pfn_pud(vmf->vma,addr,vmf->pud, pfn, true);
}
#else
typedef __bitwise unsigned int vm_fault_t;
static vm_fault_t vm_huge_fault(struct vm_fault *vmf, 
				enum page_entry_size pe_size)
{
	pfn_t pfn = phys_to_pfn_t(PFN_PHYSADDR, PFN_DEV);
	//dump_stack();
	return vmf_insert_pfn_pud(vmf, pfn, true);
}
#endif

static const struct vm_operations_struct test_vm_ops = {
	.huge_fault	= vm_huge_fault,
};

static int test_mmap(struct file *filp, struct vm_area_struct *vma)
{
	/* setup vm_ops */
	vma->vm_ops = &test_vm_ops;
	/* fake DAX */
	filp->f_inode->i_flags |= S_DAX;
	/* FPNMAP */
	vma->vm_flags |= VM_PFNMAP;

	return 0;
}

static unsigned long test_get_unmapped_area(struct file *filp,
                unsigned long uaddr, unsigned long len,
                unsigned long pgoff, unsigned long flags)
{
	unsigned long align_addr;

	//dump_stack();
	align_addr = current->mm->get_unmapped_area(NULL, 0,
			len + PUD_SIZE, 0, flags);
	/* Aligned on 1Gig */
	align_addr = round_up(align_addr, PUD_SIZE);

	return align_addr;
}

static struct file_operations test_fops = {
	.owner             = THIS_MODULE,
	.mmap              = test_mmap,
	.get_unmapped_area = test_get_unmapped_area,
};

static struct miscdevice test_drv = {
	.minor	= MISC_DYNAMIC_MINOR,
	.name	= SPECIAL_DEV_NAME,
	.fops	= &test_fops,
};

static int __init test_init(void)
{
	misc_register(&test_drv);
	return 0;
}

static void __exit test_exit(void)
{
	misc_deregister(&test_drv);
}

module_init(test_init);
module_exit(test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("test");
MODULE_DESCRIPTION("test PAGING Project");
