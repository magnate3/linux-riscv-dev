/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/fs.h>
#include <linux/slab.h>
#define PAGE_NUM  3
static struct page *pages[PAGE_NUM];
static int my_fault(struct vm_fault *vmf)
{
	struct vm_area_struct *vma = vmf->vma;
        int ret;
        unsigned long addr;
        unsigned long page_num;
        if (!pages[vmf->pgoff])
		pages[vmf->pgoff] = alloc_page(GFP_KERNEL);

        addr = vma->vm_start+vmf->pgoff*PAGE_SIZE; 
        page_num = (vma->vm_end - vma->vm_start) >> PAGE_SHIFT;
        pr_info("addr %lx ,pages %lx  \n", addr, page_num);
        pr_info("vma->vm_end %lx vm_start %lx len %lx \n", vma->vm_end, vma->vm_start, vma->vm_end - vma->vm_start);
        pr_info("*********** my_fault3 *******vmf->address:  %lx , pgoff :%lx , page addr: %p \n",  vmf->address, vmf->pgoff,  page_address(pages[vmf->pgoff]));
	ret = vm_insert_page(vma, vmf->address, pages[vmf->pgoff]);
	if (ret)
		return VM_FAULT_SIGBUS;
        // TODO __free_page now, should be ok, since vm_insert_page incremented its
       //     // refcount. that way, upon munmap, refcount hits zer0, pages get freed 
        __free_page(pages[vmf->pgoff]);
	return VM_FAULT_NOPAGE;
}

static const struct vm_operations_struct vm_ops = {
	.fault = my_fault,
};

static int my_mmap(struct file *file, struct vm_area_struct *vma)
{
	vma->vm_flags |= VM_MIXEDMAP;
	vma->vm_ops = &vm_ops;
	return 0;
}

static struct file_operations my_fops = {
	.owner	= THIS_MODULE,
	.mmap	= my_mmap,
};
 
static struct miscdevice mdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "my_dev",
	.fops = &my_fops,
};
 
static int __init my_init(void)
{
        int i;
        pr_info("******* mmap 5 ********** \n");
        for (i = 0; i<  PAGE_NUM; ++i)
        {
            pages[i] = NULL;
        }
	return misc_register(&mdev);
}
static void __exit my_exit(void)
{
      
      //int i;
      //for (i = 0; i<  PAGE_NUM; ++i)
      //{
      //     if (pages[i]) {
      //          __free_page(pages[i]);
      //     }
      //}
      misc_deregister(&mdev);
}
module_init(my_init);
module_exit(my_exit);
MODULE_AUTHOR("Leon He <hexiaolong2008@gmail.com");
MODULE_DESCRIPTION("mmap simple demo");
MODULE_LICENSE("GPL v2");
