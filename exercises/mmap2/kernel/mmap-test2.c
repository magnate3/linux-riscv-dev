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

static void *kaddr;

static int my_fault(struct vm_fault *vmf)
{
	struct vm_area_struct *vma = vmf->vma;
	int offset, ret;

	offset = vmf->pgoff * PAGE_SIZE;
         pr_info("*********** my_fault2 *******vma->vm_start:  %lx , offset :%x \n",  vmf->address, offset);
	ret = vm_insert_page(vma, vmf->address, virt_to_page(kaddr + offset));
	if (ret)
		return VM_FAULT_SIGBUS;

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
	kaddr = kzalloc(PAGE_SIZE * 3, GFP_KERNEL);
        pr_info("******* alloc mem2 %p ********** \n", kaddr);
	return misc_register(&mdev);
}
static void __exit my_exit(void)
{
      kfree(kaddr);
      misc_deregister(&mdev);
}
module_init(my_init);
module_exit(my_exit);
MODULE_AUTHOR("Leon He <hexiaolong2008@gmail.com");
MODULE_DESCRIPTION("mmap simple demo");
MODULE_LICENSE("GPL v2");
