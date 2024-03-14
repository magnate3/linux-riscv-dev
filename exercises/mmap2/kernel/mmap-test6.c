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

static int my_mmap(struct file *file, struct vm_area_struct *vma)
{
  unsigned long pages = (vma->vm_end - vma->vm_start) >> PAGE_SHIFT;
  unsigned long addr;
  struct page *page;
  int i,rc=-ENODEV;
  // TODO need any semaphore for vma manipulation?
  printk(KERN_DEBUG "vma->vm_end %lx vm_start %lx len %lx pages %lx vm_pgoff %lx\n",
    vma->vm_end, vma->vm_start, vma->vm_end - vma->vm_start, pages, vma->vm_pgoff);
  /* allocate and insert pages to fill the vma. */ 
  for(i=0; i < pages; i++) {
    page = alloc_page(GFP_KERNEL); // TODO IO RESERVE?
    if (!page) { 
      // TODO free previous pages
      printk(KERN_DEBUG "alloc_page failed\n");
      goto done;
    }
   pr_info("*********** my_fault6 ******* page addr: %p \n",  page_address(page));
    addr = vma->vm_start+i*PAGE_SIZE;
    if (vm_insert_page(vma,addr,page) < 0) {
      // TODO free previous pages
      printk(KERN_DEBUG "vm_insert_page failed\n");
      goto done;
    }
    printk(KERN_DEBUG "inserted page %d at %p\n",i,(void*)addr);
    // TODO __free_page now, should be ok, since vm_insert_page incremented its
    // refcount. that way, upon munmap, refcount hits zer0, pages get freed 
    __free_page(page);
  }
  printk(KERN_DEBUG "completed inserting %lu pages\n", pages);
  
  rc = 0;

 done:
  return rc;
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
	return misc_register(&mdev);
}
static void __exit my_exit(void)
{
  
      misc_deregister(&mdev);
}
module_init(my_init);
module_exit(my_exit);
MODULE_AUTHOR("Leon He <hexiaolong2008@gmail.com");
MODULE_DESCRIPTION("mmap simple demo");
MODULE_LICENSE("GPL v2");
