#include <linux/module.h>
#include <linux/kernel.h>
#include <asm/uaccess.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/mm.h>

/* module information */
MODULE_AUTHOR("Troy D. Hanson");
MODULE_DESCRIPTION("Example of character device");
MODULE_LICENSE("Dual BSD/GPL");

#define NUM_MINORS 1

/* a global to keep state. must be thread safe. */
struct chardev_t {
  dev_t dev;        /* has major and minor bits */
  struct cdev cdev; /* has our ops, owner, etc */
} c;
int address_space(struct vm_area_struct *vma)
{
     //struct vm_area_struct *vma = vmf->vma;
     if (!vma->vm_file) {
         printk(KERN_INFO"anon vma %p\n", vma->anon_vma);
         return 0;
     }
     struct inode *inode = file_inode(vma->vm_file);
     struct address_space *mapping = inode->i_mapping;
     const char *name = vma->vm_file->f_path.dentry->d_iname;
     printk(KERN_INFO"mapping->a_ops %p, name %s\n", mapping->a_ops, name);
     return 0;
}
int _mmap(struct file *f, struct vm_area_struct *vma) {
  unsigned long pages = (vma->vm_end - vma->vm_start) >> PAGE_SHIFT;
  unsigned long addr;
  struct page *page;
  int i,rc=-ENODEV;
  // TODO need any semaphore for vma manipulation?
  printk(KERN_DEBUG "vma->vm_end %lu vm_start %lu len %lu pages %lu vm_pgoff %lu\n",
    vma->vm_end, vma->vm_start, vma->vm_end - vma->vm_start, pages, vma->vm_pgoff);
  /* allocate and insert pages to fill the vma. */ 
  for(i=0; i < pages; i++) {
    page = alloc_page(GFP_KERNEL); // TODO IO RESERVE?
    if (!page) { 
      // TODO free previous pages
      printk(KERN_DEBUG "alloc_page failed\n");
      goto done;
    }
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
    address_space(vma);
  }
  printk(KERN_DEBUG "completed inserting %lu pages\n", pages);
  
  rc = 0;

 done:
  return rc;
}

struct file_operations ops = {
  .mmap = _mmap
};

int __init chardev_init(void) {
  int rc;

  /* ask for a dynamic major */
  rc = alloc_chrdev_region(&c.dev, 0, NUM_MINORS, "kex");
  if (rc) { 
    rc = -ENODEV;
    goto done; 
  }

  /* init the struct cdev */
  cdev_init(&c.cdev, &ops);
  c.cdev.owner = THIS_MODULE;

  /* make device live */
  rc = cdev_add(&c.cdev, c.dev, NUM_MINORS);
  if (rc) {
    rc = -ENODEV;
    printk(KERN_WARNING "cdev_add: can't add device\n");
    unregister_chrdev_region(c.dev, NUM_MINORS);
    cdev_del(&c.cdev);
    goto done;
  }

  rc = 0;

 done:
  return rc;
}

void __exit chardev_cleanup(void) {
  cdev_del(&c.cdev);
  unregister_chrdev_region(c.dev, NUM_MINORS);
}

module_init(chardev_init);
module_exit(chardev_cleanup);
