/*
 * Simple - REALLY simple memory mapping demonstration.
 *
 * Copyright (C) 2001 Alessandro Rubini and Jonathan Corbet
 * Copyright (C) 2001 O'Reilly & Associates
 *
 * The source code in this file can be freely used, adapted,
 * and redistributed in source or binary form, so long as an
 * acknowledgment appears in derived source files.  The citation
 * should list that the code comes from the book "Linux Device
 * Drivers" by Alessandro Rubini and Jonathan Corbet, published
 * by O'Reilly & Associates.   No warranty is attached;
 * we cannot take responsibility for errors or fitness for use.
 *
 * $Id: simple.c,v 1.12 2005/01/31 16:15:31 rubini Exp $
 */

#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>

#include <linux/kernel.h>   /* printk() */
#include <linux/slab.h>   /* kmalloc() */
#include <linux/fs.h>       /* everything... */
#include <linux/errno.h>    /* error codes */
#include <linux/types.h>    /* size_t */
#include <linux/mm.h>
#include <linux/kdev_t.h>
#include <asm/page.h>
#include <linux/cdev.h>
#include <linux/sched/mm.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <uapi/asm-generic/mman-common.h>
#include<linux/uaccess.h>
#include"simple.h"


#include <linux/device.h>

static struct class *demo_class;
struct device *demo_device;
static int simple_major = 0;
module_param(simple_major, int, 0);
MODULE_AUTHOR("Jonathan Corbet");
MODULE_LICENSE("Dual BSD/GPL");
static int move_vma_to(unsigned long src, unsigned long dst);

/*
 * Open the device; in fact, there's nothing to do here.
 */
static int ldd_simple_open (struct inode *inode, struct file *filp)
{
	return 0;
}


/*
 * Closing is just as simpler.
 */
static int simple_release(struct inode *inode, struct file *filp)
{
	return 0;
}



/*
 * Common VMA ops.
 */

void simple_vma_open(struct vm_area_struct *vma)
{
	printk(KERN_NOTICE "Simple VMA open, virt %lx, phys %lx\n",
			vma->vm_start, vma->vm_pgoff << PAGE_SHIFT);
}

void simple_vma_close(struct vm_area_struct *vma)
{
	printk(KERN_NOTICE "Simple VMA close.\n");
}


/*
 * The remap_pfn_range version of mmap.  This one is heavily borrowed
 * from drivers/char/mem.c.
 */

static struct vm_operations_struct simple_remap_vm_ops = {
	.open =  simple_vma_open,
	.close = simple_vma_close,
};

static int simple_remap_mmap(struct file *filp, struct vm_area_struct *vma)
{
	printk (KERN_NOTICE "---- simple_remap_mmap \n");
        // remap_pfn_range(vma, vma->vm_start, pfn_start, size, vma->vm_page_prot);
	if (remap_pfn_range(vma, vma->vm_start, vma->vm_pgoff,
			    vma->vm_end - vma->vm_start,
			    vma->vm_page_prot))
		return -EAGAIN;
	printk (KERN_NOTICE "---- simple_remap_mmap and call  simple_vma_open\n");

	vma->vm_ops = &simple_remap_vm_ops;
	simple_vma_open(vma);
	return 0;
}




static inline
struct vm_area_struct *vma_lookup(struct mm_struct *mm, unsigned long addr)
{
	struct vm_area_struct *vma = find_vma(mm, addr);

	if (vma && addr < vma->vm_start)
		vma = NULL;

	return vma;
}
long device_ioctl(struct file *file,	
		 unsigned int ioctl_num,
		 unsigned long ioctl_param)
{
	//unsigned long addr = 1234;
	int ret = 0; // on failure return -1
	struct address * buff = NULL;
	unsigned long vma_addr = 0;
	unsigned long to_addr = 0;
	//unsigned length = 0;
	//struct input* ip;
	// unsigned index = 0;
	//struct address temp;
        //struct address *mapping;
	/*
	 * Switch according to the ioctl called
	 */
	switch (ioctl_num) {
	case IOCTL_MVE_VMA_TO:
	    buff = (struct address*)vmalloc(sizeof(struct address)) ;
	    printk("move VMA at a given address");
	    if(copy_from_user(buff,(char*)ioctl_param,sizeof(struct address))){
	        pr_err("MVE_VMA address write error\n");
		return ret;
	    }
	    vma_addr = buff->from_addr;
	    to_addr = buff->to_addr;
	    printk("address from :%lx, to:%lx \n",vma_addr,to_addr);
	    vfree(buff);
            ret = move_vma_to(vma_addr, to_addr);

	    return ret;
	}
	return ret;
}
static int move_vma_to(unsigned long src, unsigned long dst) {
        struct vm_area_struct *src_vma, *after_dst_vma;
        struct mm_struct *mm;
        size_t copy_len;
        unsigned long dst_addr;
        unsigned long src_prot, prot_flags = 0, map_flags = 0;
        char *src_buffer;
        mm = get_task_mm(current);
        src_vma = vma_lookup(mm, src);
        if (!src_vma) {
                printk(KERN_INFO "illegal source address: vma not found\n");
                return -1;
        }
        after_dst_vma = find_vma(mm, dst);
        copy_len =  (size_t)(src_vma->vm_end - src_vma->vm_start);
        src_buffer = (char*)kzalloc(copy_len*sizeof(char), GFP_KERNEL);
#if 0
        if(after_dst_vma != NULL) {
                // check if dst address lies inside this vma
                //if (dst >= after_dst_vma->vm_start) {
                if (dst >= after_dst_vma->vm_start && dst <= after_dst_vma->vm_end) {
                        printk(KERN_INFO "destination address already mapped\n");
                        goto err;
                }
                // check if space between dst and after_dst_vma->vm_start is large enough
                else if (copy_len >= after_dst_vma->vm_start - dst) {
                        printk(KERN_INFO "not enough space present at the destination\n");
                        goto err;
                }
        }
        
#endif
        src_prot = src_vma->vm_page_prot.pgprot;

        if (src_prot & PROT_READ) {
                prot_flags |= PROT_READ;
        }
        if (src_prot & PROT_WRITE) {
                prot_flags |= PROT_WRITE;
        }
        if (src_prot & PROT_EXEC) {
                prot_flags |= PROT_EXEC;
        }
        if (src_vma->vm_flags & MAP_SHARED) {
                map_flags |= MAP_SHARED;
        }
        if (src_vma->vm_flags & MAP_FIXED) {
                map_flags |= MAP_FIXED;
        }
        if(src_vma->vm_flags & MAP_PRIVATE) {
                map_flags |= MAP_PRIVATE;
        }
	if(src_vma->vm_flags & MAP_ANONYMOUS) {
		map_flags |= MAP_ANONYMOUS;
	}

        dst_addr = vm_mmap(NULL, dst, copy_len, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, 0);
        
        if (dst_addr != dst) {
                printk(KERN_INFO "vm_mmap failed\n");
                // vm_munmap(dst_addr, copy_len);
                goto err;
        }
        if(copy_from_user(src_buffer, (void*)(src_vma->vm_start), copy_len)){
                printk(KERN_INFO "copy_from_user failed\n");
                goto err;
        }

        if(copy_to_user((void*)dst_addr, src_buffer, copy_len)){
                printk(KERN_INFO "copy_to_user failed\n");
                goto err;
        }
	
	vm_munmap(src_vma->vm_start, copy_len);
        kfree(src_buffer);
        return 0;
err:
        kfree(src_buffer);
	return -1;
}


/*
 * Set up the cdev structure for a device.
 */
static void simple_setup_cdev(struct cdev *dev, int minor,
		struct file_operations *fops)
{
	int err, devno = MKDEV(simple_major, minor);
    
	cdev_init(dev, fops);
	dev->owner = THIS_MODULE;
	dev->ops = fops;
	err = cdev_add (dev, devno, 1);
	/* Fail gracefully if need be */
	if (err)
		printk (KERN_NOTICE "Error %d adding simple%d", err, minor);
}


/*
 * Our various sub-devices.
 */
/* Device 0 uses remap_pfn_range */
static struct file_operations simple_remap_ops = {
	.owner   = THIS_MODULE,
	.open    = ldd_simple_open,
	.release = simple_release,
	//.mmap    = simple_remap_mmap,
	.unlocked_ioctl = device_ioctl,
};

#define MAX_SIMPLE_DEV 1

#if 0
static struct file_operations *simple_fops[MAX_SIMPLE_DEV] = {
	&simple_remap_ops,
	&simple_fault_ops,
};
#endif


#if 0
/*
 * We export two simple devices.  There's no need for us to maintain any
 * special housekeeping info, so we just deal with raw cdevs.
 */
static struct cdev SimpleDevs[MAX_SIMPLE_DEV];
/*
 * Module housekeeping.
 */
static int simple_init(void)
{
	int result;
	dev_t dev = MKDEV(simple_major, 0);

	/* Figure out our device number. */
	if (simple_major)
		result = register_chrdev_region(dev, 2, "simple");
	else {
		result = alloc_chrdev_region(&dev, 0, 2, "simple");
		simple_major = MAJOR(dev);
	}
	if (result < 0) {
		printk(KERN_WARNING "simple: unable to get major %d\n", simple_major);
		return result;
	}
	if (simple_major == 0)
		simple_major = result;

	/* Now set up two cdevs. */
	simple_setup_cdev(SimpleDevs, 0, &simple_remap_ops);
	return 0;
}


static void simple_cleanup(void)
{
	cdev_del(SimpleDevs);
	unregister_chrdev_region(MKDEV(simple_major, 0), 1);
}
#else
static char *demo_devnode(struct device *dev, umode_t *mode)
{
        if (mode && dev->devt == MKDEV(simple_major, 0))
                *mode = 0666;
        return NULL;
}
int simple_init(void)
{
        int err;
	printk(KERN_INFO "Hello kernel\n");
        simple_major = register_chrdev(0, DEVNAME, &simple_remap_ops);
        err = simple_major;
        if (err < 0) {      
             printk(KERN_ALERT "Registering char device failed with %d\n", simple_major);   
             goto error_regdev;
        }                 
        
        demo_class = class_create(THIS_MODULE, "simple");
        err = PTR_ERR(demo_class);
        if (IS_ERR(demo_class))
                goto error_class;

        //demo_class->devnode = demo_devnode;

        demo_device = device_create(demo_class, NULL,
                                        MKDEV(simple_major, 0),
                                        NULL, "simpler");
        err = PTR_ERR(demo_device);
        if (IS_ERR(demo_device))
                goto error_device;
 
        printk(KERN_INFO "I was assigned major number %d. To talk to\n", simple_major);                                                              
        return 0;
error_device:
         class_destroy(demo_class);
error_class:
        unregister_chrdev(simple_major, DEVNAME);
error_regdev:
        return  err;
}

void simple_cleanup(void)
{
        device_destroy(demo_class, MKDEV(simple_major, 0));
        class_destroy(demo_class);
        unregister_chrdev(simple_major, DEVNAME);
	printk(KERN_INFO "Goodbye kernel\n");
}
#endif

module_init(simple_init);
module_exit(simple_cleanup);
