// #include <linux/config.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>
#include <linux/debugfs.h>
#include <linux/kernel.h>   /* printk() */
#include <linux/slab.h>   /* kmalloc() */
#include <linux/fs.h>       /* everything... */
#include <linux/errno.h>    /* error codes */
#include <linux/types.h>    /* size_t */
#include <linux/mm.h>
#include <linux/kdev_t.h>
#include <asm/page.h>
#include <linux/cdev.h>
#include <linux/device.h>

#ifndef VM_RESERVED
# define  VM_RESERVED   (VM_DONTEXPAND | VM_DONTDUMP)
#endif

#define RAW_DATA_SIZE 31457280
#define RAW_DATA_OFFSET 0x80000000UL



void *rawdataStart;

struct dentry  *file;



/*
 * Open the device; in fact, there's nothing to do here.
 */
int simple_open (struct inode *inode, struct file *filp)
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



static int simple_remap_mmap(struct file *filp, struct vm_area_struct *vma)
{

    int ret;
        unsigned long mapoffset;
        mapoffset = RAW_DATA_OFFSET + (vma->vm_pgoff << PAGE_SHIFT);
        ret = remap_pfn_range(vma, vma->vm_start, mapoffset >> PAGE_SHIFT,
                              vma->vm_end - vma->vm_start, PAGE_SHARED);

        if ( ret != 0 ) {
            printk("Error remap_pfn_range. \n");
            return -EAGAIN;
        }
        return 0;
}

/* Device  uses remap_pfn_range */
static struct file_operations simple_remap_ops = {
    .owner   = THIS_MODULE,
    .open    = simple_open,
    .release = simple_release,
    .mmap    = simple_remap_mmap,
};

/*
 * Module housekeeping.
 */
static int simple_init(void)
{
    file = debugfs_create_file("mmap_example", 0644, NULL, NULL, &simple_remap_ops);
    rawdataStart = ioremap(RAW_DATA_OFFSET, RAW_DATA_SIZE);
    if (rawdataStart!=NULL){
        printk("rawdataStart at:%p  \n", rawdataStart);
        memset(rawdataStart, 'c', 20971520);
        memset(rawdataStart+20971520, '$', 100);

    }else{
        printk("rawdataStart is NULL \n");
        return -1;
    }



    return 0;
}


static void simple_cleanup(void)
{
    debugfs_remove(file);
    if (rawdataStart != NULL) {
            printk(KERN_INFO "Unmapping memory at %p\n", rawdataStart);
            iounmap(rawdataStart);
        } else {
            printk(KERN_WARNING "No memory to unmap!\n");
        }
}


module_init(simple_init);
module_exit(simple_cleanup);
MODULE_AUTHOR("Jonathan Corbet");
MODULE_LICENSE("Dual BSD/GPL");
