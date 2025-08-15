#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/miscdevice.h>  
#include <linux/delay.h>  
#include <linux/kernel.h>  
#include <linux/module.h>  
#include <linux/init.h>  
#include <linux/mm.h>  
#include <linux/fs.h>  
#include <linux/types.h>  
#include <linux/delay.h>  
#include <linux/moduleparam.h>  
#include <linux/slab.h>  
#include <linux/errno.h>  
#include <linux/ioctl.h>  
#include <linux/cdev.h>  
#include <linux/string.h>  
#include <linux/list.h>  
#include <linux/pci.h>  
#include <linux/gpio.h>  

static char MMAP_TEST_STR[1024] = "reverse shell event{\n\t'evt':'rvshell',\n\t'pid':'19256',\n\t'exe':'/bin/bash',\n\t'cmdline':'bash',\n\t'cwd':'/root/Felicia',\n\t'ppid':'19255',\n\t'pexe':'/usr/bin/socat'\n}";

static struct cdev mmap_drv;
static dev_t ndev;
char *buffer;


static int mmap_drv_open(struct inode *nd, struct file *filp)
{
    return 0;
}


static int mmap_drv_mmap(struct file *filp, struct vm_area_struct *vma)
{
    // unsigned long page; 
    struct page *page = NULL; 
    unsigned char i;  
    unsigned long int len;
    unsigned long start = (unsigned long)vma->vm_start;  
    //unsigned long end =  (unsigned long)vma->vm_end;  
    unsigned long size = (unsigned long)(vma->vm_end - vma->vm_start);  
    // page = virt_to_phys(buffer);  
    page = virt_to_page((unsigned long) buffer + (vma->vm_pgoff << PAGE_SHIFT));    
    // if(remap_pfn_range(vma,start,page>>PAGE_SHIFT,size,vma->vm_page_prot))
    if(remap_pfn_range(vma,start,page_to_pfn(page),size,vma->vm_page_prot))
        return -1;  
    len = strlen(MMAP_TEST_STR);
    for(i=0;i<len;i++)  
        buffer[i] = MMAP_TEST_STR[i];  
    return 0;  
}

struct file_operations mmap_drv_ops = {
    .owner = THIS_MODULE,
    .open = mmap_drv_open,
    .mmap = mmap_drv_mmap,
};


static int mmap_drv_init(void)
{
    int ret;

    cdev_init(&mmap_drv, &mmap_drv_ops);
    ret = alloc_chrdev_region(&ndev, 0, 1, "evt_map");
    if (ret < 0) {
        return ret;
    }
    buffer = (unsigned char *)kmalloc(PAGE_SIZE,GFP_KERNEL);
    printk(KERN_INFO "mmap_drv_init: major=%d minor=%d\n", MAJOR(ndev), MINOR(ndev));
    ret = cdev_add(&mmap_drv, ndev, 1);
    if (ret < 0) {
        return ret;
    }
    
    return 0;
}

static void mmap_drv_exit(void)
{
    cdev_del(&mmap_drv);
    unregister_chrdev_region(ndev, 1);
}

module_init(mmap_drv_init);
module_exit(mmap_drv_exit);
MODULE_LICENSE("GPL");