#include <linux/module.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/platform_device.h>
#include <linux/cdev.h>
#include <linux/ioctl.h>
#include <linux/gpio.h>
#include <linux/mm.h>
#include <linux/errno.h>
#include <linux/uaccess.h>
//#include <asm/arch/hardware.h>
//#include <asm/arch/gpio.h>
#include <asm/uaccess.h>
#include <linux/slab.h> /*for Kmalloc*/
#include <asm/io.h>/*for virt_to_phys*/
//#include <asm/hardware.h>
//#include <asm/arch/at91_pio.h>
//#include <asm/arch/at91rm9200.h>/*for sram address and size*/
 
#define SRAM0_PHYADDR 0x300000
#define SRAM0_SIZE 0x1000  /*4K*/
 
 
int sram_major = 242;
 
int sram_open(struct inode *inode,struct file *filp);
int sram_release(struct inode *inode,struct file *filp);
int sram_release(struct inode *inode,struct file *filp);
ssize_t sram_read(struct file *filp , char __user *buf ,size_t count,loff_t *f_pos);
ssize_t sram_write(struct file *filp , char __user *buf ,size_t count,loff_t *f_pos);
//int sram_mmap(struct file *filp, struct vm_area_struct *vma );
//static void sram_setup_cdev(struct sram_dev *dev,int index);
int sram_init(void);
void sram_cleanup(void);
 
 
MODULE_AUTHOR("Cun Tian Rui");
MODULE_LICENSE("Dual BSD/GPL");
 
 
struct sram_dev
{
	struct cdev cdev;	
	unsigned long startAddr;
	unsigned long size;
	char *mapPtr;
}sram_dev;
 
struct sram_dev *sramdevp;
 
struct file_operations sram_fops = 
{
    .owner = THIS_MODULE,
   // .ioctl = sram_ioctl,
    .open  = sram_open,
    .release = sram_release,
    .read = sram_read,
    .write = sram_write,
   // .mmap = sram_mmap
};
 
 
 
int sram_open(struct inode *inode,struct file *filp)
{
    struct sram_dev *devp;
    devp = container_of(inode->i_cdev,struct sram_dev,cdev);
    filp->private_data = devp;
    
    return 0;
}
 
int sram_release(struct inode *inode,struct file *filp)
{
    return 0;    
}
 
ssize_t sram_read(struct file *filp , char __user *buf ,size_t count,loff_t *f_pos)
{
	
	unsigned long p = *f_pos;
        struct sram_dev *devp = filp->private_data;
	printk("Driver read function running\n");
	printk("p is %d\n",p);
	 if (p >= devp->size)
  		return -EINVAL;
	 if (count > devp->size-p)
 		 count = devp->size-p;
	 printk("count is %d\n",count);
         if (copy_to_user(buf, devp->mapPtr+p,count))
  		return -EFAULT;
        *f_pos+= count;
 	return count;	 
} 
 
ssize_t sram_write(struct file *filp , char __user *buf ,size_t count,loff_t *f_pos)
{
	
        unsigned long p = *f_pos;
        printk("Driver write function running\n");
        printk("p is %d\n",p);
	struct sram_dev *devp = filp->private_data;
	if (p >= devp->size)
  		return -EINVAL;
 	if (count > devp->size-p)
  		count = devp->size-p;
        printk("count is %d\n",count);  
 	if (copy_from_user( devp->mapPtr+p,buf,count))
  		return -EFAULT;
 		*f_pos += count;
 	return count;
	
}
/*
int sram_mmap(struct file *filp, struct vm_area_struct *vma )
{
	unsigned long offset = vma->vm_pgoff<<PAGE_SHIFT; 
	unsigned long size = vma->vm_end - vma->vm_start; 
	if ( size > SRAM0_SIZE ) 
	{ 
		printk("size too big\n"); 
		return(-ENXIO); 
	} 
        offset = offset + SRAM0_PHYADDR; 
	
	vma->vm_flags |= VM_LOCKED; 
	if ( remap_pfn_range(vma,vma->vm_start,offset,size,PAGE_SHARED)) 
	{ 
		printk("remap page range failed\n"); 
		return -ENXIO; 
	} 
	return(0); 
}
*/
 
int sram_init(void)
{
    int result;
    int err,ret;
    int devno = MKDEV(sram_major,0);
 
    dev_t dev = MKDEV(sram_major,0);
    if(sram_major)
    {
        
        result = register_chrdev_region(dev,1,"SRAM0");
    }
 
    if(result < 0)
    {
        return result;
    }
 
    sramdevp = kmalloc(sizeof(struct sram_dev),GFP_KERNEL);
    if(!sramdevp)
    {
        result = - ENOMEM;
        goto fail_malloc;
    }
 
    memset(sramdevp,0,sizeof(struct sram_dev));
    //sram_setup_cdev(sramdevp,0);
    
 
    sramdevp->startAddr = SRAM0_PHYADDR;
    sramdevp->size = SRAM0_SIZE;
    ret = request_mem_region(SRAM0_PHYADDR, SRAM0_SIZE, "SRAM0 Region");
    if(ret==NULL)
	{
		printk("Request Memory Region Failed!\n");
		return -1;
	}
    sramdevp->mapPtr = ioremap(SRAM0_PHYADDR,SRAM0_SIZE);
    cdev_init(&sramdevp->cdev,&sram_fops);
    sramdevp->cdev.owner = THIS_MODULE;
    sramdevp->cdev.ops = &sram_fops;
    err = cdev_add(&sramdevp->cdev,devno,1);
 
    if(err)
    {
        printk(KERN_NOTICE "Error %d adding SRAM%d",err,0);
    }
    
    printk( "SRAM0_virt_addr = 0x%lx\n", (unsigned long)sramdevp->mapPtr );  
    return 0;
 
    fail_malloc:
		unregister_chrdev_region(dev,sramdevp);
		kfree(sramdevp);
    return result;
    
}
 
void sram_cleanup(void)
{
    cdev_del(&sramdevp->cdev);   
    iounmap(sramdevp->mapPtr);
    kfree(sramdevp);
    unregister_chrdev_region(MKDEV(sram_major,0),1);  
}
 
module_init(sram_init);
module_exit(sram_cleanup);
 
