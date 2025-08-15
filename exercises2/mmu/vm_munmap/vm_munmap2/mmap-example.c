/*
 * Inspired in https://coherentmusings.wordpress.com/2014/06/10/implementing-mmap-for-transferring-data-from-user-space-to-kernel-space/
 *
 *  This approach uses a char device to comunicate with user space instead debugfs
 *
 */
#include <linux/mman.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>
#include "ioctl_example.h"

#ifndef VM_RESERVED
#define  VM_RESERVED   (VM_DONTEXPAND | VM_DONTDUMP)
#endif

#define DEVICE_NAME "mmap-test"
#define DEVICE_CLASS "tmmap"

#define NR_PAGES_ORDER  2
/* A mutex will ensure that only one process accesses our device */
static DEFINE_MUTEX(mmap_device_mutex);

struct mmap_info {
	char *data;
	int reference;
};

/* Device variables */
static dev_t sample_dev_t = 0;
static struct cdev *sample_cdev;
static struct class *sample_class;

void mmap_open(struct vm_area_struct *vma)
{
	struct mmap_info *info = (struct mmap_info *)vma->vm_private_data;
	info->reference++;
}

void mmap_close(struct vm_area_struct *vma)
{
	struct mmap_info *info = (struct mmap_info *)vma->vm_private_data;
	info->reference--;
}

//static int mmap_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
static int mmap_fault(struct vm_fault *vmf)
{
	struct page *page;
	struct mmap_info *info;
        unsigned long offset;
        struct vm_area_struct *vma = vmf->vma;
	info = (struct mmap_info *)vma->vm_private_data;
	if (!info->data) {
		printk("No data\n");
		return 0;
	}
        offset = ((unsigned long)vmf->address) - ((unsigned long)vmf->vma->vm_start);
        offset = offset >>  PAGE_SHIFT;
        if (offset > (1 << NR_PAGES_ORDER)) {
            printk(KERN_ERR "Invalid address deference, offset = %lu \n",
           offset);
          return 0;
        }
        printk(KERN_ERR "page index offset = %lu \n",offset);
	page = virt_to_page(info->data) + offset;

	get_page(page);
	vmf->page = page;

	return 0;
}

struct vm_operations_struct mmap_vm_ops = {
	.open = mmap_open,
	.close = mmap_close,
	.fault = mmap_fault,
};

int op_mmap(struct file *filp, struct vm_area_struct *vma)
{
	vma->vm_ops = &mmap_vm_ops;
	vma->vm_flags |= VM_RESERVED;
	vma->vm_private_data = filp->private_data;
	mmap_open(vma);
	return 0;
}

int mmapfop_close(struct inode *inode, struct file *filp)
{
	struct mmap_info *info = filp->private_data;

	free_page((unsigned long)info->data);
	kfree(info);
	filp->private_data = NULL;

	mutex_unlock(&mmap_device_mutex);

	return 0;
}

int mmapfop_open(struct inode *inode, struct file *filp)
{
	struct mmap_info *info = NULL;

	if (!mutex_trylock(&mmap_device_mutex)) {
		printk(KERN_WARNING
		       "Another process is accessing the device\n");
		return -EBUSY;
	}

	info = kmalloc(sizeof(struct mmap_info), GFP_KERNEL);
#if 0
	info->data = (char *)get_zeroed_page(GFP_KERNEL);
#else
         info->data = (char *)__get_free_pages(GFP_KERNEL, NR_PAGES_ORDER);
#endif
	memcpy(info->data, "Hello from kernel this is file: ", 32);
	memcpy(info->data + 32, filp->f_path.dentry->d_name.name,
	       strlen(filp->f_path.dentry->d_name.name));
	/* assign this info struct to the file */
	filp->private_data = info;
	return 0;
}
const int total = (1 << NR_PAGES_ORDER)*PAGE_SIZE;
long device_ioctl(struct file *file,
                 unsigned int ioctl_num,
                 unsigned long ioctl_param)
{
       int ret = 0; 
       unsigned long user_addr; 
       struct vm_area_struct *vm_area;
       struct address addr;
       switch (ioctl_num) {
        case IOCTL_ALLOC_VMA:
        if(copy_from_user(&addr,(char*)ioctl_param,sizeof(struct address))){
                pr_err("alloc vma address error\n");
                return ret;
        }
        user_addr = vm_mmap(file, 0, total,
                            PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_ANONYMOUS | MAP_PRIVATE, 0);
        if (user_addr >= (unsigned long)(TASK_SIZE)) {
                pr_warn("Failed to allocate user memory\n");
                goto err1;
        }
         
        vm_area = find_vma(current->mm, user_addr);
        if(NULL == vm_area)
        {
            goto err2;
        }
        addr.user_addr = user_addr;
        if(copy_to_user((char*)ioctl_param,&addr,sizeof(struct address))){
                pr_err("alloc vma address error\n");
                return ret;
        }
        return 0;
        case IOCTL_FREE_VMA:
        if(copy_from_user(&addr,(char*)ioctl_param,sizeof(struct address))){
                pr_err("alloc vma address error\n");
                return ret;
        }
        user_addr = addr.user_addr;
            vm_munmap(user_addr, total);
            return ret;
       }
       return ret;
err2:
       vm_munmap(user_addr, total);
err1:
       return -ENOMEM;
}
static const struct file_operations mmap_fops = {
	.owner = THIS_MODULE,
	.open = mmapfop_open,
	.release = mmapfop_close,
	.mmap = op_mmap,
        .unlocked_ioctl = device_ioctl,
};

static int my_dev_uevent(struct device *dev, struct kobj_uevent_env *env)
{
	add_uevent_var(env, "DEVMODE=%#o", 0666);
	return 0;
}

static int __init mmapexample_module_init(void)
{
	int result;

	printk(KERN_DEBUG "sample char device init\n");

	result = alloc_chrdev_region(&sample_dev_t, 0, 1, "sample-cdev");
	if (result < 0) {
		printk(KERN_ERR "failed to alloc chrdev region\n");
		goto fail_alloc_chrdev_region;
	}
	sample_cdev = cdev_alloc();
	if (!sample_cdev) {
		result = -ENOMEM;
		printk(KERN_ERR "failed to alloc cdev\n");
		goto fail_alloc_cdev;
	}
	cdev_init(sample_cdev, &mmap_fops);
	result = cdev_add(sample_cdev, sample_dev_t, 1);
	if (result < 0) {
		printk(KERN_ERR "failed to add cdev\n");
		goto fail_add_cdev;
	}
	sample_class = class_create(THIS_MODULE, DEVICE_CLASS);
	if (!sample_class) {
		result = -EEXIST;
		printk(KERN_ERR "failed to create class\n");
		goto fail_create_class;
	}
	sample_class->dev_uevent = my_dev_uevent;
	if (!device_create(sample_class, NULL, sample_dev_t, NULL, DEVICE_NAME)) {
		result = -EINVAL;
		printk(KERN_ERR "failed to create device\n");
		goto fail_create_device;
	}
	mutex_init(&mmap_device_mutex);
	printk(KERN_INFO "mmap-example: %s registered with major %d\n",
	       DEVICE_NAME, MAJOR(sample_dev_t));

	return 0;

 fail_create_device:
	class_destroy(sample_class);
 fail_create_class:
	cdev_del(sample_cdev);
 fail_add_cdev:
 fail_alloc_cdev:
	unregister_chrdev_region(sample_dev_t, 1);
 fail_alloc_chrdev_region:
	return result;

	return 0;
}

static void __exit mmapexample_module_exit(void)
{
	printk(KERN_INFO "mmap-example: Module exit correctly\n");
	device_destroy(sample_class, sample_dev_t);
	class_destroy(sample_class);
	cdev_del(sample_cdev);
	unregister_chrdev_region(sample_dev_t, 1);
}

module_init(mmapexample_module_init);
module_exit(mmapexample_module_exit);
MODULE_LICENSE("GPL");
