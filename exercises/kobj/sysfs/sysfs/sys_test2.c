/*
 * file: cdev03.c
 *
 * Desc: An example of sysfs and character device buffer
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <asm/uaccess.h>
#include <linux/device.h>
#include <linux/sysfs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "cdev03"
#define CLASS_NAME "gko_buffer"

/*
 * Prototypes 
 */
int init_module(void);
void cleanup_module(void);
static int dev_open(struct inode *inode, struct file *);
static int dev_release(struct inode *inode, struct file *);
static ssize_t dev_read(struct file *fp, char *buf, size_t len, loff_t *off);
static ssize_t dev_write(struct file *, const char *buf, size_t len, 
                         loff_t *off);

/*
 * variables
 */
static struct cdev *gko_cdev;    
static struct device *gko_device;
static struct class *gko_class;
static dev_t gko_dev;
static char *gko_buffer;           /* dynamic allocated */
static int gko_buffer_end = -1;
static atomic_t gko_buffer_start; /* may be changed by sysfs attr */


static struct file_operations fops = {
        .owner = THIS_MODULE,
        .read = dev_read,
        .write = dev_write,
        .open = dev_open,
        .release = dev_release
};


/*
 * Called when device is opened
 */
static int dev_open(struct inode *inode, struct file *fp)
{
        return 0;
}

/* 
 * Called when device is released. The device is
 * released when there is no process using it.
 */
static int dev_release(struct inode *inode, struct file *fp)
{
        return 0;
}


/*
 * always read the whole buffer
 */
static ssize_t dev_read(struct file *fp, char *buf, size_t len, loff_t *off)
{
        unsigned long rval;
        size_t copied;
        
        if (len > (gko_buffer_end - *off))
                len = gko_buffer_end - *off;
        
        rval = copy_to_user(buf, 
                            gko_buffer + *off,
                            len);
        
        if (rval < 0)
                return -EFAULT;

        copied = len - rval;
        *off += copied;

        return copied;
}

/*
 * start writing from gko_buffer_start
 */
static ssize_t dev_write(struct file *fp, const char *buf, size_t len, 
                         loff_t *off)
{
        unsigned long rval;
        size_t copied;
       
        printk(KERN_DEBUG DEVICE_NAME 
               " dev_write(fp, buf, len = %zu, off = %d\n", len, (int)*off);


        if (len > gko_buffer_end - *off)
                len = gko_buffer_end - *off;

        rval = copy_from_user(gko_buffer + atomic_read(&gko_buffer_start) + *off, 
                              buf,
                              len);

        if (rval < 0) {
                printk(KERN_DEBUG DEVICE_NAME " copy_from_user() failed\n");
                return -EFAULT;
        }

        copied = len - rval;
        *off += copied;

        return copied;
}


                              
static ssize_t buffer_start_show(struct device *dev,
                                 struct device_attribute *attr,
                                 char *buf)
{
        return snprintf(buf, PAGE_SIZE, "%d\n", atomic_read(&gko_buffer_start));
}

static ssize_t buffer_end_show(struct device *dev,
                               struct device_attribute *attr,
                               char *buf)
{
        return snprintf(buf, PAGE_SIZE, "%d\n", gko_buffer_end);
}

static ssize_t buffer_start_store(struct device *dev,
                                  struct device_attribute *attr,
                                  const char *buf,
                                  size_t count)
{
        int tmp;
        sscanf(buf, "%d", &tmp);
        if (tmp < 0 || gko_buffer_end < 0)
                tmp = 0;
        else if (tmp > gko_buffer_end)
                tmp = gko_buffer_end - 1;
                        
        atomic_set(&gko_buffer_start, tmp);
        return PAGE_SIZE;
}

static DEVICE_ATTR(buffer_start, S_IRUSR | S_IWUSR, buffer_start_show, buffer_start_store);
static DEVICE_ATTR(buffer_end, S_IRUSR, buffer_end_show, NULL);        

/*
 * Called when module is load
 */
static int __init test_init_module(void)
{
        int rval;

        printk(KERN_INFO DEVICE_NAME " init_module");

/* Alloc buffer */
        gko_buffer = kmalloc(PAGE_SIZE, GFP_KERNEL);
        if (!gko_buffer)
                return -ENOMEM;
        gko_buffer_end = PAGE_SIZE;

/* Alloc a device region */
        rval = alloc_chrdev_region(&gko_dev, 1, 1, DEVICE_NAME);
        if (rval != 0)          /* error */
                goto cdev_alloc_err;

/* Registring */
        gko_cdev = cdev_alloc();
        if (!gko_cdev) 
                goto cdev_alloc_err;

/* Init it! */
        cdev_init(gko_cdev, &fops); 

/* Tell the kernel "hey, I'm exist" */
        rval = cdev_add(gko_cdev, gko_dev, 1);
        if (rval < 0) 
                goto cdev_add_out;

/* class */
        gko_class = class_create(THIS_MODULE, CLASS_NAME);
        if (IS_ERR(gko_class)) {
                printk(KERN_ERR DEVICE_NAME " cant create class %s\n", CLASS_NAME);
                goto class_err;
        }

/* device */
        gko_device = device_create(gko_class, NULL, gko_dev, NULL, DEVICE_NAME);
        if (IS_ERR(gko_device)) {
                printk(KERN_ERR DEVICE_NAME " cant create device %s\n", DEVICE_NAME);
                goto device_err;
        }

/* device attribute on sysfs */
        rval = device_create_file(gko_device, &dev_attr_buffer_start);
        if (rval < 0) {
                printk(KERN_ERR DEVICE_NAME " cant create device attribute %s %s\n", 
                       DEVICE_NAME, dev_attr_buffer_start.attr.name);
        }

        rval = device_create_file(gko_device, &dev_attr_buffer_end);
        if (rval < 0) {
                printk(KERN_ERR DEVICE_NAME " cant create device attribute %s %s\n", 
                       DEVICE_NAME, dev_attr_buffer_start.attr.name);
        }
                
        return 0;


device_err:
        device_destroy(gko_class, gko_dev);
class_err:
        class_unregister(gko_class);
        class_destroy(gko_class);
cdev_add_out:
        cdev_del(gko_cdev);
cdev_alloc_err:
        kfree(gko_buffer);
        return -EFAULT;
}

static void __exit test_cleanup_module(void)
{
        device_destroy(gko_class, gko_dev);
        class_unregister(gko_class);
        class_destroy(gko_class);
        cdev_del(gko_cdev);
        kfree(gko_buffer);
}

module_init(test_init_module);
module_exit(test_cleanup_module);

MODULE_LICENSE("GPL");
