/*
 * chardev_main.c
 *
 *  Created on: Sep 28, 2021
 *      Author: Sylwester Dziedziuch
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <asm/uaccess.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/uaccess.h>

#include "chardev.h"

u16 chardev_major = 0;

static struct chardev *device = NULL;

/**
 * dev_open: function handling device open requests
 *
 * Function increments curr_num_proc for every process
 * that has opened the device. Device can be opened only if
 * number of processes that have the device open is smaller
 * than max_num_proc.
 *
 * @param inode inode of device file
 * @param file file struct corresponding to the device
 *
 * @return 0 on success -EBUSY if number of processes that
 * have the device currently open is greater or equal max_num_proc
 * -ERESTARTSYS if waiting for mutex was interrupted
 */
static int dev_open(struct inode *inode, struct file *file)
{
   struct chardev *data =
      container_of(inode->i_cdev, struct chardev, cdev);

   if (device->max_num_proc <= data->curr_num_proc)
      return -EBUSY;

   if (mutex_lock_interruptible(&data->num_proc_lock))
      return -ERESTARTSYS;

   data->curr_num_proc++;

   mutex_unlock(&data->num_proc_lock);

   file->private_data = data;

   return 0;
}

/**
 * dev_release: Function responsible for handling device close requests
 *
 * @param inode inode of device file
 * @param file file struct corresponding to the device
 *
 * @return 0 or -ERESTARTSYS if waiting for mutex was interrupted
 */
int dev_release(struct inode *inode, struct file *filp)
{
   struct chardev *data =
      container_of(inode->i_cdev, struct chardev, cdev);

   if (mutex_lock_interruptible(&data->num_proc_lock))
      return -ERESTARTSYS;

   data->curr_num_proc--;

   mutex_unlock(&data->num_proc_lock);

   return 0;
}

/**
 * dev_read: Function responsible for handling read requests
 *
 * Reads data from module buffer and saves it in user buffer
 *
 * @param file pointer to device file struct
 * @param user_buffer user buffer to save data to
 * @param size how much to read from module buffer
 * @param offset offset to module buffer
 *
 * @return returns number of bytes read
 */
static ssize_t dev_read(struct file *file, char __user *user_buffer,
                        size_t size, loff_t *offset)
{
   struct chardev *data = (struct chardev *) file->private_data;
   ssize_t len = min((size_t)(MAX_BUFF_SIZE - *offset), size);
   int result = 0;

   if (len <= 0)
      return 0;

   if (mutex_lock_interruptible(&data->buffer_lock))
         return -ERESTARTSYS;

   result = copy_to_user(user_buffer, data->buffer + *offset, len);

   mutex_unlock(&data->buffer_lock);

   *offset += len - result;

   return len - result;
}

/**
 * dev_write: Function responsible for handling write requests
 *
 * Reads data from user buffer and saves it in module buffer
 *
 * @param file pointer to device file struct
 * @param user_buffer buffer to read data from
 * @param size size of the data to write
 * @param offset offset to start writing to module buffer from
 *
 * @return number of bytes written
 */
static ssize_t dev_write(struct file *file, const char __user *user_buffer,
                         size_t size, loff_t * offset)
{
   struct chardev *data = (struct chardev *) file->private_data;
   ssize_t len = min((size_t)(MAX_BUFF_SIZE - *offset), size);
   int result = 0;

   if (len <= 0)
      return 0;

   if (mutex_lock_interruptible(&data->buffer_lock))
         return -ERESTARTSYS;

   result = copy_from_user(data->buffer + *offset, user_buffer, len);

   mutex_unlock(&data->buffer_lock);

   *offset += len - result;

   return len - result;
}

const struct file_operations chardev_fops = {
   .owner = THIS_MODULE,
   .open = dev_open,
   .read = dev_read,
   .write = dev_write,
   .release = dev_release
};

/**
 * dev_init: module initialization function
 *
 * Function responsible for initializing all the module structs on load
 *
 * @return 0 on success errno on failure
 */
static int __init dev_init(void)
{
   int result = 0;
   dev_t dev = 0;

   if (!(device = kzalloc(sizeof(*device), GFP_KERNEL))) {
      printk(KERN_WARNING "Memory allocation for chardev failed\n");
      return -ENOMEM;
   }

   result = alloc_chrdev_region(&dev, DEF_MINOR, 1, DEV_NAME);
   if (result) {
      printk(KERN_WARNING "Cannot allocate chrdev\n");
      goto err_alloc;
   }

   chardev_major = MAJOR(dev);
   device->minor = MINOR(dev);
   device->max_num_proc = MIN_NUM_PROC;
   device->curr_num_proc = 0;

   cdev_init(&device->cdev, &chardev_fops);
   device->cdev.owner = THIS_MODULE;
   device->cdev.ops = &chardev_fops;

   result = cdev_add(&device->cdev, dev, 1);
   if (result) {
      printk(KERN_WARNING "Unable to add cdev\n");
      goto err_add;
   }

   if (!(device->class = class_create(THIS_MODULE, "exercise"))) {
      printk(KERN_WARNING "Class create failed\n");
      goto err_class;
   }

   if (!(device->dev = device_create(device->class, NULL, dev,
                                     NULL, DEV_NAME))) {
      printk(KERN_WARNING "Device create failed\n");
      goto err_dev;
   }

   dev_set_drvdata(device->dev, device);

   result = chardev_sysfs_init(device);
   if (result)
      goto err_sysfs;

   mutex_init(&device->buffer_lock);
   mutex_init(&device->num_proc_lock);
   chardev_dbg_init();

   return 0;

err_sysfs:
   device_destroy(device->class, dev);
err_dev:
   class_destroy(device->class);
err_class:
   cdev_del(&device->cdev);
err_add:
   unregister_chrdev_region(dev, 1);
err_alloc:
   kfree(device);
   return result;
}

/**
 * dev_cleanup: module cleanup function
 *
 * Function responsible for cleaning up the module called on module exit
 */
static void __exit dev_cleanup(void)
{
   chardev_sysfs_cleanup();
   chardev_dbg_exit();
   mutex_destroy(&device->num_proc_lock);
   mutex_destroy(&device->buffer_lock);
   device_destroy(device->class, device->cdev.dev);
   class_destroy(device->class);
   cdev_del(&device->cdev);
   unregister_chrdev_region(MKDEV(chardev_major, device->minor), 1);
   kfree(device);
}

module_init(dev_init);
module_exit(dev_cleanup);
MODULE_AUTHOR("Sylwester Dziedziuch");
MODULE_DESCRIPTION("Excercise char device");
MODULE_LICENSE("GPL");
