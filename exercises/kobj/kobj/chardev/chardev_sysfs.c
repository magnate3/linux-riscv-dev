/*
 * chardev_sysfs.c
 *
 *  Created on: Sep 30, 2021
 *      Author: Sylwester Dziedziuch
 */

#include <linux/kobject.h>

#include "chardev.h"

#define to_dev(obj) container_of(obj, struct device, kobj)

static struct kobject *num_proc_kobj;

/**
 * num_proc_show: get the value of max_num_proc
 *
 * Sends the current value of max_num_proc to user.
 *
 * @param kobj kobject structure for max_num_proc attribute
 * @param attr pointer to attribute structure for max_num_proc
 * @param buf buffer to save the value to
 *
 * @return number of bytes read
 */
static ssize_t num_proc_show(struct kobject *kobj,
                             struct kobj_attribute *attr, char *buf)
{
   struct device *dev = to_dev(kobj->parent);
   struct chardev *device = dev_get_drvdata(dev);

   return sprintf(buf, "%hu\n", device->max_num_proc);
}

/**
 * num_proc_store: stores the value of max_num_proc
 *
 * Stores the max_num_proc value. The value can be set in <1,10> range.
 *
 * @param kobj kobject structure for max_num_proc attribute
 * @param attr pointer to attribute structure for max_num_proc
 * @param buf buffer containing the request
 * @param count number of bytes
 *
 * @return count on success -EINVAL if value to set is out of range or invalid
 */
static ssize_t num_proc_store(struct kobject *kobj,
                              struct kobj_attribute *attr,
                              const char *buf, size_t count)
{
   struct device *dev = to_dev(kobj->parent);
   struct chardev *device = dev_get_drvdata(dev);
   u8 num_proc;

   sscanf(buf, "%hhu", &num_proc);

   if (num_proc < MIN_NUM_PROC || num_proc > MAX_NUM_PROC) {
      printk(KERN_WARNING "Value %hu out of range\n", num_proc);
      return -EINVAL;
   }

   device->max_num_proc = num_proc;

   return count;
}

static struct kobj_attribute max_num_proc_attr =
   __ATTR(max_num_proc, S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP,
          num_proc_show, num_proc_store);

/**
 * chardev_sysfs_init: initialize chardev sysfs
 *
 * Function responsible for initializing chardevs sysfs tree.
 * It creates one folder exercise_sysfs containing one configurable argument: max_num_proc.
 *
 * @param device: pointer to chardevs device struct
 *
 * @return returns 0 on success or -EFAULT if kobject creation for exercise_sysfs failed
 */
int chardev_sysfs_init(struct chardev *device) {
   int result = 0;

   num_proc_kobj = kobject_create_and_add("exercise_sysfs", &device->dev->kobj);
   if (!num_proc_kobj) {
      printk(KERN_WARNING "Sysfs kobj create failed\n");
      return -EFAULT;
   }

   result = sysfs_create_file(num_proc_kobj, &max_num_proc_attr.attr);
   if (result) {
      printk(KERN_WARNING "Sysfs attribute create failed\n");
      kobject_put(num_proc_kobj);
   }

   return result;
}

/**
 * chardev_sysfs_cleanup: Cleanup chardev sysfs
 */
void chardev_sysfs_cleanup(void) {
   if (num_proc_kobj)
      kobject_put(num_proc_kobj);
}
