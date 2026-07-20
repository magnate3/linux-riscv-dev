#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/uaccess.h>

#include "ioctl_test.h"

// Meta Information
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Takehiro");
MODULE_DESCRIPTION("Example for ioctl in Linux kernel module");

// Called when the device file is opened
static int driver_open(struct inode *device_file, struct file *instance) {
  printk("ioctl_example open was called\n");
  return 0;
}

// Called when the device file is closed
static int driver_close(struct inode *device_file, struct file *instance) {
  printk("ioctl_example close was called\n");
  return 0;
}

int32_t answer = 123;

static long int my_ioctl(struct file *f, unsigned cmd, unsigned long arg) {
  struct my_struct ms;

  switch (cmd) {
    case WR_VALUE:
      if (copy_from_user(&answer, (int32_t *)arg, sizeof(answer))) {
        printk("ioctl_example - failed to copy data from user\n");
      }
      printk("ioctl_example - updated to the answer: %d\n", answer);
      break;
    case RD_VALUE:
      if (copy_to_user((int32_t *)arg, &answer, sizeof(answer))) {
        printk("ioctl_example - failed to copy data to user\n");
      }
      printk("ioctl_example - the answer copied: %d\n", answer);
      break;
    case GREETER:
      if (copy_from_user(&ms, (struct my_struct *)arg, sizeof(ms))) {
        printk("ioctl_example - failed to copy data from user\n");
      }
      printk("ioctl_example - %d greets to %s\n", ms.repeat, ms.name);
      break;
  }

  return 0;
}

static struct file_operations fops = {
  .owner = THIS_MODULE,
  .open = driver_open,
  .release = driver_close,
  .unlocked_ioctl = my_ioctl
};

// Update device number if required.
// Check available device number by command below:
//   $ cat /proc/devices 
#define MY_MAJOR_NUMBER 90

// Called when the module is loaded into the kernel
static int __init ModuleInit(void) {
  int ret;

  printk("Hello, Linux kernel!\n");

  // register device number
  ret = register_chrdev(MY_MAJOR_NUMBER, "ioctl_example", &fops);
  if (ret == 0) {
    printk("ioctl_example - registered Device numver Major: %d, Minor: %d\n",
           MY_MAJOR_NUMBER, 0);
  } else if (ret > 0) { // The major number is already in use.
                        // It registers same major number and incremented minor number.
    printk("ioctl_example - registered Device numver Major: %d, Minor: %d\n",
           ret>>20, ret&0xfffff); // ret contains Major device number in high-order 12 bit
                                  // and Minor device number in lower 20 bit
  } else {
    printk("Could not register device number\n");
    return -1;
  }

  return 0;
}

// Called when the module is removed from the kernel
static void __exit ModuleExit(void) {
  unregister_chrdev(MY_MAJOR_NUMBER, "ioctl_example");
  printk("Goodbye, Linux kernel!\n");
}

module_init(ModuleInit);
module_exit(ModuleExit);
