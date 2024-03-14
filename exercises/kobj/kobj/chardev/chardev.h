/*
 * chardev.h
 *
 *  Created on: Sep 29, 2021
 *      Author: Sylwester Dziedziuch
 */

#ifndef _CHARDEV_H_
#define _CHARDEV_H_

#include <linux/types.h>
#include <linux/cdev.h>

#define MAX_BUFF_SIZE 256
#define MIN_NUM_PROC 1
#define MAX_NUM_PROC 10
#define DEF_MINOR 0
#define DEV_NAME "exercise_char_dev"

struct chardev {
   struct cdev cdev;
   struct class *class;
   struct device *dev;
   struct mutex buffer_lock;
   struct mutex num_proc_lock;
   int minor;

   struct {
      u8 max_num_proc:4;
      u8 curr_num_proc:4;
   };

   char buffer[MAX_BUFF_SIZE];
};

extern u16 chardev_major;

int chardev_sysfs_init(struct chardev *device);
void chardev_sysfs_cleanup(void);

void chardev_dbg_init(void);
void chardev_dbg_exit(void);

#endif /* _CHARDEV_H_ */
