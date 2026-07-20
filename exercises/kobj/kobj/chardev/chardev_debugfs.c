/*
 * chardev_debugfs.c
 *
 *  Created on: Sep 29, 2021
 *      Author: Sylwester Dziedziuch
 */

#include <linux/debugfs.h>

#include "chardev.h"

static struct dentry *chardev_dbg;

/**
 * chardev_dbg_init: chardev debugfs initialization function
 *
 * Function responsible for initializing chardev debugfs tree.
 * Creates exercise_debugfs and saves its dentry in chardev_dbg
 * for later use in cleanup function.
 * Excercise debugfs can be used to read chardevs major number.
 *
 */
void chardev_dbg_init(void) {
   debugfs_create_u16("exercise_debugfs", S_IRUSR | S_IRGRP,
                      NULL,(u16 *)&chardev_major);
   chardev_dbg = debugfs_lookup("exercise_debugfs", NULL);
}

/**
 * chardev_dbg_exit: chardev debugfs cleanup function
 *
 * Removes chardev debugfs tree.
 *
 */
void chardev_dbg_exit(void) {
   debugfs_remove(chardev_dbg);
}
