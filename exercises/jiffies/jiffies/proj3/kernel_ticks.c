/*
 * kernel_ticks.c: an exercise to start learning Linux Kernel programming.
 *
 * This code is part of the course "An introduction to Linux Kernel
 * programming", taught by Robert P. J. Day in http://crashcourse.ca/
 *
 * The program is an exercise related to the lesson 12 of the course. It
 * is a module for the Linux kernel which takes the 'jiffies' and the 'HZ'
 * values from the running kernel and shows it in a created /proc files
 * under its own directory /proc/ticks.
 *
 */

#include <linux/module.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/jiffies.h>

static struct proc_dir_entry *ticks_dir;

/* The jiffies */
static int jiffies_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%llu\n",
            (unsigned long long) get_jiffies_64());
    seq_printf(m, "jiffies=%lu &jiffies=%p (u64)jiffies=%llu (u64)jiffies_64=%llu &jiffies_64=%p get_jiffies_64()=%llu\n", jiffies, &jiffies, (u64)jiffies, (u64)jiffies_64, &jiffies_64, get_jiffies_64());
    return 0;
}

static int jiffies_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, jiffies_proc_show, NULL);
}

static const struct file_operations jiffies_proc_fops = {
    .owner      = THIS_MODULE,
    .open       = jiffies_proc_open,
    .read       = seq_read,
    .llseek     = seq_lseek,
    .release    = single_release,
};

/* The HZs */
static int hz_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%d\n", HZ);
    return 0;
}

static int hz_proc_open(struct inode *inode, struct file *file)
{
    return single_open(file, hz_proc_show, NULL);
}

static const struct file_operations hz_proc_fops = {
    .owner      = THIS_MODULE,
    .open       = hz_proc_open,
    .read       = seq_read,
    .llseek     = seq_lseek,
    .release    = single_release,
};

/* Entry routine */
static int __init ticks_proc_init(void)
{
    /* Creation of the directory under /proc */
    ticks_dir = proc_mkdir("ticks", NULL);
    if (!ticks_dir)     // error handling
        return -ENOMEM;

    printk(KERN_INFO "Loading ticks module.\n");
    proc_create("jiffies", 0, ticks_dir, &jiffies_proc_fops);
    proc_create("hzs", 0, ticks_dir, &hz_proc_fops);
    return 0;
}

/* Exit routine */
static void __exit ticks_proc_exit(void)
{
    /* Remove the created files and the directory */
    remove_proc_entry("jiffies", ticks_dir);
    remove_proc_entry("hzs", ticks_dir);
    remove_proc_entry("ticks", NULL);

    printk(KERN_INFO "Directory /proc/ticks removed.\n");
    printk(KERN_INFO "Unloading ticks module.\n");
}

module_init(ticks_proc_init);
module_exit(ticks_proc_exit);
