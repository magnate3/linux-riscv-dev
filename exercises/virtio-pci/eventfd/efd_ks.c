#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pid.h>
#include <linux/sched.h>
#include <linux/fdtable.h>
#include <linux/rcupdate.h>
#include <linux/eventfd.h>

/* Received from userspace. Process ID and eventfd's File descriptor are enough 
 * to uniquely identify an eventfd object.*/
int pid;
int efd;

//Resolved references...
struct task_struct * userspace_task = NULL; /*to userspace program's task struct*/
struct file * efd_file = NULL;          /*to eventfd's file struct */
struct eventfd_ctx * efd_ctx = NULL;    /* and finally to eventfd context */

/* Increment Counter by 1*/
static uint64_t plus_one = 1;

int init_module(void) {
    printk(KERN_ALERT "~~~Received from userspace: pid=%d efd=%d\n",pid,efd);

    userspace_task = pid_task(find_vpid(pid), PIDTYPE_PID);
    printk(KERN_ALERT "~~~Resolved pointer to the userspace program's task struct: %p\n",userspace_task);

    printk(KERN_ALERT "~~~Resolved pointer to the userspace program's files struct: %p\n",userspace_task->files);

    rcu_read_lock();
    efd_file = fcheck_files(userspace_task->files, efd);
    rcu_read_unlock();
    printk(KERN_ALERT "~~~Resolved pointer to the userspace program's eventfd's file struct: %p\n",efd_file);


    efd_ctx = eventfd_ctx_fileget(efd_file);
    if (!efd_ctx) {
        printk(KERN_ALERT "~~~eventfd_ctx_fileget() Jhol, Bye.\n");
        return -1;
    }
    printk(KERN_ALERT "~~~Resolved pointer to the userspace program's eventfd's context: %p\n",efd_ctx);

    eventfd_signal(efd_ctx, plus_one);

    printk(KERN_ALERT "~~~Incremented userspace program's eventfd's counter by 1\n");

    eventfd_ctx_put(efd_ctx);

    return 0;
}


void cleanup_module(void) {
    printk(KERN_ALERT "~~~Module Exiting...\n");
}  

MODULE_LICENSE("GPL");
module_param(pid, int, 0);
module_param(efd, int, 0);
