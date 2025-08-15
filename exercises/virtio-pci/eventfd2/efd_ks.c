#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pid.h>
#include <linux/sched.h>
#include <linux/fdtable.h>
#include <linux/rcupdate.h>
#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/eventfd.h>
#include <linux/vfio.h>
#include <linux/eventfd.h>
#include <linux/file.h>
/* Received from userspace. Process ID and eventfd's File descriptor are enough 
 * to uniquely identify an eventfd object.*/
int times = 0;
int pid;
int efd;

//Resolved references...
struct task_struct * userspace_task = NULL; /*to userspace program's task struct*/
struct file * efd_file = NULL;          /*to eventfd's file struct */
struct eventfd_ctx * efd_ctx = NULL;    /* and finally to eventfd context */
static struct workqueue_struct *vfio_irqfd_cleanup_wq;
/* Increment Counter by 1*/
static uint64_t plus_one = 1;
static struct virqfd *virqfd = NULL;
static void virqfd_ptable_queue_proc(struct file *file,
				     wait_queue_head_t *wqh, poll_table *pt)
{
	struct virqfd *virqfd = container_of(pt, struct virqfd, pt);
	add_wait_queue(wqh, &virqfd->wait);
}
//static int p9_pollwake(wait_queue_entry_t *wait, unsigned int mode, int sync, void *key)
static int virqfd_wakeup( wait_queue_entry_t *wait, unsigned mode, int sync, void *key)
{
      struct virqfd *virqfd = container_of(wait, struct virqfd, wait);
      unsigned long flags = (unsigned long)key;
      if (flags & POLLIN) {
          pr_info("pollin event, %s times %d  \n", __func__, ++times);
      }
      return 0;
}
static int vfio_virqfd_init(void)
{
	vfio_irqfd_cleanup_wq =
	create_singlethread_workqueue("vfio-irqfd-cleanup");
	if (!vfio_irqfd_cleanup_wq)
		return -ENOMEM;

	return 0;
}
static void vfio_virqfd_exit(void)
{
	destroy_workqueue(vfio_irqfd_cleanup_wq);
}
int init_module(void) {
    unsigned int events;
    struct fd irqfd;
    printk(KERN_ALERT "~~~Received from userspace: pid=%d efd=%d\n",pid,efd);

    userspace_task = pid_task(find_vpid(pid), PIDTYPE_PID);
    printk(KERN_ALERT "~~~Resolved pointer to the userspace program's task struct: %p\n",userspace_task);

    printk(KERN_ALERT "~~~Resolved pointer to the userspace program's files struct: %p\n",userspace_task->files);
    vfio_virqfd_init();

    virqfd = kzalloc(sizeof(*virqfd), GFP_KERNEL);
    if (!virqfd)
    {
         pr_info("kzalloc virqfd fail \n");
		return -ENOMEM;
    }
    	/*
 * 	 * Install our own custom wake-up handling so we are notified via
 * 	 	 * a callback whenever someone signals the underlying eventfd.
 * 	 	 	 */
    init_waitqueue_func_entry(&virqfd->wait, virqfd_wakeup);
    init_poll_funcptr(&virqfd->pt, virqfd_ptable_queue_proc);
    virqfd->eventfd = NULL; 

    rcu_read_lock();
    efd_file = fcheck_files(userspace_task->files, efd);
    rcu_read_unlock();


    efd_ctx = eventfd_ctx_fileget(efd_file);
    if (!efd_ctx) {
        printk(KERN_ALERT "~~~eventfd_ctx_fileget() Jhol, Bye.\n");
        goto err1;
    }
    printk(KERN_ALERT "~~~Resolved pointer to the userspace program's eventfd's context: %p\n",efd_ctx);


    virqfd->eventfd = efd_ctx; 
    printk(KERN_ALERT "kernel begin poll \n");
    events = efd_file->f_op->poll(efd_file, &virqfd->pt);
    //eventfd_ctx_put(efd_ctx);
err1:
    //kfree(virqfd);
    //fdput(irqfd);
#if 0
err2:
    //kfree(virqfd);
#endif
    return 0;
}

static void virqfd_free(void)
{
       u64 cnt;
       if (NULL != virqfd)
       {
           if (NULL !=  virqfd->eventfd)
           {
               eventfd_ctx_remove_wait_queue(virqfd->eventfd, &virqfd->wait, &cnt);
               eventfd_ctx_put(virqfd->eventfd);
           }
           kfree(virqfd);
       }
}
void cleanup_module(void) {
    virqfd_free();
    vfio_virqfd_exit();
    printk(KERN_ALERT "~~~Module Exiting...\n");
}  

MODULE_LICENSE("GPL");
module_param(pid, int, 0);
module_param(efd, int, 0);
