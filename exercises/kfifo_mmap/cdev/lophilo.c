// hrtimers: http://www.ibm.com/developerworks/linux/library/l-timers-list/index.htm
#include "lophilo.h"

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/hrtimer.h> // hrtimer_init, hrtimer_start
#include <linux/jiffies.h> // jiffies_to_usecs
#include <asm/uaccess.h>
#include <linux/slab.h>
#include <linux/kfifo.h> // kfifo_put, kfifo_to_user, etc...
#include <linux/wait.h> // wake_up_interruptible
#include <linux/sched.h> // TASK_INTERRUPTIBLE
#include <linux/poll.h> // poll_table

#define MAP_MAJOR 126
#define MS_TO_NS(x)	(x * 1E6L)

unsigned long DELAY_IN_NS = MS_TO_NS(10L);
 
static int map_lophilo(struct file *filp, struct vm_area_struct *vma);

static int read_lophilo(struct file *filp, char __user *buf, size_t count, loff_t *offset);

static unsigned int poll_lophilo(struct file *filp, poll_table *wait);
 
struct file_operations map_fops =
{
  .owner   = THIS_MODULE,
  .open    = nonseekable_open,
  .mmap    = map_lophilo,
  .read    = read_lophilo,
  .llseek  = noop_llseek,
  .poll    = poll_lophilo,
};

static struct hrtimer hr_timer;
static struct lophilo_data* loldata;

static DEFINE_KFIFO(updates, lophilo_update_t, LOPHILO_FIFO_SIZE);
wait_queue_head_t fifo_wq; 

static int dir = 1;

enum hrtimer_restart my_timer_callback(struct hrtimer *timer)
{
	struct timeval tv;
	if(loldata) {
		loldata->updates[LOPHILO_PIN_XA0].value += dir;
		if(loldata->updates[LOPHILO_PIN_XA0].value > 359) {
			dir = -1;
		}
		if(loldata->updates[LOPHILO_PIN_XA0].value == 0) {
			dir = 1;
		}
		
		do_gettimeofday(&tv);
		//printk("tv_sec: %ld tv_usec: %ld", tv.tv_sec, tv.tv_usec);
		loldata->updates[LOPHILO_LAST_UPDATE_SEC].value = tv.tv_sec;
		loldata->updates[LOPHILO_LAST_UPDATE_USEC].value = tv.tv_usec;
		kfifo_put(&updates, (loldata->updates[LOPHILO_LAST_UPDATE_SEC]));
		kfifo_put(&updates, (loldata->updates[LOPHILO_LAST_UPDATE_USEC]));
		kfifo_put(&updates, (loldata->updates[LOPHILO_PIN_XA0]));
		wake_up_interruptible(&fifo_wq);
	}
	hrtimer_forward(timer, hrtimer_get_expires(timer), ns_to_ktime(DELAY_IN_NS));
	return HRTIMER_RESTART;
} 

static int
map_lophilo(struct file *filp, struct vm_area_struct *vma)
{
	long unsigned int size = vma->vm_end - vma->vm_start;
	long unsigned int target_size = PAGE_SIZE;//sizeof(struct lophilo_data);
	
	if(size != target_size) {
		printk(KERN_INFO "Invalid allocation request, expected %ld, got %ld", target_size, size);
                return -EAGAIN;
	}
	if (remap_pfn_range(vma, vma->vm_start, __pa(loldata) >> PAGE_SHIFT,
	                    size, vma->vm_page_prot)) {
		printk(KERN_INFO "Allocation failed!");
                return -EAGAIN;
	}
	return 0;
}

static unsigned int poll_lophilo(struct file *filp, poll_table *wait)
{
	unsigned int mask = 0;
	poll_wait(filp, &fifo_wq, wait);
	if(kfifo_len(&updates)) {
		mask |= POLLIN | POLLRDNORM;
	}
	mask |= POLLOUT | POLLWRNORM;
	return  mask;

}

static int read_lophilo(struct file *filp, char __user *buf, size_t count, loff_t *offset) 
{
	int ret;
	unsigned int copied;
	if(!kfifo_len(&updates)) {
		if(filp->f_flags & O_NONBLOCK) 
			return -EAGAIN;
	        /* Blocking read.  Block if no data available */
	        else {
			wait_event_interruptible(fifo_wq,  kfifo_len(&updates));
		}
	}
	ret = kfifo_to_user(&updates, buf, count, &copied);
	if(ret < 0)
		printk("Error reading fifo %d\n", ret);
	return ret ? ret : copied;
}
 
static int __init 
lophilo_init(void)
{
	ktime_t ktime;

	// set datastore
	loldata = (struct lophilo_data*) kmalloc(PAGE_SIZE, GFP_KERNEL);
	if(!loldata) {
		printk("unable to allocate with kmalloc");
		return -EBUSY;
	}
	loldata->updates[LOPHILO_LAST_UPDATE_SEC].source = LOPHILO_LAST_UPDATE_SEC; 
	loldata->updates[LOPHILO_LAST_UPDATE_SEC].value = 0;
	loldata->updates[LOPHILO_LAST_UPDATE_USEC].source = LOPHILO_LAST_UPDATE_USEC; 
	loldata->updates[LOPHILO_LAST_UPDATE_USEC].value = 0;
	loldata->updates[LOPHILO_PIN_XA0].source = LOPHILO_PIN_XA0; 
	loldata->updates[LOPHILO_PIN_XA0].value = 0;

	// init fifo queue
	init_waitqueue_head(&(fifo_wq));
	INIT_KFIFO(updates);

	// register device
	if (register_chrdev(MAP_MAJOR,"map", &map_fops) <0 ) 
	{
		printk("unable to get major for map module\n");
		return -EBUSY;
	}

	// set timer
	ktime = ktime_set( 0, DELAY_IN_NS);
	hrtimer_init( &hr_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL );
	hr_timer.function = &my_timer_callback;
	printk( "Starting timer to fire in %ld ms (%ld)\n", DELAY_IN_NS, jiffies );
	hrtimer_start( &hr_timer, ktime, HRTIMER_MODE_REL );

	return 0;
}
 
void __exit
lophilo_cleanup(void)
{
	int ret;

	printk("Timer module uninstalling\n");
	ret = hrtimer_cancel( &hr_timer );
	if (ret) printk("The timer was still in use...\n");

	unregister_chrdev(MAP_MAJOR, "map");

	kfree(loldata);

	return;
}
 
MODULE_LICENSE("GPL");
 
module_init(lophilo_init);
module_exit(lophilo_cleanup);
