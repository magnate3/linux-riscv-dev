/*
Usage:

	insmod /workqueue.ko
	# dmesg => worker
	rmmod workqueue

Creates a separate thread. So init_module can return, but some work will still get done.

Can't call this just workqueue.c because there is already a built-in with that name:
https://unix.stackexchange.com/questions/364956/how-can-insmod-fail-with-kernel-module-is-already-loaded-even-is-lsmod-does-not
*/

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/workqueue.h>

MODULE_LICENSE("GPL");

static struct workqueue_struct *queue;

static void work_func(struct work_struct *work)
{
	printk(KERN_INFO "worker\n");
}

DECLARE_WORK(work, work_func);

int init_module(void)
{
	queue = create_singlethread_workqueue("myworkqueue");
	queue_work(queue, &work);
	return 0;
}

void cleanup_module(void)
{
	/* TODO why is this needed? Why flush_workqueue doesn't work? (re-insmod panics)
	 * http://stackoverflow.com/questions/37216038/whats-the-difference-between-flush-delayed-work-and-cancel-delayed-work-sync */
	/*flush_workqueue(queue);*/
	cancel_work_sync(&work);
	destroy_workqueue(queue);
}