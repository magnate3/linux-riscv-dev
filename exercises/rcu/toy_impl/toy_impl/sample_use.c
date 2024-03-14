#include <linux/module.h>
#include <linux/slab.h>
#include <linux/kthread.h>
#include <linux/random.h>
#include "toy_rcu.h"
#include "utils.h"

struct foo {
	int a;
	char b;
	long c;
};
DEFINE_SPINLOCK(foo_mutex);
/*
 * #define __rcu 	__attribute__((noderef, address_space(4)))
 */
struct foo __rcu *gbl_foo;

static int init_foo(void)
{
	gbl_foo = kmalloc(sizeof(*gbl_foo), GFP_KERNEL);
	if (!gbl_foo)
		return -1;

	gbl_foo->a = 5;

	return 0;
}

void foo_update_a(int new_a)
{
	struct foo *new_fp;
	struct foo *old_fp;

	//START_THREAD;

	new_fp = kmalloc(sizeof(*new_fp), GFP_KERNEL);

	spin_lock(&foo_mutex);
	old_fp = rcu_dereference(gbl_foo);
	*new_fp = *old_fp;
	new_fp->a = new_a;
	rcu_assign_pointer(gbl_foo, new_fp);
	spin_unlock(&foo_mutex);

	toy_synchronize_rcu();
	kfree(old_fp);

	//END_THREAD;
}

int foo_get_a(void)
{
	int retval;

	//START_THREAD;

	toy_rcu_read_lock();
	retval = rcu_dereference(gbl_foo)->a;
	toy_rcu_read_unlock();

	//END_THREAD;
	return retval;
}

#define NUM_WRITER_THREADS 1
#define NUM_READER_THREADS 2
#define NUM_THREADS (NUM_WRITER_THREADS + NUM_READER_THREADS)

static struct task_struct *k[NUM_THREADS];

static int kthread_reader(void *arg)
{
	void kthread_reader_main(void)
	{
		int seed;
		int val;

		seed = get_random_int() % 2 + 1;

		set_current_state(TASK_INTERRUPTIBLE);
		schedule_timeout(seed * HZ);

		//START_THREAD;

		val = foo_get_a();
		pr_info("READER-%d:%d(%ld)\n", current->pid, val, jiffies);

		//END_THREAD;
	}

	while (!kthread_should_stop())
		kthread_reader_main();

	return 0;
}


static int kthread_writer(void *arg)
{
	void kthread_writer_main(void)
	{
		int val;

		val = get_random_int() % 100;

		set_current_state(TASK_INTERRUPTIBLE);
		schedule_timeout(3 * HZ);

		//START_THREAD;

		//pr_info("WRITER-%d:%d(%ld)\n", current->pid, val, jiffies);
		foo_update_a(val);

		//END_THREAD;
	}
	while (!kthread_should_stop())
		kthread_writer_main();

	return 0;
}

static int init_kthread(void)
{
	int i;

	for (i = 0; i < NUM_READER_THREADS; i++) {
		k[i] = kthread_run(kthread_reader, NULL, "reader kthread");
		if (IS_ERR(k))
			return -1;

		pr_info("pid->%d:prio->%d:comm->%s\n",
			k[i]->pid,
			k[i]->static_prio,
			k[i]->comm);
	}

	for (; i < NUM_THREADS; i++) {
		k[i] = kthread_run(kthread_writer, NULL, "writer kthread");
		if (IS_ERR(k))
			return -1;

		pr_info("pid->%d:prio->%d:comm->%s\n",
			k[i]->pid,
			k[i]->static_prio,
			k[i]->comm);
	}

	return 0;
}

static int __init init_sample_toy_rcu(void)
{
	int err = 0;

	err = init_foo();
	if (err)
		return err;

	err = init_kthread();
	if (err)
		goto out;

	pr_info("------------------------\n");
	pr_info("--- RCU sample module start ---\n");

	return 0;

out:
	kfree(gbl_foo);
	return err;
}

static void __exit exit_sample_toy_rcu(void)
{
	int i;

	for (i = 0; i < NUM_THREADS; i++)
		kthread_stop(k[i]);

	kfree(gbl_foo);

	pr_info("--- RCU sample stop ---\n");
}

MODULE_AUTHOR("Fumiya Shigemitsu");
MODULE_DESCRIPTION("sample: Using TOY RCU");
MODULE_LICENSE("GPL");

module_init(init_sample_toy_rcu)
module_exit(exit_sample_toy_rcu)
