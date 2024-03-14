/* The sample of call_rcu
 * My updating thread cannot block - using call_rcu
 */
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/jiffies.h>
#include <linux/random.h>
#include "utils.h"

struct foo {
	int a;
	char b;
	long c;
	struct rcu_head rcu;
};
DEFINE_SPINLOCK(foo_mutex);

struct foo __rcu *gbl_foo;

static void foo_reclaim(struct rcu_head *p)
{
	struct foo *fp = container_of(p, struct foo, rcu);
	pr_info("RECLAIM!\n");
	kfree(fp);
}

/*
 * Create a new struct foo that is the same as the one currently
 * pointed to by gbl_foo, except that field "a" is replaced
 * with "new_a".  Points gbl_foo to the new structure, and
 * frees up the old structure after a grace period.
 *
 * Uses rcu_assign_pointer() to ensure that concurrent readers
 * see the initialized version of the new structure.
 *
 * Uses call_rcu() to ensure that any readers that might have
 * references to the old structure complete before freeing the
 * old structures.
 */
void foo_update_a(int new_a)
{
	struct foo *new_fp;
	struct foo *old_fp;

	//START_THREAD;

	new_fp = kmalloc(sizeof(*new_fp), GFP_KERNEL);

	spin_lock(&foo_mutex);
	old_fp = rcu_dereference_protected(gbl_foo, lockdep_is_held(&foo_mutex));
	*new_fp = *old_fp;
	new_fp->a = new_a;
	rcu_assign_pointer(gbl_foo, new_fp);
	spin_unlock(&foo_mutex);

	/*
	 * Marks the end of updater code and the beginning of reclaimer code.
	 * It does this by blocking until all pre-existing RCU read-side
	 * critical sections on all CPUs have completed.
	 * Note that synchronoize_rcu() will -not- necessarily wait for
	 * any subsequent RCU read-side critical sections to complete.
	 */

	/*
	 * If the callback for call_rcu() is not doing anything
	 * more than calling kfree() on the structure, we can
	 * kfree_rcu instead of call_rcu to avoid having
	 * to write our own callback.
	 *
	 * kfree_rcu(old_fp, rcu);
	 */
	call_rcu(&old_fp->rcu, foo_reclaim);

	//END_THREAD;
}

/*
 * Return the value of field "a" of the current gbl_foo
 * structure.  Use rcu_read_lock() and rcu_read_unlock()
 * to ensure that the structure does not get deleted out
 * from under us, and use rcu_dereference() to ensure that
 * we see the initialized version of the structure (important
 * for DEC Alpha and for people reading the code).
 */
int foo_get_a(void)
{
	int retval;

	//START_THREAD;

	rcu_read_lock();
	retval = rcu_dereference(gbl_foo)->a;
	rcu_read_unlock();

	//END_THREAD;
	return retval;
}

#define NUM_WRITER_THREADS 1
#define NUM_READER_THREADS 10
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

		START_THREAD;

		val = foo_get_a();
		pr_info("READER-%d:%d(%ld)\n", current->pid, val, jiffies);

		END_THREAD;
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

		START_THREAD;

		pr_info("WRITER-%d:%d(%ld)\n", current->pid, val, jiffies);
		foo_update_a(val);

		END_THREAD;
	}
	while (!kthread_should_stop())
		kthread_writer_main();

	return 0;
}

static int init_foo(void)
{
	gbl_foo = kmalloc(sizeof(*gbl_foo), GFP_KERNEL);
	if (!gbl_foo)
		return -1;

	gbl_foo->a = 5;

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

static int __init init_sample_(void)
{
	int err = 0;

	err = init_foo();
	if (err)
		return err;

	err = init_kthread();
	if (err)
		goto out;

	pr_info("------------------------\n");
	pr_info("--- RCU sample start ---\n");

	return 0;

out:
	kfree(gbl_foo);
	return err;
}

static void __exit exit_sample_(void)
{
	int i;

	for (i = 0; i < NUM_THREADS; i++)
		kthread_stop(k[i]);

	kfree(gbl_foo);


	pr_info("--- RCU sample stop %d---\n", i);
}

MODULE_AUTHOR("Fumiya Shigemitsu");
MODULE_DESCRIPTION("sample: Using RCU");
MODULE_LICENSE("GPL");

module_init(init_sample_)
module_exit(exit_sample_)
