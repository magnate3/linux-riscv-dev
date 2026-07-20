#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/rcupdate.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/atomic.h>
#define  TEST_MOD_REF 1

struct foo {
	int a;
	struct rcu_head rcu;
};

static struct foo *g_ptr;
atomic_t  ref ;
static int myrcu_reader_thread1(void *data) //读者线程1
{
	struct foo *p1 = NULL;

	while (1) {
		if(kthread_should_stop())
			break;
		msleep(10);
		rcu_read_lock();
		p1 = rcu_dereference(g_ptr);
		if (p1)
			printk("%s: read a=%d\n", __func__, p1->a);
		rcu_read_unlock();
	}

	return 0;
}

static int myrcu_reader_thread2(void *data) //读者线程2
{
	struct foo *p2 = NULL;

	while (1) {
		if(kthread_should_stop())
			break;
		msleep(100);
		rcu_read_lock();
		mdelay(1500);
		p2 = rcu_dereference(g_ptr);
		if (p2)
			printk("%s: read a=%d\n", __func__, p2->a);
		
		rcu_read_unlock();
	}

	return 0;
}

static void myrcu_del(struct rcu_head *rh)
{
	struct foo *p = container_of(rh, struct foo, rcu);
	printk("%s: a=%d\n", __func__, p->a);
        ///module_put(THIS_MODULE);
        atomic_dec(&ref);
	kfree(p);
}

static int myrcu_writer_thread(void *p) //写者线程
{
	struct foo *old;
	struct foo *new_ptr;
	int value = (unsigned long)p;

	while (1) {
		if(kthread_should_stop())
			break;
		msleep(100);
		new_ptr = kmalloc(sizeof (struct foo), GFP_KERNEL);
		old = g_ptr;
		printk("%s: write to new %d\n", __func__, value);
		*new_ptr = *old;
		new_ptr->a = value;
		rcu_assign_pointer(g_ptr, new_ptr);
#if TEST_MOD_REF
                atomic_inc(&ref);
                //try_module_get(THIS_MODULE);
	        call_rcu(&old->rcu, myrcu_del); 
#else

               synchronize_rcu();
               kfree(old);
#endif
                
		value++;
	}
	return 0;
}     

static struct task_struct *reader_thread1;
static struct task_struct *reader_thread2;
static struct task_struct *writer_thread;
#if 0
static void rcu_wake_cond(struct task_struct *t, int status)
{
	/*
 * 	 * If the thread is yielding, only wake it when this
 * 	 	 * is invoked from idle
 * 	 	 	 */
	if (t && (status != RCU_KTHREAD_YIELDING || is_idle_task(current)))
		wake_up_process(t);
}
static void invoke_rcu_core_kthread(void)
{
	struct task_struct *t;
	unsigned long flags;

	local_irq_save(flags);
	//__this_cpu_write(rcu_data.rcu_cpu_has_work, 1);
	t = __this_cpu_read(rcu_data.rcu_cpu_kthread_task);
	if (t != NULL && t != current)
		rcu_wake_cond(t, __this_cpu_read(rcu_data.rcu_cpu_kthread_status));
	local_irq_restore(flags);
}
#endif
static int __init my_test_init(void)
{   
	int value = 5;
	printk("figo: my module init %d\n", get_current()->pid);
	g_ptr = kzalloc(sizeof (struct foo), GFP_KERNEL);

	reader_thread1 = kthread_run(myrcu_reader_thread1, NULL, "rcu_reader1");
	reader_thread2 = kthread_run(myrcu_reader_thread2, NULL, "rcu_reader2");
	writer_thread = kthread_run(myrcu_writer_thread, (void *)(unsigned long)value, "rcu_writer");

	return 0;
}
static void __exit my_test_exit(void)
{
	printk("goodbye\n");
	kthread_stop(reader_thread1);
	kthread_stop(reader_thread2);
	kthread_stop(writer_thread);
        while(atomic_read(&ref))
        {
             msleep(1000);
        }
#if 1
	if (g_ptr)
		kfree_rcu(g_ptr, rcu);
#endif
}
MODULE_LICENSE("GPL");
module_init(my_test_init);
module_exit(my_test_exit);
