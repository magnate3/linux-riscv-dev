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
static wait_queue_head_t queue_wr;
static wait_queue_head_t queue_r;
static atomic_t awake_wr  = ATOMIC_INIT(0);
static atomic_t awake_r = ATOMIC_INIT(0);
static int myrcu_reader_thread1(void *data) //读者线程1
{
	struct foo *p1 = NULL;
	rcu_read_lock();
	p1 = rcu_dereference(g_ptr);
	if (p1)
      	   printk("%s: read a=%d\n", __func__, p1->a);
	rcu_read_unlock();
        atomic_set(&awake_wr, 1);
	wake_up(&queue_wr);
	return 0;
}

static int myrcu_reader_thread2(void *data) //读者线程2
{
	struct foo *p2 = NULL;
	wait_event(queue_r, atomic_read(&awake_r));
	atomic_set(&awake_r, 0);
	rcu_read_lock();
	p2 = rcu_dereference(g_ptr);
	if (p2)
		printk("%s: read a=%d\n", __func__, p2->a);
	rcu_read_unlock();
	return 0;
}

static void myrcu_del(struct rcu_head *rh)
{
	struct foo *p = container_of(rh, struct foo, rcu);
	printk("%s: a=%d\n", __func__, p->a);
        ///module_put(THIS_MODULE);
        atomic_inc(&ref);
	kfree(p);
}

static int myrcu_writer_thread(void *p) //写者线程
{
	struct foo *old;
	struct foo *new_ptr;
	int value = (unsigned long)p;
		wait_event(queue_wr, atomic_read(&awake_wr));
		atomic_set(&awake_wr, 0);
		new_ptr = kmalloc(sizeof (struct foo), GFP_KERNEL);
		old = g_ptr;
		printk("%s: write to new %d\n", __func__, value);
                ///////////////////////////////////////////////////
		*new_ptr = *old; // is important
		new_ptr->a = value;
		rcu_assign_pointer(g_ptr, new_ptr);
                atomic_set(&awake_r, 1);
                wake_up(&queue_r);
                msleep(1000);
                pr_info(" writer wake up from sleep \n");
#if TEST_MOD_REF
                //atomic_inc(&ref);
                //try_module_get(THIS_MODULE);
	        call_rcu(&old->rcu, myrcu_del); 
#else

               synchronize_rcu();
               kfree(old);
#endif
                
		value++;
	return 0;
}     

static struct task_struct *reader_thread1;
static struct task_struct *reader_thread2;
static struct task_struct *writer_thread;
static int __init my_test_init(void)
{   
	int value = 5;
	printk("figo: my module init %d\n", get_current()->pid);
        init_waitqueue_head(&queue_wr);
        init_waitqueue_head(&queue_r);
	g_ptr = kzalloc(sizeof (struct foo), GFP_KERNEL);
        g_ptr->a = 99;
	reader_thread1 = kthread_run(myrcu_reader_thread1, NULL, "rcu_reader1");
	reader_thread2 = kthread_run(myrcu_reader_thread2, NULL, "rcu_reader2");
	writer_thread = kthread_run(myrcu_writer_thread, (void *)(unsigned long)value, "rcu_writer");
        wake_up_process(reader_thread1);
        wake_up_process(writer_thread);
        wake_up_process(reader_thread2);
	return 0;
}
static void __exit my_test_exit(void)
{
	printk("goodbye\n");
	//kthread_stop(reader_thread1);
	//kthread_stop(reader_thread2);
	//kthread_stop(writer_thread);
#if TEST_MOD_REF
        while(!atomic_read(&ref))
        {
             msleep(1000);
        }
#endif
#if 1
	if (g_ptr)
		kfree(g_ptr);
#endif
}
MODULE_LICENSE("GPL");
module_init(my_test_init);
module_exit(my_test_exit);
