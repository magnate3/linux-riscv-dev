/*
 * 模拟net_device{} 中 ip_ptr 指针的使用，研究RCU 的作用.
 *
 * do_task1 首先获取指针，进入到临近区中；
 * do_task2 设置指针为空，设置 call_rcu() 钩子函数；
 * do_task1 在临界区内可以继续操作该部分内存，当退出临界区之后，call_rcu() 中
 * 设置的钩子函数会被调用，释放内存.
 *
 * 编译通过内核列表:
 * 2.6.35.6
 */
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/kthread.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>

struct TEST {
	struct rcu_head	rcu_head;
};
struct TEST *ptest;

struct timeval tv;

static struct task_struct *task1;
static struct task_struct *task2;

/* rcu_read_lock()/rcu_read_unlock() 中不能休眠 */
static int do_task1(void *data)
{
#if 1
        struct TEST *ptest2;
	/* 忙等待1s */
	mdelay(1000);
        pr_info("task1 enter rcu \n");
	rcu_read_lock();
	ptest2 = rcu_dereference(ptest);
	mdelay(10000);		//忙等待10s
        if(ptest) {
            pr_info("ptest is %p \n", ptest);
        } 
        else {
            pr_info("ptest is null \n");
        }
        if(ptest2) {
            pr_info("ptest2 is %p \n", ptest2);
        } 
        else {
            pr_info("ptest2 is null \n");
        }
	rcu_read_unlock();
#if 1
	mdelay(10000);		//忙等待10s
        if(ptest) {
            pr_info("after rcu unlock , ptest is %p \n", ptest);
        } 
        else {
            pr_info("after rcu unlock,  ptest is null \n");
        }
#endif 
#endif
	while (1) {
		msleep(10);
		if (kthread_should_stop())
			break;
	}
	return 0;
}

static void rcu_demo_free(struct rcu_head *head)
{
	struct TEST *tmp = container_of(head, struct TEST, rcu_head);

	do_gettimeofday(&tv);
	printk("%s %ld %ld:free ptest\n",__func__, tv.tv_sec, tv.tv_usec);
	kfree(tmp);
}

static int do_task2(void *data)
{
	struct TEST *tmp = ptest;

	/* taks2 not nedd 忙等待1s */
	//mdelay(1000);
        pr_info("task2 enter rcu \n");
	rcu_assign_pointer(ptest, NULL);
        pr_info("task2 assign ptest null \n");
	call_rcu(&tmp->rcu_head, rcu_demo_free);

	while (1) {
		msleep(10);
		if (kthread_should_stop())
			break;
	}
	return 0;
}

static int __init demo_init(void)
{
	int ret;

	ret = -ENOMEM;
	ptest = kzalloc(sizeof(*ptest), GFP_KERNEL);
	if (!ptest) {
		goto out;
	}
	rcu_assign_pointer(ptest, ptest);

	/* 开启两个内核线程 */
	task1 = kthread_create(do_task1, NULL, "demo_task1");
	if (IS_ERR(task1)) {
		printk("kthread err 1\n");
		ret = PTR_ERR(task1);
		task1 = NULL;
		goto err0;
	}
	do_gettimeofday(&tv);
	printk("%ld %ld:create task 1 success\n", tv.tv_sec, tv.tv_usec);
	task2 = kthread_create(do_task2, NULL, "demo_task2");
	if (IS_ERR(task2)) {
		printk("kthread err 2\n");
		ret = PTR_ERR(task2);
		task2 = NULL;
		goto err1;
	}
	do_gettimeofday(&tv);
	printk("%ld %ld:create task 2 success\n", tv.tv_sec, tv.tv_usec);

	wake_up_process(task1);
	wake_up_process(task2);

	return 0;
err1:
	kthread_stop(task1);
err0:
	kfree(ptest);
out:
	return ret;
}
      
static void __exit demo_exit(void)
{
	/* 内存释放 */
	if (ptest)
		kfree(ptest);

	/* 结束两个内核线程 */
	if (task1)
		kthread_stop(task1);
	if (task2)
		kthread_stop(task2);
}

module_init(demo_init);
module_exit(demo_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("yuchao");
