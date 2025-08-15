/**********************************************
 * Author: lewiyon@hotmail.com
 * File name: delay_wq.c
 * Description: learn the function queue_delayed_work()
 * Date: 2011-12-24
 *********************************************/

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/delay.h>

struct workqueue_struct *test_wq;
struct delayed_work test_dwq;

void delay_func(struct work_struct *work);

void delay_func(struct work_struct *work)
{
    int i;

    printk(KERN_INFO "*********** My name is delay_func!\n");
    for (i = 0; i < 3; i++) {
        printk(KERN_ERR "delay_fun:i=%d\n", i);
        msleep(10);
    }
}

static int __init example_init(void)
{
    int i;
    int ret;

 
    printk(KERN_ERR "work queueu test start \n");
    test_wq = create_workqueue("test_wq");
    if (!test_wq) {
        printk(KERN_ERR "No memory for workqueue\n");
        return 1;   
    }
    printk(KERN_INFO "Create Workqueue successful!\n");

    INIT_DELAYED_WORK(&test_dwq, delay_func);
    
    ret = queue_delayed_work(test_wq, &test_dwq, 5);
    printk(KERN_INFO "*************** first ret=%d!\n", ret);
    
    for (i = 0; i < 3; i++) { 
        printk(KERN_INFO "Example:ret= %d,i=%d\n", ret, i);
        msleep(100);
    }

    ret = queue_delayed_work(test_wq, &test_dwq, 0);
    printk(KERN_INFO "************** second ret=%d!\n", ret);

    return 0;
}

static void __exit example_exit(void)
{
    int ret;
    ret = cancel_delayed_work(&test_dwq);
    flush_workqueue(test_wq);
    destroy_workqueue(test_wq);
    printk(KERN_INFO "Goodbay! ret=%d\n", ret);
}


module_init(example_init);
module_exit(example_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("lewiyon <lewiyon@hotmail.com>");
