#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/kthread.h>
#include <linux/wait.h>


static int pid1 = 1;
static int pid2 = 2;
static int condition;

DECLARE_WAIT_QUEUE_HEAD(wq);


struct task_struct *task1;
struct task_struct *task2;

static int thread_function(void *data){

    int *thread_id = (int*)data;
    int i = 0;
    while(i < 10){
        printk(KERN_INFO "install kernel thread: %d\n", *thread_id);
        i++;

        if(*thread_id == 1)
        {
            wait_event(wq, condition == 0xA);

            condition = 0xB;
            wake_up(&wq);
        }
        else{
            wait_event(wq, condition == 0xB);

            condition = 0xA;
            wake_up(&wq);
        }
    }
    return 0;
}


static int __init kernel_init(void)
{
    printk("Module starting ... ... ...\n");

    condition = 0xA;

    task1 = kthread_create(&thread_function, (void *)&pid1, "pradeep");
    task2 = kthread_create(&thread_function, (void *)&pid2, "pradeep");

    wake_up_process(task1);
    wake_up_process(task2);

    return 0;
}

static void __exit kernel_fini (void)
{
    printk("Module terminating ... ... ...\n");

}

module_init(kernel_init);
module_exit(kernel_fini);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("NINHLD");
MODULE_VERSION("1.0.0");