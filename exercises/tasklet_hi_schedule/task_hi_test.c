#include <linux/interrupt.h>
#include <linux/module.h>
#include <linux/init.h>
static unsigned long data=0;
static struct tasklet_struct tasklet, tasklet1;
static void irq_tasklet_action(unsigned long data)
{
    printk("in irq_tasklet_action the state of the tasklet is :%ld\n", (&tasklet)->state);
    printk("tasklet running. by author\n");
    return;
}
static void irq_tasklet_action1(unsigned long data)
{
    printk("in irq_tasklet_action1 the state of the tasklet1 is :%ld\n", (&tasklet1)->state);
    printk("tasklet1 running. by author\n");
    return;
}
static int   __init tasklet_hi_schedule_init(void)
{
    printk("into tasklet_hi_schedule\n");

    tasklet_init(&tasklet, irq_tasklet_action, data);
    tasklet_init(&tasklet1, irq_tasklet_action1, data);

    printk("The state of the tasklet is :%ld\n", (&tasklet)->state);
    printk("The state of the tasklet1 is :%ld\n", (&tasklet1)->state);
    tasklet_schedule(&tasklet);  //���ж�������ͨ�ж϶���
    //tasklet_hi_schedule(&tasklet1); //���ú���tasklet_hi_schedule( )���ж���������ȼ�����
    tasklet_schedule(&tasklet1);

    printk("The state of the tasklet is :%ld\n", (&tasklet)->state);
    printk("The state of the tasklet1 is :%ld\n", (&tasklet1)->state);
    tasklet_kill(&tasklet);        
    tasklet_kill(&tasklet1);
    printk("out tasklet_hi_schedule\n");
    return 0;
}
static void   __exit tasklet_hi_schedule_exit(void)
{
    printk("Goodbye tasklet_hi_schedule\n");
    return;
}
MODULE_LICENSE("GPL");
module_init(tasklet_hi_schedule_init);
module_exit(tasklet_hi_schedule_exit);
