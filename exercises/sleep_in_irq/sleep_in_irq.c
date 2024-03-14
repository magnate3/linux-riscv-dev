/*********************************************
 * Function: module example
 * Author: asmcos@gmail.com
 * Date: 2005-08-24
 * $Id: kegui.c, v 1.6 2006/06/22 13:20:50 asmcos Exp $
 *********************************************/

#include <linux/init.h>
#include <linux/module.h>
#include <mach/gpio.h>
#include <plat/irqs.h> /*IRQ_EINT(x)*/
#include <mach/irqs.h>
#include <linux/interrupt.h> /* request_irq, free_irq */
#include <linux/irq.h> /* set_irq_type */
#include <linux/sched.h>
#include <asm/io.h> /* ioremap */

MODULE_LICENSE("GPL");

volatile u8 * vmem_buzzer;
volatile u32 * vmem_gpd0;

struct tasklet_struct mytasklet;

void print_cpsr(void);

//底半部tasklet仍旧是中断上下文，但是比顶半部延后执行。
//顶半部中断不可嵌套，底半部则可以。它可以被顶半部打断。
//tasklet里面仍旧不可睡眠。
void do_sth(unsigned long data)
{
	volatile int i;
	printk("start in tasklet ......data=%d\n", data);
	print_cpsr();
	for(i=0; i<100000000; i++);
	printk("end in tasklet ......data=%d\n", data);
	//set_current_state(TASK_INTERRUPTIBLE);
	//schedule_timeout(20*HZ);//error!
}

//打印CPU内部寄存器，C语言做不到，内嵌汇编才可以。
void print_cpsr(void)
{
	volatile u32 mode;
	__asm__
	(
	"MRS R0, CPSR \n"
	"MOV %0, R0 \n":"=r"(mode)
	);
	printk("mode in irq=0x%x\n", mode);

}

//中断处理函数, 中断上下文中睡眠会怎么样？
//系统会跑飞了。
//中断上下文中，为保证实时性，调度器是不工作的，
//中断中是假设你不会睡眠放弃CPU控制权的，
//如果你睡眠了，没有代码来唤醒这个放弃了CPU的中断。
//睡眠仅针对进程来说的。
irqreturn_t key_isr(int irq, void *dev_id)
{
	printk("K1 is pressed!!!\n");
	//set_current_state(TASK_INTERRUPTIBLE);
	//schedule_timeout(20*HZ);
	print_cpsr();

	*vmem_buzzer |= (1<<0);

	tasklet_schedule(&mytasklet);
	printk("=========!!!\n");

	return IRQ_HANDLED;
}

int __init akae_init(void)
{
	int rc;

	printk("hello,akaedu\n");

	//让对应进程上下文的用户进程睡眠5秒
	//set_current_state(TASK_INTERRUPTIBLE);
	//schedule_timeout(20*HZ);

	/*在开发板上执行结果，两个都一样，都是160作为K1按键的中断号*/
	printk("K1_irq=%d\n", gpio_to_irq(S5PV210_GPH2(0)));
	printk("K1_irq=%d\n", IRQ_EINT(16));

	set_irq_type(gpio_to_irq(S5PV210_GPH2(0)), IRQF_TRIGGER_FALLING);
	rc = request_irq(gpio_to_irq(S5PV210_GPH2(0)), 
			key_isr,
			0,
			"key isr",
			NULL);

	if(rc < 0)
		printk("request_irq error!\n");

	vmem_buzzer = ioremap(0xE02000A4, 1);//GPD0DAT
	vmem_gpd0 = ioremap(0xE02000A0, 4); //GPD0CON
	*vmem_gpd0 &= ~0xf; 
	*vmem_gpd0 |= 0x1; //设置GPD0(0)输出模式

	//初始化tasklet
	tasklet_init(&mytasklet, do_sth, 0);

	return 0;
}

void __exit akae_exit(void)
{
	printk("module exit\n");
	free_irq(gpio_to_irq(S5PV210_GPH2(0)), NULL);
	iounmap(vmem_buzzer);
	iounmap(vmem_gpd0);

	return;
}

module_init(akae_init);
module_exit(akae_exit);