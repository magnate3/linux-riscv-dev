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

//�װ벿tasklet�Ծ����ж������ģ����Ǳȶ��벿�Ӻ�ִ�С�
//���벿�жϲ���Ƕ�ף��װ벿����ԡ������Ա����벿��ϡ�
//tasklet�����Ծɲ���˯�ߡ�
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

//��ӡCPU�ڲ��Ĵ�����C��������������Ƕ���ſ��ԡ�
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

//�жϴ�����, �ж���������˯�߻���ô����
//ϵͳ���ܷ��ˡ�
//�ж��������У�Ϊ��֤ʵʱ�ԣ��������ǲ������ģ�
//�ж����Ǽ����㲻��˯�߷���CPU����Ȩ�ģ�
//�����˯���ˣ�û�д������������������CPU���жϡ�
//˯�߽���Խ�����˵�ġ�
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

	//�ö�Ӧ���������ĵ��û�����˯��5��
	//set_current_state(TASK_INTERRUPTIBLE);
	//schedule_timeout(20*HZ);

	/*�ڿ�������ִ�н����������һ��������160��ΪK1�������жϺ�*/
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
	*vmem_gpd0 |= 0x1; //����GPD0(0)���ģʽ

	//��ʼ��tasklet
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