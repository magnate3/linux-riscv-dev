#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/utsname.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/timer.h>
#include <linux/rtc.h>
#include <linux/workqueue.h>
#include <asm/uaccess.h>
#define TIMER_OVER 3/*����ʱ����*/
static struct workqueue_struct *queue = NULL;
static struct delayed_work work;//��������
static struct timer_list queue_timer;/*�ں˶�ʱ��*/
static int count = 0;/*�����жϴ���ͳ��*/
//��ʱ���жϷ������
static void queue_timer_function(int para){
	queue_delayed_work(queue,&work,0);//��ʱִ�й�������
	count++;
	printk("Timer expired and para is %d !\n",para);
	printk("count = %d\n",count);
}
//ע��һ���ں˶�ʱ��
static void queue_timer_register(struct timer_list* ptimer,unsigned int timeover){
	printk("timer_register!!!\n");
	init_timer(&queue_timer);
	queue_timer.data = timeover;
	queue_timer.expires = jiffies + (5*HZ);
	queue_timer.function = queue_timer_function;
	add_timer(&queue_timer);
}
//��ʱ���װ벿������
static void work_handler(struct work_struct *data){
	struct timeval tv;	//struct timeval Ϊ�趨ʱ����ȡʱ��ʱʹ�õĽṹ�壬tv_sec �����ѵ�ǰʱ�任��Ϊ�룬tv_usec ֵָ�����ȡ tv_usec �޷���ʾ�� us ��λ������ʱ�䡣
	struct rtc_time tm;
	printk(KERN_ALERT"Work handler function\n");
	do_gettimeofday(&tv);//�˺�����ȡ��1970-1-1 0:0:0�����ڵ�ʱ��ֵ������timeval�Ľṹ����ߡ�
	rtc_time_to_tm(tv.tv_sec+8*3600,&tm);//���ں˺�����ϵͳʵʱʱ��ʱ��ת��Ϊ�������α�׼ʱ�䣨GMT�������Ҫ�õ�����ʱ���衣��Ҫ����ʱ�䴦����ݼ���1900���·ݼ���1��Сʱ����8��������ο��ں���rtc.h��rtc_time_to_tm����ʵ�֡�
	printk("BeiJing time :%d-%d-%d %d:%d:%d\n",tm.tm_year+1900,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min,tm.tm_sec);//����ʱ��	
	queue_timer_register(&queue_timer,TIMER_OVER);
}

static int __init hello_proc_init(void)
{
	printk(KERN_ALERT"loading time ....\n");
	queue = create_singlethread_workqueue("myqueue");//������һ�߳�
	if(queue == NULL){
		printk(KERN_ALERT"create myqueue workqueue error\n");
	}
	queue_timer_register(&queue_timer,TIMER_OVER);
	//��ʼ���������в�����������Ԥ��������
	INIT_DELAYED_WORK(&work,work_handler);//��ʼ����������	
	
	return 0;
}
static void __exit hello_proc_exit(void)
{
	printk("GoodBye kernel\n");
	del_timer(&queue_timer);
	flush_workqueue(queue);//ˢ�¹�������
	destroy_workqueue(queue);//�ͷŴ����Ĺ�������
}
MODULE_LICENSE("GPL");
module_init(hello_proc_init);
module_exit(hello_proc_exit);