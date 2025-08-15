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
#define TIMER_OVER 3/*调用时间间隔*/
static struct workqueue_struct *queue = NULL;
static struct delayed_work work;//工作队列
static struct timer_list queue_timer;/*内核定时器*/
static int count = 0;/*进入中断次数统计*/
//定时器中断服务程序
static void queue_timer_function(int para){
	queue_delayed_work(queue,&work,0);//延时执行工作任务
	count++;
	printk("Timer expired and para is %d !\n",para);
	printk("count = %d\n",count);
}
//注册一个内核定时器
static void queue_timer_register(struct timer_list* ptimer,unsigned int timeover){
	printk("timer_register!!!\n");
	init_timer(&queue_timer);
	queue_timer.data = timeover;
	queue_timer.expires = jiffies + (5*HZ);
	queue_timer.function = queue_timer_function;
	add_timer(&queue_timer);
}
//定时器底半部处理函数
static void work_handler(struct work_struct *data){
	struct timeval tv;	//struct timeval 为设定时间或获取时间时使用的结构体，tv_sec 变量把当前时间换算为秒，tv_usec 值指定或获取 tv_usec 无法表示的 us 单位经过的时间。
	struct rtc_time tm;
	printk(KERN_ALERT"Work handler function\n");
	do_gettimeofday(&tv);//此函数获取从1970-1-1 0:0:0到现在的时间值，存在timeval的结构体里边。
	rtc_time_to_tm(tv.tv_sec+8*3600,&tm);//此内核函数将系统实时时钟时间转换为格林尼治标准时间（GMT）。如果要得到北京时间需。需要将此时间处理（年份加上1900，月份加上1，小时加上8）。具体参看内核中rtc.h中rtc_time_to_tm代码实现。
	printk("BeiJing time :%d-%d-%d %d:%d:%d\n",tm.tm_year+1900,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min,tm.tm_sec);//北京时间	
	queue_timer_register(&queue_timer,TIMER_OVER);
}

static int __init hello_proc_init(void)
{
	printk(KERN_ALERT"loading time ....\n");
	queue = create_singlethread_workqueue("myqueue");//创建单一线程
	if(queue == NULL){
		printk(KERN_ALERT"create myqueue workqueue error\n");
	}
	queue_timer_register(&queue_timer,TIMER_OVER);
	//初始化工作队列并将工作队列预处理函数绑定
	INIT_DELAYED_WORK(&work,work_handler);//初始化工作任务	
	
	return 0;
}
static void __exit hello_proc_exit(void)
{
	printk("GoodBye kernel\n");
	del_timer(&queue_timer);
	flush_workqueue(queue);//刷新工作队列
	destroy_workqueue(queue);//释放创建的工作队列
}
MODULE_LICENSE("GPL");
module_init(hello_proc_init);
module_exit(hello_proc_exit);