/*:kernel_sem.c
本程序演示了内核信号量使用方法、内核线程的创建方法
编译命令：$ make
添加命令：$sudo insmod kernel_sem.ko
查看内核日志信息：$dmesg
删除：$sudo rmmod kernel_sem
*/

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/semaphore.h>

static struct task_struct *test_task1; 
static struct task_struct *test_task2;  
struct semaphore sem1;
struct semaphore sem2;
 
int num[2][5] = {
       {0,2,4,6,8},
       {1,3,5,7,9}
};
int thread_one(void *p);
int thread_two(void *p);

int thread_one(void *p)
{
       int *num = (int *)p;
       int i;
       for(i = 0; i < 5; i++){
              down(&sem1);      //获取信号量1
              printk("kthread %d: %d  ",current->pid, num[i]);
              up(&sem2);    //释放信号量2
       }
       if(current->mm ==NULL)	
              printk("\nkthread %d MM STRUCT IS null, actitve->mm address is %p\n",current->pid,current->active_mm);
       while(!kthread_should_stop()){ //与kthread_stop配合使用
        	printk("\nkthread %d has finished working, waiting for exit\n",current->pid);
		set_current_state(TASK_UNINTERRUPTIBLE); //设置进程状态
		schedule_timeout(5*HZ);  	//设置唤醒、重新调度时间，5*HZ约等于5秒
       }

       return 0;
}
int thread_two(void *p)
{
       int *num = (int *)p;
       int i;
       for(i = 0; i < 5; i++){
              down(&sem2);             //获取信号量2
              printk("kthread %d: %d  ",current->pid, num[i]);	
              up(&sem1);           //释放信号量1
       }
       if(current->mm ==NULL)	
              printk("\nkthread %d MM STRUCT IS null, actitve->mm address is %p\n",current->pid,current->active_mm);
       while(!kthread_should_stop()){ //与kthread_stop配合使用
        	printk("\nkthread %d has finished working, waiting for exit\n",current->pid);
		set_current_state(TASK_UNINTERRUPTIBLE);  
		schedule_timeout(5*HZ);  
       }
       return 0;
}
static int kernelsem_init(void)
{       
       printk("kernel_sem is installed\n");
       sema_init(&sem1,1);  //初始化信号量1， 使信号量1最初可被获取
       sema_init(&sem2,0);  //初始化信号量2，使信号量2只有被释放后才可被获取
       //sema_init(&sem2,2);//观察信号量初值不同导致的线程运行顺序	
       test_task1 = kthread_run(thread_one, num[0], "test_task1");
       test_task2 = kthread_run(thread_two, num[1], "test_task2");
        // 如果不适用 kthread_run，也已使用 kthread_create与wake_up_process ，二者配合使用
  /*
	int err;	       
	test_task1 = kthread_create(thread_one, num[0], "test_task1");	 
        test_task2 = kthread_create(thread_two, num[1], "test_task2");
       if(IS_ERR(test_task1)){    
	      printk("Unable to start kernel thread.\n");    
	      err = PTR_ERR(test_task1);    
	      test_task1 = NULL;    
	      return err;    
       } 
       if(IS_ERR(test_task2)){    
	      printk("Unable to start kernel thread.\n");    
	      err = PTR_ERR(test_task2);    
	      test_task2 = NULL;    
	      return err;    
       }
	wake_up_process(test_task1);   //与kthread_create配合使用
	wake_up_process(test_task2);  
	*/
       return 0;
}
static void kernelsem_exit(void)
{
       kthread_stop(test_task1);    //如果线程使用while(1)循环，需要使用该函数停止线程
       kthread_stop(test_task2);    //本程序与while(!kthread_should_stop()配合使用
       printk("\nkernel_sem says goodbye\n");

}
module_init(kernelsem_init);
module_exit(kernelsem_exit);