/*:kernel_sem.c
��������ʾ���ں��ź���ʹ�÷������ں��̵߳Ĵ�������
�������$ make
������$sudo insmod kernel_sem.ko
�鿴�ں���־��Ϣ��$dmesg
ɾ����$sudo rmmod kernel_sem
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
              down(&sem1);      //��ȡ�ź���1
              printk("kthread %d: %d  ",current->pid, num[i]);
              up(&sem2);    //�ͷ��ź���2
       }
       if(current->mm ==NULL)	
              printk("\nkthread %d MM STRUCT IS null, actitve->mm address is %p\n",current->pid,current->active_mm);
       while(!kthread_should_stop()){ //��kthread_stop���ʹ��
        	printk("\nkthread %d has finished working, waiting for exit\n",current->pid);
		set_current_state(TASK_UNINTERRUPTIBLE); //���ý���״̬
		schedule_timeout(5*HZ);  	//���û��ѡ����µ���ʱ�䣬5*HZԼ����5��
       }

       return 0;
}
int thread_two(void *p)
{
       int *num = (int *)p;
       int i;
       for(i = 0; i < 5; i++){
              down(&sem2);             //��ȡ�ź���2
              printk("kthread %d: %d  ",current->pid, num[i]);	
              up(&sem1);           //�ͷ��ź���1
       }
       if(current->mm ==NULL)	
              printk("\nkthread %d MM STRUCT IS null, actitve->mm address is %p\n",current->pid,current->active_mm);
       while(!kthread_should_stop()){ //��kthread_stop���ʹ��
        	printk("\nkthread %d has finished working, waiting for exit\n",current->pid);
		set_current_state(TASK_UNINTERRUPTIBLE);  
		schedule_timeout(5*HZ);  
       }
       return 0;
}
static int kernelsem_init(void)
{       
       printk("kernel_sem is installed\n");
       sema_init(&sem1,1);  //��ʼ���ź���1�� ʹ�ź���1����ɱ���ȡ
       sema_init(&sem2,0);  //��ʼ���ź���2��ʹ�ź���2ֻ�б��ͷź�ſɱ���ȡ
       //sema_init(&sem2,2);//�۲��ź�����ֵ��ͬ���µ��߳�����˳��	
       test_task1 = kthread_run(thread_one, num[0], "test_task1");
       test_task2 = kthread_run(thread_two, num[1], "test_task2");
        // ��������� kthread_run��Ҳ��ʹ�� kthread_create��wake_up_process ���������ʹ��
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
	wake_up_process(test_task1);   //��kthread_create���ʹ��
	wake_up_process(test_task2);  
	*/
       return 0;
}
static void kernelsem_exit(void)
{
       kthread_stop(test_task1);    //����߳�ʹ��while(1)ѭ������Ҫʹ�øú���ֹͣ�߳�
       kthread_stop(test_task2);    //��������while(!kthread_should_stop()���ʹ��
       printk("\nkernel_sem says goodbye\n");

}
module_init(kernelsem_init);
module_exit(kernelsem_exit);