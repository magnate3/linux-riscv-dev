

#  insmod  workqueue_test.ko 

创建了一个myworkqueue线程
```
[root@centos7 work_queue]# insmod  workqueue_test.ko 
[root@centos7 work_queue]# ps -elf | grep myworkqueue
1 I root      42009      2  0  60 -20 -     0 rescue 21:12 ?        00:00:00 [myworkqueue]
0 S root      42028  40767  0  80   0 -  1729 pipe_w 21:13 pts/0    00:00:00 grep --color=auto myworkqueue
[root@centos7 work_queue]# rmmod  workqueue_test.ko 
[root@centos7 work_queue]# ps -elf | grep myworkqueue
0 S root      42032  40767  0  80   0 -  1729 pipe_w 21:13 pts/0    00:00:00 grep --color=auto myworkqueue
```

#  cancel_work_sync

Don't forget to cancel the possible work in the queue (cancel_delayed_work_sync) when you close the workqueue, otherwise your kernel will crash. 



# insmod  workqueue_test2.ko 

```
[root@centos7 work_queue]# insmod  workqueue_test2.ko 
[451890.628503] work queueu test start 
[451890.632414] Create Workqueue successful!
[451890.636406] *************** first ret=1!
[451890.640401] Example:ret= 1,i=0
[451890.698679] *********** My name is delay_func!
[451890.703189] delay_fun:i=0
[451890.738677] delay_fun:i=1
[451890.758685] Example:ret= 1,i=1
[451890.768705] delay_fun:i=2
[451890.878675] Example:ret= 1,i=2
[451890.998674] ************** second ret=1!
[451891.002666] *********** My name is delay_func!  //再一次执行
[451891.007175] delay_fun:i=0
[451891.038673] delay_fun:i=1
[451891.078671] delay_fun:i=2
[root@centos7 work_queue]# 
``` 