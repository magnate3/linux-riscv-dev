 # insmod wakeup.ko 
  ```
 [ 2480.357403] wake thread 0 0
[ 2480.357404] sleep thread 1 0
[ 2480.357406] sleep thread 2 0
[ 2481.365936] wake thread 0 1
[ 2481.365937] sleep thread 2 1
[ 2481.365938] sleep thread 1 1
[ 2482.374446] wake thread 0 2
[ 2482.374449] sleep thread 1 2
[ 2482.374449] sleep thread 2 2
[ 2483.382961] wake thread 0 3
[ 2483.382963] sleep thread 2 3
[ 2483.382964] sleep thread 1 3
[ 2484.391487] wake thread 0 4
[ 2484.391488] sleep thread 1 4
[ 2484.391490] sleep thread 2 4
[ 2485.400012] wake thread 0 5
[ 2485.400014] sleep thread 2 5
[ 2485.400015] sleep thread 1 5
[ 2486.408541] wake thread 0 6
[ 2486.408543] sleep thread 1 6
[ 2486.408543] sleep thread 2 6
[ 2487.417048] wake thread 0 7
[ 2487.417050] sleep thread 2 7
[ 2487.417052] sleep thread 1 7
[ 2488.425564] wake thread 0 8
[ 2488.425564] sleep thread 1 8
[ 2488.425566] sleep thread 2 8
[ 2489.434076] wake thread 0 9
[ 2489.434077] sleep thread 2 9
[ 2489.434078] sleep thread 1 9
[ 2490.442601] wake thread 0 10
[ 2490.442601] sleep thread 1 10
[ 2490.442603] sleep thread 2 10
[ 2491.451384] wake thread 0 11
[ 2491.451386] sleep thread 2 11
[ 2491.451387] sleep thread 1 11
[ 2492.460155] wake thread 0 12
[ 2492.460157] sleep thread 1 12
[ 2492.460157] sleep thread 2 12
[ 2493.468926] wake thread 0 13
[ 2493.468928] sleep thread 2 13
[ 2493.468928] sleep thread 1 13
[ 2494.477707] wake thread 0 14
[ 2494.477708] sleep thread 1 14
[ 2494.477709] sleep thread 2 14
[ 2495.486491] wake thread 0 15
[ 2495.486493] sleep thread 2 15
[ 2495.486494] sleep thread 1 15
[ 2496.495276] wake thread 0 16
[ 2496.495277] sleep thread 1 16
[ 2496.495279] sleep thread 2 16
[ 2497.504048] wake thread 0 17
[ 2497.504050] sleep thread 2 17
[ 2497.504051] sleep thread 1 17
[ 2498.512819] wake thread 0 18
[ 2498.512819] sleep thread 1 18
[ 2498.512821] sleep thread 2 18
[ 2499.521588] wake thread 0 19
[ 2499.521590] sleep thread 2 19
[ 2499.521590] sleep thread 1 19
[ 2500.530372] wake thread 0 20
[ 2500.530374] sleep thread 1 20
[ 2500.530375] sleep thread 2 20
[ 2501.539155] wake thread 0 21
[ 2501.539157] sleep thread 2 21
[ 2501.539158] sleep thread 1 21
[ 2502.547924] wake thread 0 22
[ 2502.547924] sleep thread 1 22
[ 2502.547926] sleep thread 2 22
[ 2503.556694] wake thread 0 23
[ 2503.556697] sleep thread 1 23
[ 2504.562518] wake thread 0 24
  ```
  
#  TASK_INTERRUPTIBLE 和TASK_UNINTERRUPTIBLE 的区别
TASK_INTERRUPTIBLE是可以被信号和wake_up()唤醒的，当信号到来时，进程会被设置为可运行。
而TASK_UNINTERRUPTIBLE只能被wake_up()唤醒。

# 信号本质
信号是在软件层次上对中断机制的一种模拟，软中断

信号来源
信号事件的发生有两个来源：
硬件来源：(比如我们按下了键盘或者其它硬件故障)；
软件来源：最常用发送信号的系统函数是kill, raise, alarm和setitimer以及sigqueue函数，软件来源还包括一些非法运算等操作。

区分是什么原因唤醒进程，用signal_pending( current )；
检查当前进程是否有信号处理，返回不为0表示有信号需要处理。-ERESTARTSYS 表示信号函数处理完毕后重新执行信号函数前的某个系统调用。也就是说,如果信号函数前有发生系统调用，在调度用户信号函数之前,内核会检查系统调用的返回值，看看是不是因为这个信号而中断了系统调用.如果返回值-ERESTARTSYS,并且当前调度的信号具备-ERESTARTSYS属性,系统就会在用户信号函数返回之后再执行该系统调用。
————————————————
 
 
   
   
   
  