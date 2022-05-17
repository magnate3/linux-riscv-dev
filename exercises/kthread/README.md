insmod kthread_example.ko 
 ```
[root@centos7 kthread]# dmesg | tail -n 20
[84542.414294] ktime[8] = 0.000000000
[84542.417679] ktime[9] = 0.000000000
[84542.421064] ktime[10] = 0.000000000
[84542.424539] ktime[11] = 0.000000000
[84542.428010] ktime[12] = 0.000000000
[84542.431482] ktime[13] = 0.000000000
[84542.434956] ktime[14] = 0.000000000
[84542.438427] ktime[15] = 0.000000000
[84542.441899] latency for creating kthread: 0.000063051
[84542.446929] latency for set kthread policy: 0.000003010
[84542.452129] time interval for thread creation and run: 1.004245349
[84542.458284] latency to wake up a sleeping thread: 0.000047341
[84542.464003] time of kthread prempting internal_kthread: 0.000006140
[84542.470242] timing module is being unloaded
[85813.066077] I am here!
[85813.068469] kthread_create call was successful!
[85813.068473] kthread state is TASK_INTERRUPTIBLE
[85814.112415] wake up the thread from the init routine
[85814.117362] thread has been woken up in TASK_RUNNING //************************************* wakeup
[85814.122304] kthread state is TASK_INTERRUPTIBLE
[root@centos7 kthread]# rmmod kthread_example.ko 
[root@centos7 kthread]# dmesg | tail -n 20
[84542.428010] ktime[12] = 0.000000000
[84542.431482] ktime[13] = 0.000000000
[84542.434956] ktime[14] = 0.000000000
[84542.438427] ktime[15] = 0.000000000
[84542.441899] latency for creating kthread: 0.000063051
[84542.446929] latency for set kthread policy: 0.000003010
[84542.452129] time interval for thread creation and run: 1.004245349
[84542.458284] latency to wake up a sleeping thread: 0.000047341
[84542.464003] time of kthread prempting internal_kthread: 0.000006140
[84542.470242] timing module is being unloaded
[85813.066077] I am here!
[85813.068469] kthread_create call was successful!
[85813.068473] kthread state is TASK_INTERRUPTIBLE
[85814.112415] wake up the thread from the init routine
[85814.117362] thread has been woken up in TASK_RUNNING
[85814.122304] kthread state is TASK_INTERRUPTIBLE
[86141.580282] thread has been woken up in TASK_RUNNING
[86141.585226] kthread state is TASK_RUNNING
[86141.589230] the terminated thread state is TASK_RUNNING
[86141.594430] Bye!
 ```
 # insmod kthread_example2.ko 
 ```
 [root@centos7 kthread]# dmesg | tail -n 10
[87913.305050] 
kthread 28486 has finished working, waiting for exit
[87918.316684] 
kthread 28486 has finished working, waiting for exit
[87918.316685] 
kthread 28487 has finished working, waiting for exit
[87923.356615] 
kthread 28487 has finished working, waiting for exit
[87923.356616] 
kthread 28486 has finished working, waiting for exit
[root@centos7 kthread]# dmesg | tail -n 20
[87938.476344] 
kthread 28486 has finished working, waiting for exit
[87938.476350] 
kthread 28487 has finished working, waiting for exit
[87943.516266] 
kthread 28486 has finished working, waiting for exit
[87943.516270] 
kthread 28487 has finished working, waiting for exit
[87948.556192] 
kthread 28487 has finished working, waiting for exit
[87948.556214] 
kthread 28486 has finished working, waiting for exit
[87953.596141] 
kthread 28487 has finished working, waiting for exit
[87953.596144] 
kthread 28486 has finished working, waiting for exit
[87958.636006] 
kthread 28486 has finished working, waiting for exit
[87958.636007] 
kthread 28487 has finished working, waiting for exit
 ```

