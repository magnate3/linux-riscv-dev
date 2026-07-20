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
# kthread_example3.ko

## insmod kthread_example3.ko

```
[root@centos7 kthread]# dmesg | tail -n 20
[   44.980995] virbr0: port 1(virbr0-nic) entered listening state
[   45.011475] virbr0: port 1(virbr0-nic) entered disabled state
[   45.130083] IPv6: ADDRCONF(NETDEV_UP): docker0: link is not ready
[89041.762988] rt_test: loading out-of-tree module taints kernel.
[89041.768852] rt_test: module verification failed: signature and/or required key missing - tainting kernel
[602180.619511] Initializing kernel mode thread example module
[602180.625064] Creating Threads
[602180.628028] Getting current CPU 118 to binding worker thread
[602180.630921] Worker task created successfully
[602180.635259] Getting current CPU 2 to binding default thread
[602180.635305] Worker thread running
[602180.635347] Default thread executing on system CPU:2 
[602180.643813] Default thread running
[602180.647424] Worker thread executing on system CPU:118 
[602184.675366] Worker thread executing on system CPU:118 
[602186.675357] Default thread executing on system CPU:2 
[602188.755338] Worker thread executing on system CPU:118 
[602192.835317] Worker thread executing on system CPU:118 
[602193.075309] Default thread executing on system CPU:2 
[602196.925286] Worker thread executing on system CPU:118 
[root@centos7 kthread]# dmesg | tail -n 20
[602180.628028] Getting current CPU 118 to binding worker thread
[602180.630921] Worker task created successfully
[602180.635259] Getting current CPU 2 to binding default thread
[602180.635305] Worker thread running
[602180.635347] Default thread executing on system CPU:2 
[602180.643813] Default thread running
[602180.647424] Worker thread executing on system CPU:118 
[602184.675366] Worker thread executing on system CPU:118 
[602186.675357] Default thread executing on system CPU:2 
[602188.755338] Worker thread executing on system CPU:118 
[602192.835317] Worker thread executing on system CPU:118 
[602193.075309] Default thread executing on system CPU:2 
[602196.925286] Worker thread executing on system CPU:118 
[602199.475282] Default thread executing on system CPU:2 
[602200.995267] Worker thread executing on system CPU:118 
[602205.075252] Worker thread executing on system CPU:118 
[602205.875240] Default thread executing on system CPU:2 
[602209.155216] Worker thread executing on system CPU:118 
[602212.275247] Default thread executing on system CPU:2 
[602213.235193] Worker thread executing on system CPU:118 
[root@centos7 kthread]# 
```
## rmmod kthread_example3.ko 
```
[root@centos7 kthread]# dmesg | tail -n 20
[602254.034961] Worker thread executing on system CPU:118 
[602257.074939] Default thread executing on system CPU:2 
[602258.114957] Worker thread executing on system CPU:118 
[602262.194920] Worker thread executing on system CPU:118 
[602263.474910] Default thread executing on system CPU:2 
[602266.274886] Worker thread executing on system CPU:118 
[602269.874850] Default thread executing on system CPU:2 
[602270.354848] Worker thread executing on system CPU:118 
[602274.434787] Worker thread executing on system CPU:118 
[602276.274801] Default thread executing on system CPU:2 
[602278.514709] Worker thread executing on system CPU:118 
[602282.594646] Worker thread executing on system CPU:118 
[602282.674632] Default thread executing on system CPU:2 
[602286.674574] Worker thread executing on system CPU:118 
[602288.759093] Module removing from kernel, threads stopping
[602289.074526] Default thread executing on system CPU:2 
[602290.754516] Worker task stopped
[602295.154431] Default task exiting
[602295.157744] Default task stopped
[602295.161046] Bye Bye
```