 # make
  ```
[root@centos7 workqueue2]# make
make -C /usr/src/kernels/4.14.0-115.el7a.0.1.aarch64 M=/root/private_lkm/others/half_bottom/workqueue2 modules
make[1]: Entering directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
  CC [M]  /root/private_lkm/others/half_bottom/workqueue2/hello_workqueue.o
/root/private_lkm/others/half_bottom/workqueue2/hello_workqueue.c: In function ‘queue_timer_register’:
/root/private_lkm/others/half_bottom/workqueue2/hello_workqueue.c:31:23: warning: assignment from incompatible pointer type [enabled by default]
  queue_timer.function = queue_timer_function;
                       ^
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /root/private_lkm/others/half_bottom/workqueue2/hello_workqueue.mod.o
  LD [M]  /root/private_lkm/others/half_bottom/workqueue2/hello_workqueue.ko
make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
[root@centos7 workqueue2]# ls  /usr/src/kernels/
4.14.0-115.el7a.0.1.aarch64  4.18.0-305.10.2.el7.aarch64
4.18.0-193.28.1.el7.aarch64  4.18.0-305.10.2.el7.aarch64+debug
[root@centos7 workqueue2]#
  ```
 # insmod  hello_workqueue.ko
 
 ```
 [  168.163809] hello_workqueue: loading out-of-tree module taints kernel.
[  168.170381] hello_workqueue: module verification failed: signature and/or required key missing - tainting kernel
[  168.181034] loading time ....
[  168.184042] timer_register!!!
[  173.253873] Timer expired and para is 3 !
[  173.257867] count = 1
[  173.260143] Work handler function
[  173.263443] BeiJing time :2022-4-1 17:3:3
[  173.267438] timer_register!!!
[  178.293787] Timer expired and para is 3 !
[  178.297780] count = 2
[  178.300056] Work handler function
[  178.303355] BeiJing time :2022-4-1 17:3:8
[  178.307350] timer_register!!!
[  183.333684] Timer expired and para is 3 !
[  183.337677] count = 3
[  183.339945] Work handler function
[  183.343244] BeiJing time :2022-4-1 17:3:13
[  183.347327] timer_register!!!
 ```
  
 
   
   
   
  