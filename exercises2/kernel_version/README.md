
# insmod  version_test.ko 
```
root@ubuntu:~# insmod  version_test.ko 
root@ubuntu:~# uname -a
Linux ubuntu 6.3.2 #1 SMP PREEMPT_DYNAMIC Mon May 15 19:50:18 CST 2023 x86_64 x86_64 x86_64 GNU/Linux
```

```
root@ubuntu:~# dmesg | tail -n 3
[  865.296243] ct_init:12 new version 5
[  865.296256] ct_init:16 KERNEL_VERSION(5,15,71) = 331591
[  865.296260] ct_init:17 KERNEL_VERSION(4,1,15) = 262415
root@ubuntu:~# 
```