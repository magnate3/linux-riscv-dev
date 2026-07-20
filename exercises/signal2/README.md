# insmod signal_kernel_test.ko 

```
[root@centos7 singnal2]# insmod signal_kernel_test.ko 
[root@centos7 singnal2]# ./user 
write configfd 
received value 1234

[root@centos7 singnal2]# dmesg | tail -n 10
[233074.844773] pid = 50713
[root@centos7 singnal2]# 
```

# reference

https://github.com/blue119/kernel_user_space_interfaces_example/blob/master/signal_kernel.c