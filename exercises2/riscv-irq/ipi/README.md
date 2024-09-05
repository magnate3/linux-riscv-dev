


# smp_call_function_single
```
static void kvm_timer_init(void)
{


	pr_debug("kvm: max_tsc_khz = %ld\n", max_tsc_khz);
	for_each_online_cpu(cpu)
		smp_call_function_single(cpu, tsc_khz_changed, NULL, 1);


}

```


#  test   

```
[root@centos7 ipi]# cat /proc/devices  | grep my
241 mymod
[root@centos7 ipi]# mknod /dev/my_device c 241 0
[root@centos7 ipi]# ./user  /dev/my_device
0: quit
1: mutex lock
2: mutex unlock
3: spin lock
4: spin unlock
5: ipi good
6: ipi blocking
7: ipi spinning
choice? 5
0: quit
1: mutex lock
2: mutex unlock
3: spin lock
4: spin unlock
5: ipi good
6: ipi blocking
7: ipi spinning
choice? 6
0: quit
1: mutex lock
2: mutex unlock
3: spin lock
4: spin unlock
5: ipi good
6: ipi blocking
7: ipi spinning
choice? 7
0: quit
1: mutex lock
2: mutex unlock
3: spin lock
4: spin unlock
5: ipi good
6: ipi blocking
7: ipi spinning
choice? 0
quitting
[root@centos7 ipi]# 241 mymod
[root@centos7 ipi]# 
```

+ dmesg   
```
root@centos7 ipi]# dmesg | tail -n 20
[2326288.152400] mymod_init end
[2326404.561189] mymod_cdev_open
[2326404.564150] mymod_cdev_open end
[2326412.231231] mymod_cdev_ioctl start
[2326412.234794] mymod_cdev_ioctl smp call good
[2326412.239048] ipi_good start
[2326412.241918] ipi_good end
[2326412.244613] mymod_cdev_ioctl end
[2326423.510232] mymod_cdev_ioctl start
[2326423.513792] mymod_cdev_ioctl smp call blocking
[2326423.518388] ipi_bad_blocking start
[2326423.521948] ipi_bad_blocking end
[2326423.525333] mymod_cdev_ioctl end
[2326425.940416] mymod_cdev_ioctl start
[2326425.943973] mymod_cdev_ioctl smp call non-blocking
[2326425.948916] ipi_bad_nonblocking start
[2326425.952736] ipi_bad_nonblocking end
[2326425.956382] mymod_cdev_ioctl end
[2326428.656100] mymod_cdev_release start
[2326428.659836] mymod_cdev_release end
[root@centos7 ipi]# 
```

# references


[核间中断（inter-process interrupt，IPI）](https://www.cnblogs.com/harrypotterjackson/p/17548837.html)   