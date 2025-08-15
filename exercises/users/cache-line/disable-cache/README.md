
# Disabling L1 and L2 caches

```
root@ubuntux86:/work/kernel_learn/mem_test# insmod  disableCache.ko 
root@ubuntux86:/work/kernel_learn/mem_test#  dmesg | grep Disabling
[    3.839122] snd_hda_intel 0000:01:00.1: Disabling MSI
[    4.345107] nouveau 0000:01:00.0: DRM: Disabling PCI power management to avoid bug
[25823.301613] Disabling L1 and L2 caches
```

```
root@ubuntux86:/work/kernel_learn/mem_test# gcc band.c  -o band
root@ubuntux86:/work/kernel_learn/mem_test# ./band
read 999999 times from main memory need 0.001637 sec
    The bandwidth is 610872938.301772byte/sec
root@ubuntux86:/work/kernel_learn/mem_test# 
```

# enabling L1 and L2 caches

```
root@ubuntux86:/work/kernel_learn/mem_test# ./band
read 999999 times from main memory need 0.001638 sec
    The bandwidth is 610500000.000000byte/sec
```

# references

[测试内存读取速率](https://blog.csdn.net/weixin_43252268/article/details/105477282)