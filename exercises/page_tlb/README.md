
# insmod get_vm_area_test.ko 

```
[root@centos7 page_table]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 page_table]# 
```

```
[root@centos7 page_table]# insmod get_vm_area_test.ko 
[root@centos7 page_table]# dmesg | tail -n 5
[   44.392537] IPv6: ADDRCONF(NETDEV_UP): docker0: link is not ready
[152541.245164] get_vm_area_test: loading out-of-tree module taints kernel.
[152541.251890] get_vm_area_test: module verification failed: signature and/or required key missing - tainting kernel
[152541.262701] <0>vm->size ：131072
[152541.266096] <0>vm->addr ：0xc0000100
[root@centos7 page_table]# 
```

# reference
https://github.com/ljrcore/linuxmooc/blob/bdaf02620e55bf06e9c84b72afb8ff47e7384447/%E7%B2%BE%E5%BD%A9%E6%96%87%E7%AB%A0/%E6%96%B0%E6%89%8B%E4%B8%8A%E8%B7%AF%EF%BC%9ALinux%E5%86%85%E6%A0%B8%E4%B9%8B%E6%B5%85%E8%B0%88%E5%86%85%E5%AD%98%E5%AF%BB%E5%9D%80.md