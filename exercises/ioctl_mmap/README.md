 # insmod simple.ko 
 
 ```
 [root@centos7 ioctl_mmap]# insmod simple.ko 
[root@centos7 ioctl_mmap]# ./test
   0   1   2   3   4   5   6   7   8   9
   0   1   2   3   4   5   6 207   8   9
[root@centos7 ioctl_mmap]# rmmod simple.ko
 ```
# main
```
int main(void)
{
    create_data();
    display_data();
    mmap_data();
    display_data();
    return 0;
}
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/ioctl_mmap/dmesg.png)