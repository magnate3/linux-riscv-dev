
#  jiffies

As far as I know, "jiffies" in Linux kernel is the number of ticks since boot, and the number of ticks in one second is defined by "HZ", so in theory:

(uptime in seconds) = jiffies / HZ
But based on my tests, the above is not true. For example:

```
$ uname -r
2.6.32-504.el6.x86_64
$ grep CONFIG_HZ /boot/config-2.6.32-504.el6.x86_64
# CONFIG_HZ_100 is not set
# CONFIG_HZ_250 is not set
# CONFIG_HZ_300 is not set
CONFIG_HZ_1000=y
CONFIG_HZ=1000
```
## INITIAL_JIFFIES

It is simply because of

#define INITIAL_JIFFIES ((unsigned long)(unsigned int) (-300*HZ))
The real value of INITIAL_JIFFIES is 0xfffb6c20, if HZ is 1000. It's not 0xfffffffffffb6c20.

So if you want compute uptime from jiffies; you have to do

(jiffies - 0xfffb6c20)/HZ


## jiffies_64

```
/* The jiffies */
static int jiffies_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%llu\n",
            (unsigned long long) get_jiffies_64());
    seq_printf(m, "jiffies=%lu &jiffies=%p (u64)jiffies=%llu (u64)jiffies_64=%llu &jiffies_64=%p get_jiffies_64()=%llu\n", jiffies, &jiffies, (u64)jiffies, (u64)jiffies_64, &jiffies_64, get_jiffies_64());
    return 0;
}
[root@centos7 proj3]# cat /proc/ticks/jiffies 
4399254912
jiffies=4399254912 &jiffies=ffff000008db1a80 (u64)jiffies=4399254912 (u64)jiffies_64=4399254912 &jiffies_64=ffff000008db1a80 get_jiffies_64()=4399254912
[root@centos7 proj3]# 
```

结论：

1. jiffies 的地址和 jiffies_64 是一样的，不同的是在程序中体现的长度不同罢了。也可以看到，链接器对定义在目标文件中的全局变量(同名全局符号)是可见的。 

2. 32位的jiffies容易溢出，在开机300秒回绕。

3. 将jiffies强转成unsigned long long也没有获取jiffies_64的值。

4. 在32位系统上，读取64位变量需要两次内存访问，最好使用get_jiffies_64()避免两次内存访问数据不一致。

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/jiffies/test1.png)

#  uptime

```
[root@centos7 uptime]# insmod  uptime_test.ko 
[root@centos7 uptime]# dmesg | tail -n 10
[978114.707058] jiffies: module license 'Aut' taints kernel.
[978114.712444] Disabling lock debugging due to kernel taint
[978114.718312] /proc/jiffies created
[978164.835513] /proc/jiffies removed
[978207.440713] /proc/seconds created
[978233.585736] /proc/seconds removed
[1043137.890515] Loading ticks module.
[1044703.375298] Directory /proc/ticks removed.
[1044703.379561] Unloading ticks module.
[1045121.621612] System Up-Time: 290 Hours 17418 Minutes 56 Seconds
[root@centos7 uptime]# 
```

# implicit declaration of function ‘do_posix_clock_monotonic_gettime’

```
#define do_posix_clock_monotonic_gettime(ts) ktime_get_ts(ts)

```

You could include that define into some of driver's headers. 
Also, note that ktime_get_ts is defined in linux/timekeeping.h of 4.7.2.

# hrtime.ko 

```
[root@centos7 hrtime]# insmod  hrtime.ko 
[root@centos7 hrtime]# dmesg | tail -n 10
[1058768.665662] <ker-driver>vibe_work_func:msleep(1000)
[1058769.697279] <ker-driver>Time:1058784.47
[1058771.697268] <ker-driver>Time:1058786.47
[1058771.701261] <ker-driver>vibrator_timer_func
[1058771.705602] <ker-driver>vibe_work_func:msleep(1000)
[1058772.737226] <ker-driver>Time:1058787.51
[1058774.737198] <ker-driver>Time:1058789.51
[1058774.741191] <ker-driver>vibrator_timer_func
[1058774.745535] <ker-driver>vibe_work_func:msleep(1000)
[1058775.777183] <ker-driver>Time:1058790.55
```

# references

[Linux时间子系统之三：jiffies](https://www.cnblogs.com/arnoldlu/p/7234443.html)