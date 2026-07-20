
# test

```
[root@centos7 preempt]# insmod rt_test.ko 
[root@centos7 preempt]# ls /dev/demo
ls: cannot access /dev/demo: No such file or directory
[root@centos7 preempt]# ls /dev/demo
ls: cannot access /dev/demo: No such file or directory
[root@centos7 preempt]# mknod /dev/demo c 400 0
[root@centos7 preempt]# ls /dev/demo
/dev/demo
[root@centos7 preempt]# ./user 
thread1 start time=1654077665
thread2 start
thread1 stop time=1654077668
thread2 stop
end test
```
**  3= 1654077668 - 1654077665 **