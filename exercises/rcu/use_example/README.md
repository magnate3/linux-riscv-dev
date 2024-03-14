
# insmod  test_sample.ko 

```
[root@centos7 use_example]# insmod  test_sample.ko 
[root@centos7 use_example]# dmesg | tail -n 10
[ 1570.245115] myrcu_del: a=36
[ 1570.247897] myrcu_del: a=37
[ 1570.250679] myrcu_del: a=38
[ 1570.253465] myrcu_del: a=39
[ 1570.256247] myrcu_del: a=40
[ 1570.259029] myrcu_del: a=41
[ 1570.261815] myrcu_del: a=42
[ 1570.291217] myrcu_del: a=43
[ 3933.140888] before call rcu:foo2
[ 3934.220149] after call rcu:hoge3
[root@centos7 use_example]# rmmod  test_sample.ko 
[root@centos7 use_example]# 
```