

# run

```
[root@centos7 test]# gcc nice.c  -o nice_test -lm
[root@centos7 test]# ./nice_test 1

nice=  1 time= 2.65475 secs pid=24228  t_cpu=2.64684e+06  t_sleep=   18.72  nsched=    6  avg timeslice =   441140
[root@centos7 test]# ./nice_test -10

nice=-10 time= 2.65478 secs pid=24261  t_cpu=2.65158e+06  t_sleep=   20.08  nsched=    6  avg timeslice =   441929
[root@centos7 test]# ./nice_test -20

nice=-20 time= 2.65477 secs pid=24262  t_cpu=2.64679e+06  t_sleep=   23.33  nsched=    7  avg timeslice =   378112
[root@centos7 test]# ./nice_test 10

nice= 10 time= 2.65475 secs pid=24272  t_cpu=2.6546e+06  t_sleep=   19.08  nsched=    6  avg timeslice =   442434
[root@centos7 test]# ./nice_test 20

nice= 20 time= 2.65475 secs pid=24273  t_cpu=2.64684e+06  t_sleep=   17.26  nsched=    6  avg timeslice =   441140
```

由结果可以看出当nice的值越大的时候，其睡眠时间越短，则表示其优先级升高了。