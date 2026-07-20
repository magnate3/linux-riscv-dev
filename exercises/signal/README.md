
# insmod signal_allow_test.ko
```
[root@centos7 signal]# insmod signal_allow_test.ko
```

```
[root@centos7 test]# ps -eLf | grep  kthr_sig
root      44535      2  44535  0    1 04:51 ?        00:00:00 [kthr_sig_3]
root      44547  43768  44547  0    1 04:52 pts/0    00:00:00 grep --color=auto kthr_sig
[root@centos7 test]# kill -9  44535
[root@centos7 test]# ps -eLf | grep  kthr_sig
root      44549  43768  44549  0    1 04:52 pts/0    00:00:00 grep --color=auto kthr_sig
[root@centos7 test]#
```

## send SIGUSR1

```
[root@centos7 ~]# ps -eLf | grep kthr_sig
root       7788      2   7788  0    1 05:46 ?        00:00:00 [kthr_sig_3]
root       7842   7789   7842  0    1 05:49 pts/2    00:00:00 grep --color=auto kthr_sig
[root@centos7 ~]# kill -10  7788 
[root@centos7 ~]# ps -eLf | grep kthr_sig
root       7788      2   7788  0    1 05:46 ?        00:00:00 [kthr_sig_3]
root       7844   7789   7844  0    1 05:49 pts/2    00:00:00 grep --color=auto kthr_sig
[root@centos7 ~]# 
```
##  send  SIGINT   
```
[root@centos7 ~]# ps -eLf | grep kthr_sig
root       7788      2   7788  0    1 05:46 ?        00:00:00 [kthr_sig_3]
root       7844   7789   7844  0    1 05:49 pts/2    00:00:00 grep --color=auto kthr_sig
[root@centos7 ~]# kill -2  7788 
[root@centos7 ~]# ps -eLf | grep kthr_sig
root       7788      2   7788  0    1 05:46 ?        00:00:00 [kthr_sig_3]
root       7846   7789   7846  0    1 05:49 pts/2    00:00:00 grep --color=auto kthr_sig
[root@centos7 ~]# 
```

## send SIGKILL

```
[root@centos7 ~]# ps -eLf | grep kthr_sig
root       7788      2   7788  0    1 05:46 ?        00:00:00 [kthr_sig_3]
root       7846   7789   7846  0    1 05:49 pts/2    00:00:00 grep --color=auto kthr_sig
[root@centos7 ~]# kill -9  7788 
[root@centos7 ~]# ps -eLf | grep kthr_sig
root       7856   7789   7856  0    1 05:50 pts/2    00:00:00 grep --color=auto kthr_sig
[root@centos7 ~]# ps -eLf | grep kthr_sig
root       7858   7789   7858  0    1 05:50 pts/2    00:00:00 grep --color=auto kthr_sig
[root@centos7 ~]# 
```
