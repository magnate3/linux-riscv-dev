


```
[root@centos7 module]# insmod sample.ko 
[root@centos7 module]# cd ..
[root@centos7 gup]# ls
Makefile  module  user  user.c
[root@centos7 gup]# ./user 
Data after read is "CALLED_READ"
Data after write is "CALLED_WRITE"
[root@centos7 gup]# 
```