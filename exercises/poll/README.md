
# insmod  example.ko

```
[root@centos7 poll]# insmod  example.ko 
[root@centos7 poll]# ls /dev/example  -al
crw------- 1 root root 240, 0 Nov 21 02:48 /dev/example
[root@centos7 poll]# 
```


##  ./main  /dev/example


```
[root@centos7 test-application]# ./main  /dev/example
Select return 2
Read 12 bytes: Ker3456789

Write 3 bytes: Ker
[root@centos7 test-application]# 
```

```
[root@centos7 poll]# dmesg | grep 'EXAMPLE: poll returned mask'
[971707.137797] EXAMPLE: poll returned mask 0x0, readable 0, writealbe 0 
[971709.194918] EXAMPLE: poll returned mask 0x145, readable 1, writealbe 4 
```