
#  test0.c
```
[root@centos7 mmap]# gcc test0.c -o test
[root@centos7 mmap]# ./test 
mapped addres 0xffffffffffffffff and goal_addr 0xfffff8a04bdc  ////////map failed
[root@centos7 mmap]#  
```
# test1.c
```
[root@centos7 mmap]# gcc test1.c -o test
[root@centos7 mmap]# ./test 
mapped addres 0x10000 and goal_addr 0x10000
mapped addres 0x10000 and goal_addr 0x10000
read byte 0 to location 0x10000
read byte 1 to location 0x10001
read byte 2 to location 0x10002
read byte 3 to location 0x10003
read byte 4 to location 0x10004
read byte 5 to location 0x10005
read byte 6 to location 0x10006
read byte 7 to location 0x10007
read byte 8 to location 0x10008
read byte 9 to location 0x10009
[root@centos7 mmap]# 
```

