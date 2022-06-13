# os
```
[root@centos7 programming]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 programming]# 
```



# test2
```
[root@centos7 pagemap]# gcc test2.c  -o test2
/tmp/cc5DvZdk.o: In function `main':
test2.c:(.text+0x244): undefined reference to `numa_alloc_onnode'
test2.c:(.text+0x2ac): undefined reference to `numa_alloc_onnode'
test2.c:(.text+0x310): undefined reference to `numa_alloc_local'
test2.c:(.text+0x378): undefined reference to `numa_free'
test2.c:(.text+0x384): undefined reference to `numa_free'
test2.c:(.text+0x390): undefined reference to `numa_free'
collect2: error: ld returned 1 exit status
[root@centos7 pagemap]# gcc test2.c  -o test2 -lnuma
[root@centos7 pagemap]# ./test2
success alloc on node0, vaddress: 0x8caa0000.
-r-------- 1 root root 0 Jun 13 08:11 /proc/self/pagemap
page:0x81000000003fe0ab

the physic address alloc in node0: 0x3fe0ab4000 
success alloc on node0, vaddress: 0x8c6a0000.
-r-------- 1 root root 0 Jun 13 08:11 /proc/self/pagemap
page:0x81000000005fed27

the physic address alloc in node1: 0x5fed274000 
success alloc on local, vaddress: 0x8c2a0000.
-r-------- 1 root root 0 Jun 13 08:11 /proc/self/pagemap
page:0x81000000205feff0

the physic address alloc local: 0x205feff04000 
```