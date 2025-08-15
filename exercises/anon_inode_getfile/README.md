
# ./ioctl 
```
make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
[root@centos7 anon_inode_getfile]# insmod anon_inode_getfile.ko 
[root@centos7 anon_inode_getfile]# ./ioctl 
version = 110
fd = 4
fd = 5
total 0
dr-x------ 2 root root  0 Jul 14 08:20 .
dr-xr-xr-x 9 root root  0 Jul 14 08:20 ..
lrwx------ 1 root root 64 Jul 14 08:20 0 -> /dev/pts/0
lrwx------ 1 root root 64 Jul 14 08:20 1 -> /dev/pts/0
lrwx------ 1 root root 64 Jul 14 08:20 2 -> /dev/pts/0
l-wx------ 1 root root 64 Jul 14 08:20 3 -> /dev/hiboma
lrwx------ 1 root root 64 Jul 14 08:20 4 -> anon_inode:hiboma-anon
lrwx------ 1 root root 64 Jul 14 08:20 5 -> anon_inode:hiboma-anon

```

#  ls -hal /proc/53930/fd
```
[root@centos7 ~]# ps -elf | grep ioctl
0 S root      53930  52594  0  80   0 -    39 wait_w 08:20 pts/0    00:00:00 ./ioctl
0 S root      53966  53941  0  80   0 -  1729 pipe_w 08:20 pts/2    00:00:00 grep --color=auto ioctl
[root@centos7 ~]# ls -hal /proc/53930/fd
total 0
dr-x------ 2 root root  0 Jul 14 08:20 .
dr-xr-xr-x 9 root root  0 Jul 14 08:20 ..
lrwx------ 1 root root 64 Jul 14 08:20 0 -> /dev/pts/0
lrwx------ 1 root root 64 Jul 14 08:20 1 -> /dev/pts/0
lrwx------ 1 root root 64 Jul 14 08:20 2 -> /dev/pts/0
l-wx------ 1 root root 64 Jul 14 08:20 3 -> /dev/hiboma
lrwx------ 1 root root 64 Jul 14 08:20 4 -> anon_inode:hiboma-anon
lrwx------ 1 root root 64 Jul 14 08:20 5 -> anon_inode:hiboma-anon
[root@centos7 ~]# 

```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/anon_inode_getfile/fd.png)

# ./ioctl2 
```
[root@centos7 anon_inode_getfile]# ls
anon_inode_getfile.c  ioctl  ioctl2.c  ioctl.c  Makefile
[root@centos7 anon_inode_getfile]# gcc ioctl2.c  -o ioctl2
[root@centos7 anon_inode_getfile]# ./ioctl2 
version = 110
fd = 4
fd = 5
total 0
dr-x------ 2 root root  0 Jul 14 23:20 .
dr-xr-xr-x 9 root root  0 Jul 14 23:20 ..
lrwx------ 1 root root 64 Jul 14 23:20 0 -> /dev/pts/0
lrwx------ 1 root root 64 Jul 14 23:20 1 -> /dev/pts/0
lrwx------ 1 root root 64 Jul 14 23:20 2 -> /dev/pts/0
l-wx------ 1 root root 64 Jul 14 23:20 3 -> /dev/hiboma
lrwx------ 1 root root 64 Jul 14 23:20 4 -> anon_inode:hiboma-anon
lrwx------ 1 root root 64 Jul 14 23:20 5 -> anon_inode:hiboma-anon

```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/anon_inode_getfile/fd2.png)