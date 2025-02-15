# 13 ioctl in Linux kernel module
## Build
```
$ cd 13_ioctl
$ make
```

## Load kernel modules and check it
```
$ sudo insmod ioctl_example.ko
$ cat /proc/devices | grep ioctl
 90 ioctl_example
```

## Create device file
```
$ sudo mknod /dev/my_device c 90 0
```
Major device number is 90 and minor device number is 0.
```
$ ls -al /dev/my_device
crw-r--r-- 1 root root 90, 0 Jun 25 21:05 /dev/my_device
```

## Open and close device file from test c program
The callback for open/close is called by open/close the device file from the c program.
```
$ gcc test.c -o test
$ ./test
RD_VALUE - answer: 123
WR_VALUE and RD_VALUE - answer: 456
GREETER
succeed to open
```
The value is stored as below:
```
$ ./test
RD_VALUE - answer: 456
WR_VALUE and RD_VALUE - answer: 456
GREETER
succeed to open
```
```
$ dmesg | tail
...
[53815.124417] ioctl_example open was called
[53815.124426] ioctl_example - the answer copied: 456
[53815.124536] ioctl_example - updated to the answer: 456
[53815.124540] ioctl_example - the answer copied: 456
[53815.124546] ioctl_example - 7 greets to LKM
[53815.124551] ioctl_example close was called
```

## Remove kernel module
```
$ sudo rmmod ioctl_example.ko
```

# my test


```
root@ubuntu:~/ubuntu/13_ioctl# mknod /dev/my_device c 90 0
root@ubuntu:~/ubuntu/13_ioctl#  ls -al /dev/my_device
crw-r--r-- 1 root root 90, 0 Sep  4 11:35 /dev/my_device
root@ubuntu:~/ubuntu/13_ioctl# gcc test.c -o test
root@ubuntu:~/ubuntu/13_ioctl# ./test 
RD_VALUE - answer: 123
WR_VALUE and RD_VALUE - answer: 456
GREETER
succeed to open
root@ubuntu:~/ubuntu/13_ioctl# dmesg | tail -n 10
[14433165.646074] ioctl_example: loading out-of-tree module taints kernel.
[14433165.646179] ioctl_example: module verification failed: signature and/or required key missing - tainting kernel
[14433165.647459] Hello, Linux kernel!
[14433165.647463] ioctl_example - registered Device numver Major: 90, Minor: 0
[14433235.758769] ioctl_example open was called
[14433235.758774] ioctl_example - the answer copied: 123
[14433235.758865] ioctl_example - updated to the answer: 456
[14433235.758867] ioctl_example - the answer copied: 456
[14433235.758877] ioctl_example - 7 greets to LKM
[14433235.758893] ioctl_example close was called
```