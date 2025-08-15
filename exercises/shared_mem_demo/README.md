# shared_mem_demo
A demo(proof) which communicate between userspace and kernel(map kernel memory to userspace)

# kmalloc
```
buffer = (unsigned char *)kmalloc(PAGE_SIZE,GFP_KERNEL);
```

# output
```
root@eBPF:~/shared_mem_demo# make
make -C /lib/modules/4.15.0-20-generic/build M=/root/shared_mem_demo modules
make[1]: Entering directory '/usr/src/linux-headers-4.15.0-20-generic'
  CC [M]  /root/shared_mem_demo/kernel_module.o
/root/shared_mem_demo/kernel_module.o: warning: objtool: mmap_drv_mmap()+0x72: sibling call from callable instruction with modified stack frame
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /root/shared_mem_demo/kernel_module.mod.o
  LD [M]  /root/shared_mem_demo/kernel_module.ko
make[1]: Leaving directory '/usr/src/linux-headers-4.15.0-20-generic'
root@eBPF:~/shared_mem_demo#
root@eBPF:~/shared_mem_demo#
root@eBPF:~/shared_mem_demo# insmod kernel_module.ko
root@eBPF:~/shared_mem_demo#
root@eBPF:~/shared_mem_demo# cat /proc/devices  | grep evt_map
243 evt_map
root@eBPF:~/shared_mem_demo#
root@eBPF:~/shared_mem_demo# mknod /dev/evt_map c 243 0
root@eBPF:~/shared_mem_demo#
root@eBPF:~/shared_mem_demo# gcc userspace.c -o userspace
root@eBPF:~/shared_mem_demo#
root@eBPF:~/shared_mem_demo# ./userspace
reverse shell event{
	'evt':'rvshell',
	'pid':'19256',
	'exe':'/bin/bash',
	'cmdline':'bash',
	'cwd':'/root/Felicia',
	'ppid':'19255',
	'pexe':'/usr/bin/socat'
}
root@eBPF:~/shared_mem_demo#
```

```
kernel_module.c  kernel_module.ko  kernel_module.mod.c  kernel_module.mod.o  kernel_module.o  Makefile  modules.order  Module.symvers  userspace.c
[root@centos7 shared_mem_demo]# insmod kernel_module.ko
[root@centos7 shared_mem_demo]# ls
kernel_module.c  kernel_module.ko  kernel_module.mod.c  kernel_module.mod.o  kernel_module.o  Makefile  modules.order  Module.symvers  userspace.c
[root@centos7 shared_mem_demo]# cat Makefile 
obj-m := kernel_module.o
PWD := $(shell pwd)
all: 
        make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
        rm -rf *.o *~ core .*.cmd *.mod.c ./tmp_version Module.symvers *.ko modules.order[root@centos7 shared_mem_demo]# 
[root@centos7 shared_mem_demo]# 
[root@centos7 shared_mem_demo]# ls
kernel_module.c  kernel_module.ko  kernel_module.mod.c  kernel_module.mod.o  kernel_module.o  Makefile  modules.order  Module.symvers  userspace.c
[root@centos7 shared_mem_demo]# gcc userspace.c  -o userspace
[root@centos7 shared_mem_demo]# ./userspace 
Fail to open /dev/evt_map. Error:No such file or directory
[root@centos7 shared_mem_demo]# cat /proc/devices  | grep evt_map
241 evt_map
[root@centos7 shared_mem_demo]# ./userspace 
Fail to open /dev/evt_map. Error:No such file or directory
[root@centos7 shared_mem_demo]# mknod /dev/evt_map c 243 0
[root@centos7 shared_mem_demo]# ./userspace 
Fail to open /dev/evt_map. Error:No such device or address
[root@centos7 shared_mem_demo]# mknod /dev/evt_map c 241 0
mknod: ‘/dev/evt_map’: File exists
[root@centos7 shared_mem_demo]# rmnod /dev/evt_map 
-bash: rmnod: command not found
[root@centos7 shared_mem_demo]# rm /dev/evt_map 
rm: remove character special file ‘/dev/evt_map’? y
[root@centos7 shared_mem_demo]# mknod /dev/evt_map c 241 0
[root@centos7 shared_mem_demo]# ./userspace 
reverse shell event{
        'evt':'rvshell',
        'pid':'19256',
        'exe':'/bin/bash',
        'cmdline':'bash',
        'cwd':'/root/Felicia',
        'ppid':'19255',
        'pexe':'/usr/bin/socat'
}tracefs
nodev   securityfs
nodev   sockfs
nodev   dax
nodev   pipefs
nodev   hugetlbfs
nodev   devpts
nodev
[root@centos7 shared_mem_demo]# 
```