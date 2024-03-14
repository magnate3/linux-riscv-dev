学习EXPORT_SYMBOL宏的用法.

编译一个mod1模块, 导出func1函数的信息,
再编译一个mod2模块, 使用func1函数,

# insmod mod1.ko
```
[root@centos7 export_symbol]# cd mod1/
[root@centos7 mod1]# ls
Makefile  mod_a.c
[root@centos7 mod1]# cat Makefile 
obj-m:=mod1.o
mod1-y:=mod_a.o

KERNELDIR := /lib/modules/$(shell uname -r)/build
PWD:=$(shell pwd)

modules:
        $(MAKE) -C $(KERNELDIR) M=$(PWD) modules

modules_install:
        $(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install

clean:
        $(MAKE) -C $(KERNELDIR) M=$(PWD) clean
test:
        sudo insmod mod1.ko
        sudo rmmod mod1
        dmesg | tail -5
[root@centos7 mod1]# make
make -C /lib/modules/4.14.0-115.el7a.0.1.aarch64/build M=/root/programming/OS/module/export_symbol/mod1 modules
make[1]: Entering directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
  CC [M]  /root/programming/OS/module/export_symbol/mod1/mod_a.o
  LD [M]  /root/programming/OS/module/export_symbol/mod1/mod1.o
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /root/programming/OS/module/export_symbol/mod1/mod1.mod.o
  LD [M]  /root/programming/OS/module/export_symbol/mod1/mod1.ko
make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
[root@centos7 mod1]# ls
Makefile  mod1.ko  mod1.mod.c  mod1.mod.o  mod1.o  mod_a.c  mod_a.o  modules.order  Module.symvers
[root@centos7 mod1]# insmod mod1.ko 
```

# mod2

## KBUILD_EXTRA_SYMBOLS
KBUILD_EXTRA_SYMBOLS=/home/gxp/code/os/module/export_symbol/mod1/Module.symvers

## makefile

```
[root@centos7 mod2]# cat Makefile 
obj-m:=mod2.o
mod2-y:=mod_b.o
```

## insmod  mod2.ko

```
[root@centos7 mod1]# cd ../mod2/
[root@centos7 mod2]# make
make -C /lib/modules/4.14.0-115.el7a.0.1.aarch64/build M=/root/programming/OS/module/export_symbol/mod2 modules
make[1]: Entering directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
  CC [M]  /root/programming/OS/module/export_symbol/mod2/mod_b.o
  LD [M]  /root/programming/OS/module/export_symbol/mod2/mod2.o
  Building modules, stage 2.
  MODPOST 1 modules
WARNING: "func1" [/root/programming/OS/module/export_symbol/mod2/mod2.ko] undefined!
  CC      /root/programming/OS/module/export_symbol/mod2/mod2.mod.o
  LD [M]  /root/programming/OS/module/export_symbol/mod2/mod2.ko
make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
[root@centos7 mod2]# insmod  mod2.ko
mod2.ko     mod2.mod.c  mod2.mod.o  mod2.o   

```   


# dmesg | tail -n 10

```
[root@centos7 mod2]# dmesg | tail -n 10
[ 1101.886063] payload:         hello world
[ 1101.889979] send packet by skb success.
[ 1109.856519] testmod kernel module removed!
[335656.857206] sctp: Hash tables configured (bind 8192/8192)
[427083.848904] Module 1, Init!
[427083.851774] In Func: func1...
[427111.112586] mod2: no symbol version for func1
[427111.117490] Module 2, Init!
[427111.120358] In Func: func1...
[427111.123398] In Func: func2...
[root@centos7 mod2]# pwd
/root/programming/OS/module/export_symbol/mod2
[root@centos7 mod2]# 
```

