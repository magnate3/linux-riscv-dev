
```
[root@centos7 kernel]# dmesg | grep  "You must to build with arguments"
[3693245.041569] You must to build with arguments ALLOC=-DUSE_KMALLOC or ALLOC=-DUSE_VMALLOC or ALLOC=-DUSE_ALLOC_PAGES
[root@centos7 kernel]# 
```

# make

```
EXTRA_CFLAGS += -DUSE_ALLOC_PAGES
or
CFLAGS_mymmap.o += -DUSE_ALLOC_PAGES
```

```
[root@centos7 kernel]# cat Makefile 
obj-m := mymmap.o
ccflags-y = -Wno-unused-function -Wno-unused-label -Wno-unused-variable

EXTRA_CFLAGS=$(ALLOC)
CONFIG_MODULE_SIG=n
KDIR := /lib/modules/$(shell uname -r)/build
#EXTRA_CFLAGS += -DUSE_ALLOC_PAGES
CFLAGS_mymmap.o += -DUSE_ALLOC_PAGES
PWD := $(shell pwd)


all: mymmap.c
        @echo "extra_flags:$(EXTRA_CFLAGS)"
        make -C $(KDIR) M=$(PWD) modules

clean:
        make -C $(KDIR) M=$(PWD) clean
[root@centos7 kernel]# 
```

# insmod mymmap.ko 
```
[root@centos7 kernel]# insmod mymmap.ko 
[root@centos7 kernel]# dmesg | tail -n 10
[3697713.613730] [*]unsigned int *NORMAL_int:     Address: 0xd081dc00
[3698882.577664] [mymmap_init] Init module
[3698882.581487] [mymmap_init] used alloc_pages()
[3698882.581488] [mymmap_init] 0xfaceb00c
[root@centos7 kernel]# 
```

# ./mmap-test

```
[root@centos7 user]# make
cc -Wall    -c -o mmap-test.o mmap-test.c
cc -static  mmap-test.o   -o mmap-test
[root@centos7 user]# ./mmap-test 
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c
0xfaceb00c

Write/Read test ...
```

