obj-m   := zlog.o
KDIR    := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd) 

KDIR="~/src/kernel/3.10.0-957.el7.x86_64"
#KDIR="~/src/kernel/linux-3.10.0-1127.el7"
#KDIR="~/src/kernel/linux-4.18.0-240.22.1.el8_3"

DEBUG ?= 1
ifeq ($(DEBUG), 1)
	ccflags-y=-DDEBUG -g3
endif

  
all:
	ctags -R
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) modules

up:
	#sudo rmmod ring_buf_drv
	#sudo insmod ./ring_buf_drv.ko
	#sshpass -proot123 scp -o StrictHostKeyChecking=no ./zlog.ko root@2.2.2.111:~/
	scp -o StrictHostKeyChecking=no ./zlog.ko root@2.2.2.111:~/
	ssh -o StrictHostKeyChecking=no root@2.2.2.111 'rmmod zlog; insmod ~/zlog.ko'

clean:
	rm -f *.o
	rm -f zlog.mod.c Module.symvers  zlog.ko  zlog.mod.o  tags  modules.order 
