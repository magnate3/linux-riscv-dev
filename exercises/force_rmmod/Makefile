# If KERNELRELEASE is defined, we've been invoked from the
# kernel build system and can use its language.
ifneq ($(KERNELRELEASE),)
	obj-m := force_rmmod.o        
else
	PWD := $(shell pwd)			#PWD为当前目录
	KDIR := /lib/modules/$(shell uname -r)/build
all:
	$(MAKE) -C $(KDIR) M=$(PWD)	#调用内核模块编译
	insmod force_rmmod.ko modname=f2fs
clean:
#	rm -f *.cmd *.o *.mod *.ko
	rm -rf .*.cmd *.o *.mod.c *.ko .tmp_versions
#	$(MAKE) -C $(KDIR) M=$(PWD) clean
endif

#insmod force_rmmod.ko modname=f2fs
#  查看是否加载成功, `exit` 函数是否正常替换
#dmesg | tail -l
#  卸载 `createoops` 驱动
#sudo rmmod f2fs
#  卸载 `force_rmmod` 驱动
#sudo rmmod force_rmmod
#insmod force_rmmod.ko modname=f2fs