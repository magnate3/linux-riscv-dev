ifneq ($(KERNELRELEASE),)
    # kbuild part of makefile
obj-m  := rmem_rdma.o
rmem_rdma-y := log.o rdma_library.o rmem.o
ccflags-y=-I/usr/src/mlnx-ofed-kernel-3.2/include/ -I./init

else
    KDIR ?= /lib/modules/`uname -r`/build

PWD := $(shell pwd)

make:
	rm *.o *.ko *.mod.c;$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) clean

load:
	sudo insmod rmem_rdma.ko npages=100000 servers=10.10.49.98:18515

unload:
	sudo rmmod -f rmem_rdma.ko

endif
