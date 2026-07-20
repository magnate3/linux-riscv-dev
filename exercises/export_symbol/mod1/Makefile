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
