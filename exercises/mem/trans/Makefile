
KERNEL_VER = $(shell uname -r)
KERNEL_DIR = /lib/modules/$(KERNEL_VER)/build

PWD = $(shell pwd)


MODULE = addr_trans
obj-m := $(MODULE).o



all:
	make -C $(KERNEL_DIR) M=$(PWD)  modules  

.PHONY:clean
clean:
	make -C $(KERNEL_DIR) M=$(PWD) clean 
