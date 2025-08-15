# LINUX VERSION: 5.4
PWD := $(shell pwd) 
KVERSION := $(shell uname -r)
KERNEL_DIR = /lib/modules/$(shell uname -r)/build
#KERNEL_DIR = /home/kruztw/Downloads/linux-5.10.1/
#KERNEL_DIR = /usr/src/linux-headers-$(KVERSION)

MODULE_NAME = simple
obj-m := $(MODULE_NAME).o

all:
	make -C $(KERNEL_DIR) M=$(PWD) modules
	@rm .* modules.order Module.symvers $(MODULE_NAME).mod* $(MODULE_NAME).o .simple.* > /dev/null 2>&1 || true
	gcc test.c -o test --static
clean:
	make -C $(KERNEL_DIR) M=$(PWD) clean
