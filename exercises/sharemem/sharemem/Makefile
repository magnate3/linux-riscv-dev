# Comment/uncomment the following line to disable/enable debugging


EXTRA_CFLAGS += -O2
#CFLAGS += -I$(LDDINC)

ifneq ($(KERNELRELEASE),)
# call from kernel build system

scull-objs := sharemem.c

obj-m	:= sharemem.o

else

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD       := $(shell pwd)

modules:
	#$(MAKE) -C $(KERNELDIR) M=$(PWD) LDDINC=$(PWD)/../include modules 
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules 
	@echo "================================="
	make -f MakeUtils

endif



clean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions module* Module*
	make -f MakeUtils clean

depend .depend dep:
	$(CC) $(CFLAGS) -M *.c > .depend


ifeq (.depend,$(wildcard .depend))
include .depend
endif
install:
	sudo insmod sharemem.ko
	sudo chmod a+rw /dev/sharemem
	sudo dmesg -c
uninstall:
	sudo rmmod sharemem
