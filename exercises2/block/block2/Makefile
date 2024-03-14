#obj-m := blk.o simblk.o testblk.o jpr.o
obj-m := blk.o submit.o
blk-objs:= blockdriver.o
submit-objs:= testdriver.o submitbio.o
jpr-objs:= jprobe.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

default:
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) clean
	rm -rf Module.markers modules.order Module.symvers
