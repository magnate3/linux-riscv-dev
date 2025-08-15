obj-m   += sample_blk.o
obj-m   += simple_fs.o
obj-m   += ramdisk_test.o
obj-m   += blk_test.o

KDIR    := /lib/modules/$(shell uname -r)/build
#KDIR    := /root/HZero/qemu/linux-4.0
PWD     := $(shell pwd)

default:
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) clean
	rm -f modules.order

