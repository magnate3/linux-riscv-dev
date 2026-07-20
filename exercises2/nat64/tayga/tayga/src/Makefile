# Driver module

KERNSRC ?= /lib/modules/$(shell uname -r)/build
##EXTRA_CFLAGS = -g

obj-m := tayga.o
tayga-objs := netdev.o nat64.o params.o addrmap.o

default: tayga.ko
	@echo "Example setup:"
	@echo "sudo ../tools/setup-nat64.sh 64:ff9b:0:0::/64 10.40.0.0/16"

tayga.ko: netdev.c tayga.h nat64.c params.c addrmap.c
	$(MAKE) -C $(KERNSRC) M=$(shell pwd) modules

clean:
	rm -rf .tmp_versions *.o *.ko .*.cmd .*.d *.mod *.mod.c *.symvers modules.order .cache.mk *\~

install: tayga.ko
	cp -f tayga.ko /lib/modules/$(shell uname -r)/kernel/
	depmod

