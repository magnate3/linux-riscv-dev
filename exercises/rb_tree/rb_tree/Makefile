ifeq ($(KERNELRELEASE),)

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

.PHONY: build clean

build:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	rm -rfv *.o *~ core .depend .*.cmd *.ko *.mod.c  modules.order  Module.symvers
else

$(info Building with KERNELRELEASE = ${KERNELRELEASE})
obj-m := rb_tree_test.o
obj-m += rb_tree_test2.o
obj-m += rb_tree_test3.o
obj-m += rb_tree_cached_test4.o

endif
