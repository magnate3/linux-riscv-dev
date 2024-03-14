ifeq ($(KERNELRELEASE),)

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

.PHONY: build clean

build:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	rm -rfv *.o *~ core .depend .*.cmd *.ko *.mod.c  modules.order  Module.symvers .tmp_versions
else

$(info Building with KERNELRELEASE = ${KERNELRELEASE})
obj-m := list_test.o
obj-m := list_test2.o

endif
