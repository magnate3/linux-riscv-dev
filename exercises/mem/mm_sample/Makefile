SOC_MODE ?= 0

ifeq ($(SOC_MODE), 1)
	KERNEL_DIR = ~/code/linux-4.0.1/
else
	KERNEL_VER = $(shell uname -r)
	KERNEL_DIR = /lib/modules/$(KERNEL_VER)/build
endif
#export ARCH=arm64
#export CROSS_COMPILE=aarch64-linux-gnu-



PWD = $(shell pwd)

MODULE = mm_sample
obj-m := $(MODULE).o

all:
ifeq ($(SOC_MODE), 1)
	make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -C  $(KERNEL_DIR) M=$(PWD) modules
else
	make -C $(KERNEL_DIR) M=$(PWD)  modules
endif

.PHONY:clean
clean:
ifeq ($(SOC_MODE), 1)
	make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -C  $(KERNEL_DIR) M=$(PWD) modules clean
else
	make -C $(KERNEL_DIR) M=$(PWD)  modules clean
endif
