MODULE_NAME := dma_buf_exporter_kmd_intel

CONFIG_MODULE_SIG=n
CONFIG_MODULE_SIG_ALL=n

obj-m += $(MODULE_NAME).o
$(MODULE_NAME)-y := dma_buf_exporter_kmd.o

PWD   := $(shell pwd)
KDIR  ?= /lib/modules/$(shell uname -r)/
BUILD ?=$(KDIR)/build

CPPFLAGS += -include $(BUILD)/include/generated/autoconf.h
EXTRA_CFLAGS += -Wall -DDEBUG

all: install clean
	modprobe $(MODULE_NAME)

dma_buf_exporter_kmd.ko:
	$(MAKE) -C $(BUILD) M=$(PWD) modules

build: dma_buf_exporter_kmd.ko

install: build
	mkdir -p /lib/modules/$(shell uname -r)/kernel/drivers/dma_buf_exporter_kmd/
	cp $(MODULE_NAME).ko /lib/modules/$(shell uname -r)/kernel/drivers/dma_buf_exporter_kmd/
	depmod

uninstall:
	modprobe -r $(MODULE_NAME)
	rm /lib/modules/$(shell uname -r)/kernel/drivers/dma_buf_exporter_kmd/$(MODULE_NAME).ko
	depmod

load: build
	modprobe dma_buf_exporter_kmd
	insmod $(MODULE_NAME).ko

unload:
	rmmod $(MODULE_NAME)

clean:
	$(MAKE) -C $(BUILD) M=$(PWD) clean
