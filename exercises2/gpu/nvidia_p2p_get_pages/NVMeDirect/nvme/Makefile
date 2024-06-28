ifdef SYSSRC
 KERNEL_SOURCES	 = $(SYSSRC)
else
 KERNEL_UNAME	:= $(shell uname -r)
 KERNEL_SOURCES	 = /lib/modules/$(KERNEL_UNAME)/build
endif

obj-y		+= host/

#NVIDIA_SRC := /usr/src/nvidia-367.48
#NVIDIA_SRC := /usr/src/nvidia-375-375.66
#NVIDIA_SRC := /usr/src/nvidia-450.51.06/
#NVIDIA_SRC := /usr/src/nvidia-460.32.03/
NVIDIA_SRC := /usr/src/nvidia-410.48/
default: modules
.PHONY: default
install: modules_install
	update-initramfs -u
	mkdir -p /usr/include/nvmed
	cp host/nv_pinbuf.h /usr/include/nvmed

.PHONY: install


nvidia-syms:
	make -C $(KERNEL_SOURCES) SUBDIRS=$(NVIDIA_SRC) modules

.PHONY:


%::
	$(MAKE) -C $(KERNEL_SOURCES) \
        KBUILD_EXTRA_SYMBOLS=$(NVIDIA_SRC)/Module.symvers \
	    NVIDIA_SRC=$(NVIDIA_SRC) \
        M=$$PWD $@

#	cp donard-nvme-blacklist.conf /etc/modprobe.d
#	cp 80-donard.rules /etc/udev/rules.d
