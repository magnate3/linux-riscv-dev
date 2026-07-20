KVERS ?= $(shell uname -r)
MODNAME = tcp_inigo
MODDIR = net/ipv4

obj-m := $(MODNAME).o
all:
	$(MAKE) -C /lib/modules/$(KVERS)/build M=$(shell pwd) modules

install:
	install -m 0644 $(MODNAME).ko /lib/modules/$(KVERS)/kernel/$(MODDIR)/
	depmod -a

clean:
	rm -rf *.ko *.o *.order *.symvers *.mod.* .*cmd .tmp_versions
