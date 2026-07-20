.PHONY: all clean

DEBUG = 0
VER = 1.0.0
KVER = $(shell uname -r)
PWD = $(shell pwd)

EXTRA_CFLAGS := -DDEVICE_VERSION="\"${VER}\""

ifeq ($(DEBUG), 1)
	EXTRA_CFLAGS += -DDEBUG
endif

objs += ktask.o ktask_memcache.o
objs += ktask_hook.o ktask_cache.o
objs += ktask_cmdline.o kpath.o

ifeq ($(KVER),$(shell uname -r))
    obj-m += ktaskcmdline.o
	ktaskcmdline-objs := $(objs)
else
    obj-m += ktaskcmdline-$(KVER).o
	ktaskcmdline-objs := $(objs)
endif

all:
	make -C /lib/modules/$(KVER)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(KVER)/build M=$(PWD) clean
