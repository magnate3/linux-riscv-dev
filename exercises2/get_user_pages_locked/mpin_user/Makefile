# SPDX-License-Identifier: GPL-2.0+ WITH Linux-syscall-note
#
# Makefile & Kbuild file for building installing the mpin_user.ko driver
# options make, make install, make clean ...

MPIN_USER_VERSION ?= 1.0.1

#default is to compile against the running kernel (kernel-devel installed on the system)
# do 'make KERNEL_PATH=/home/user/my_kernel_path/ to compile against an alternative path
KERNEL_PATH ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

# ~~~ This Part is the Kbuild side
EXTRA_CFLAGS = -Wall -I$(PWD) -D "MPIN_USER_VERSION=$(MPIN_USER_VERSION)"

ifdef MPIN_NO_WARN
EXTRA_CFLAGS += -Werror
endif

obj-m := mpin_user.o

# ~~~ This part is so we can use make inside the source dir to make below target options
#
# this is the STD way to compile an out of tree Kernel module
all: mpin_user.c mpin_user.h Makefile
	make -C $(KERNEL_PATH) M=$(PWD) modules

# Kernel's modules_install is suppose to also do depmod
install:
	make -C $(KERNEL_PATH) M=$(PWD) modules_install

clean:
	make -C $(KERNEL_PATH) M=$(PWD) clean
	rm mpin_user-test

test:
	gcc -Wall -O2 -D_GNU_SOURCE  -o mpin_user-test mpin_user-test.c
