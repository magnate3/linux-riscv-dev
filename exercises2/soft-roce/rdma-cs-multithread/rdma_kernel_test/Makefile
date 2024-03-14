obj-m += crdma.o
KBUILD_EXTRA_SYMBOLS = /usr/src/ofa_kernel/default/Module.symvers
all:
	make -C /lib/modules/$(shell uname -r)/build M=$$PWD modules

install:
	make -C /lib/modules/$(shell uname -r)/build M=$$PWD modules_install
	depmod -ae

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$$PWD clean
	sudo dmesg -c

remove:
	sudo rmmod crdma

server:
	sudo insmod crdma.ko mode=0

client:
	sudo insmod crdma.ko mode=1


