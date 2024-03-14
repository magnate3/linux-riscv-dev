KMOD_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

obj-m += kmod_test.o

all: kmod_test.ko libkmod_driver.so test

kmod_test.ko: kmod_test.c kmod.h
	make  -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	#make V=1 -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

insmod:
	sudo insmod ./kmod_test.ko

install: uninstall kmod_test.ko insmod
	sudo mknod /dev/kmod_dev c 144 0

uninstall:
	sudo rm -f /dev/kmod_dev
	sudo rmmod kmod_test || true

libkmod_driver.so: kmod_driver.c kmod_driver.h
	gcc -O2 -o libkmod_driver.so kmod_driver.c -shared -fpic

test: test.c libkmod_driver.so
	gcc -O2 test.c -o test -I$(KMOD_DIR) -L$(KMOD_DIR) -Wl,-rpath=$(KMOD_DIR) -lkmod_driver

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -rf libkmod_driver.so test
