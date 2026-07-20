obj-m += sample.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	        $(MAKE) -C $(KDIR) M=$(PWD) modules

insmod:
	insmod sample.ko
	sleep 1 # Wait device to create
	chmod a+rw /dev/sample

rmmod:
	rmmod  sample

clean:
	        $(MAKE) -C $(KDIR) M=$(PWD) clean

