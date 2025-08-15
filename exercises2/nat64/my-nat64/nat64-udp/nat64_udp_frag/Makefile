obj-m :=   nat64_device.o
#obj-m +=   net_device2.o
#obj-m +=   net_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
	gcc udp_cli.c -o udp_cli
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean
	rm udp_cli

