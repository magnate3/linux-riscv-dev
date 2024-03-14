kernel ?= $(shell uname -r)
kdir ?= /lib/modules/$(kernel)/build

obj-m = ptdump.o

all: ptdump.ko ptdump_cli

ptdump_cli: ptdump_cli.c
	$(CROSS_COMPILE)gcc -Wall $< -o $@

ptdump.ko: ptdump.h ptdump.c
	$(MAKE) -C $(kdir) M=$$(pwd)

clean:
	rm -f *.ko *.o ptdump_cli

run: all
	scp ptdump.ko ptdump_cli ptdump_dmesg $(host):
	ssh $(host) \
	    "sudo insmod ./ptdump.ko ; sudo ./ptdump_cli ; sudo rmmod ptdump"
	ssh $(host) "sudo ./ptdump_dmesg"
