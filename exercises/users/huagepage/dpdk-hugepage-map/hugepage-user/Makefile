obj-m += hugepage-driver.o
hugepage-driver-y := driver.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all: hugepage-user hugepage-driver

hugepage-user: user.c
	gcc -o hugepage-user user.c -Wall
	
hugepage-driver: driver.c
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	rm -f hugepage-user
	$(MAKE) -C $(KDIR) M=$(PWD) clean