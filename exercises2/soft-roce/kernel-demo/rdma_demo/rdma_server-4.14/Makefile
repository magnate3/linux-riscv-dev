KPATH :=/lib/modules/$(shell uname -r)/build
PWD :=$(shell pwd)
obj-m = rdma_server.o

all:
	$(MAKE) -C $(KPATH) M=$(PWD) modules

clean:
	$(MAKE) -C $(KPATH) M=$(PWD) clean
