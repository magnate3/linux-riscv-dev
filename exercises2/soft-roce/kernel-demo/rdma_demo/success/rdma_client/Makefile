#KPATH :=/lib/modules/$(shell uname -r)/build
KPATH :=/work/linux-6.3.2
PWD :=$(shell pwd)
obj-m = rdma_client.o

all:
	$(MAKE) -C $(KPATH) M=$(PWD) modules

clean:
	$(MAKE) -C $(KPATH) M=$(PWD) clean
