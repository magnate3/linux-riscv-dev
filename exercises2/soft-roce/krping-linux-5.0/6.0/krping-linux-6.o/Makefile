KERNELDIR=/lib/modules/`uname -r`/build
KOBJ=/lib/modules/`uname -r`/build
KERNELDIR := /work/linux-6.3.2


obj-m += rdma_krping.o
rdma_krping-y			:= getopt.o krping.o

default:
	make -C $(KERNELDIR) M=`pwd` modules

install:
	make -C $(KERNELDIR) M=`pwd` modules_install
	depmod -a

clean:
	rm -f *.o
	rm -f *.ko
	rm -f rdma_krping.mod.c
	rm -f Module.symvers
	rm -f Module.markers
