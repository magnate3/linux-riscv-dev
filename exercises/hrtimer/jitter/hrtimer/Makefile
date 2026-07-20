
MODULE_NAME1 := hrtimer_test
obj-m := $(MODULE_NAME1).o
OBJ_LIST1 := \
	hrtimer_test1.o
$(MODULE_NAME1)-y := $(OBJ_LIST1)

ccflags-y := -O2
#ccflags-y                               += "-D CONFIG_INFINIBAND_USER_MEM"
ccflags-y                               += -I$(src)
#KERNELDIR := /lib/modules/$(shell uname -r)/build
KERNELDIR := /work/linux-6.3.2


all: test

cp:
	scp $(MODULE_NAME1).ko root@192.168.11.44:/root/
test:
	make -C $(KERNELDIR) M=$(PWD) ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu-   modules
clean:
	make -C $(KERNELDIR) M=$(PWD) ARCH=riscv  CROSS_COMPILE=riscv64-linux-gnu-  clean
