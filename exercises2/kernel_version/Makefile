MODULE_NAME1 := version_test
obj-m := $(MODULE_NAME1).o
OBJ_LIST1 := \
	version.o
$(MODULE_NAME1)-y := $(OBJ_LIST1)

ccflags-y := -O2
#ccflags-y                               += "-D CONFIG_INFINIBAND_USER_MEM"
ccflags-y                               += -I$(src)
#KERNELDIR := /lib/modules/$(shell uname -r)/build
KERNELDIR := /work/linux-6.3.2
#KERNELDIR :=  /lib/modules/6.3.2-060302-generic/build/

all: test

cp:
	scp $(MODULE_NAME1).ko root@192.168.11.44:/root/
test:
	make -C $(KERNELDIR) M=$(PWD) modules
clean:
	make -C $(KERNELDIR) M=$(PWD) clean
