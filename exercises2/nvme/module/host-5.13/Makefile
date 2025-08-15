MODULE_NAME1 := nvme_core
MODULE_NAME2 := nvme_pci
obj-m := $(MODULE_NAME1).o
obj-m += $(MODULE_NAME2).o
OBJ_LIST1 := multipath.o ioctl.o zns.o lightnvm.o  hwmon.o trace.o core.o   
$(MODULE_NAME1)-y := $(OBJ_LIST1)
#OBJ_LIST2 := multipath.o  pci.o 
OBJ_LIST2 :=   pci.o 
$(MODULE_NAME2)-y := $(OBJ_LIST2)

ccflags-y := -O2
ccflags-y                               += -I$(src)
KERNELDIR := /lib/modules/$(shell uname -r)/build

all: nvmedriver

nvmedriver:
	make -C $(KERNELDIR) M=$(PWD) modules
clean:
	make -C $(KERNELDIR) M=$(PWD) clean
