MODULE_NAME1 :=  exporter-test
MODULE_NAME2 :=  importer-test
obj-m := $(MODULE_NAME1).o
obj-m += $(MODULE_NAME2).o
OBJ_LIST1 := exporter-page.o 
$(MODULE_NAME1)-y := $(OBJ_LIST1)
OBJ_LIST2 :=   importer-page.o 
$(MODULE_NAME2)-y := $(OBJ_LIST2)

ccflags-y := -O2
ccflags-y                               += -I$(src)
KERNELDIR := /lib/modules/$(shell uname -r)/build

all: nvmedriver

nvmedriver:
	make -C $(KERNELDIR) M=$(PWD) modules
clean:
	make -C $(KERNELDIR) M=$(PWD) clean
