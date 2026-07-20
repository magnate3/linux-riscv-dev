MODULE_NAME1 :=  udp_tunnel
MODULE_NAME2 :=  ip6_udp_tunnel
obj-m := $(MODULE_NAME1).o
obj-m += $(MODULE_NAME2).o
OBJ_LIST1 := \
	 udp_tunnel.o \
OBJ_LIST2 := \
	 ip6_udp_tunnel.o \
$(MODULE_NAME1)-y := $(OBJ_LIST1)
$(MODULE_NAME2)-y := $(OBJ_LIST2)

ccflags-y := -O2
ccflags-y                               += -I$(src)
KERNELDIR := /lib/modules/$(shell uname -r)/build

all: nvmedriver

nvmedriver:
	make -C $(KERNELDIR) M=$(PWD) modules
clean:
	make -C $(KERNELDIR) M=$(PWD) clean
