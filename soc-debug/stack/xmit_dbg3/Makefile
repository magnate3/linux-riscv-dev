
MYY_KERNEL_DIR ?= /root/lotus2/linux-5.14.12
ARCH ?= riscv
CROSS_COMPILE ?= riscv64-linux-gnu-
obj-m := 
obj-m +=  icmp_dbg.o
obj-m +=  ping_dbg.o
MKD := mkdir -p $(BUILD_DIR)

all: kernel user

kernel:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR)  modules
clean:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR) clean
user:
	#$(CROSS_COMPILE)gcc  -g  nlcore.c phy_tool.c -o phy_tool 

.PHONY: kernel user clean
