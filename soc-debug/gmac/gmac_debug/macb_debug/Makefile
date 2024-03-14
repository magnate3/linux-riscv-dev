MYY_KERNEL_DIR ?= /home/ubuntu/kernel-ubuntu/fpga95/linux-5.14.12

ARCH ?= riscv

CROSS_COMPILE ?= riscv64-linux-gnu-
obj-m +=  gmac_macb_debug.o
gmac_macb_debug-y := macb_debug.o 
#EXTRA_CFLAGS += -DCONFIG_MACB_USE_HWSTAMP
all:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR)  modules

clean:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR) clean

