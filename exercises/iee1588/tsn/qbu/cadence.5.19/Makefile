# Set this variable with the path to your kernel.
# Don't use /usr/src/linux if you're cross-compiling...
MYY_KERNEL_DIR ?= /home/ubuntu/kernel-ubuntu/fpga95/linux-5.14

# If you're compiling for ARM64, this will be arm64
ARCH ?= riscv

CROSS_COMPILE ?= riscv64-linux-gnu-
obj-m +=  macb.o
macb-y := macb_ptp.o  macb_main.o
EXTRA_CFLAGS += -DCONFIG_MACB_USE_HWSTAMP
all:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR)  modules

clean:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR) clean

