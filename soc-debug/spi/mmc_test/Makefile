# Set this variable with the path to your kernel.
# Don't use /usr/src/linux if you're cross-compiling...
MYY_KERNEL_DIR ?= /root/linux-5.14.12

ARCH ?= riscv

ccflags-y += -I${MYY_KERNEL_DIR}/drivers/mmc/core
CROSS_COMPILE ?= riscv64-linux-gnu-
obj-m += mmc_test.o
obj-m += mmc_spi.o
#lotus_norm_macb-y := macb_ptp.o macb_main.o

all:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR)  modules

clean:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR) clean

