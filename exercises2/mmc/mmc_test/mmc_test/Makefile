# Set this variable with the path to your kernel.
# Don't use /usr/src/linux if you're cross-compiling...
CONFIG_MODULE_SIG=n
MYY_KERNEL_DIR ?= /root/lotus2/linux-5.14.12/
#MYY_KERNEL_DIR ?= /root/lotus2/linux-5.14.12

ARCH ?= riscv

ccflags-y += -I${MYY_KERNEL_DIR}/drivers/mmc/core
CROSS_COMPILE ?= riscv64-linux-gnu-
obj-m += dma_test.o
obj-m += mmc_test.o
obj-m += mmc_spi.o
obj-m += mmc_block.o
obj-m += spi-dw-mmio.o
mmc_block-y := queue.o block.o

all:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR)  modules

clean:
	make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR) clean

