obj-m += ioctl_example.o

MYY_KERNEL_DIR ?= /root/linux-5.15.24-rt/linux-5.15.24
#MYY_KERNEL_DIR ?= /root/lotus2/linux-5.14.12
ARCH ?= riscv
CROSS_COMPILE ?= riscv64-linux-gnu-
all:
	        make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR)  modules
		riscv64-linux-gnu-gcc test.c -o test
clean:
	        make ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) M=$(PWD) -C $(MYY_KERNEL_DIR) clean
		rm test
