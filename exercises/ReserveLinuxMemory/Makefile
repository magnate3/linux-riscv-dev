# Makefile for driver
# Derived from: 
#   http://www.opensourceforu.com/2010/12/writing-your-first-linux-driver/
# with some settings from Robert Nelson's BBB kernel build script
# if KERNELRELEASE is defined, we've been invoked from the
# kernel build system and can use its language.
ifneq (${KERNELRELEASE},)
obj-m :=simple.o
# Otherwise we were called directly from the command line.
# Invoke the kernel build system.
else
KERNEL_SOURCE := /Kernel/linux-4.4.0/
PWD := $(shell pwd)
# Linux kernel 4.4 (which has cape manager support)
#CC=${HOME}/cmpt433/work/bb-kernel/dl/gcc-linaro-5.3.1-2016.05-
#x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-
CC=/usr/bin/
#BUILD=bone14
CORES=4
image=zImage
PUBLIC_DRIVER_PWD=/modules
default:
# Trigger kernel build for this module
	${MAKE} -C ${KERNEL_SOURCE} SUBDIRS=${PWD} -j${CORES} \
	LOCALVERSION=-${BUILD} CROSS_COMPILE=${CC} ${address} \

# copy result to public folder
	cp *.ko ${PUBLIC_DRIVER_PWD}

clean:
	${MAKE} -C ${KERNEL_SOURCE} SUBDIRS=${PWD} clean
endif