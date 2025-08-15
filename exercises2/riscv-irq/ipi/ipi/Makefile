obj-m := mymod.o

KERNELDIR ?= /lib/modules/$(shell uname -r)/build
all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules 
	#make LLVM=1 M=$(PWD) -C $(HOME)/build/linux/x86 modules
	#make LLVM=1 ARCH=arm64 M=$(PWD) -C $(HOME)/build/linux/aarch64 modules

user:
	gcc -o user.out -static user.c

clean:
	rm -f *.o *.ko *.mod *.mod.c *.order *.symvers .*.cmd *.out
