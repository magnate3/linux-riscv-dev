obj-m := 
obj-m +=   mmu_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
	gcc mmap_test.c  -o mmap_test
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean
	rm mmap_test

