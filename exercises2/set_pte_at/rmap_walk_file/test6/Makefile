obj-m := 
#obj-m +=   cache_test.o
obj-m +=   hybridmem_test.o
#EXTRA_CFLAGS=-g -O0
#EXTRA_CFLAGS= -g
all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
	gcc -g  mmap_test.c  -o mmap_test
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean
	rm mmap_test

