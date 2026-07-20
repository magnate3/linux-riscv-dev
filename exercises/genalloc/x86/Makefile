obj-m := 
obj-m +=   genalloc_test.o
obj-m +=   genalloc_test2.o
obj-m +=   genalloc_test3.o
obj-m +=   genalloc_test4.o
EXTRA_CFLAGS=-g -O0
all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

