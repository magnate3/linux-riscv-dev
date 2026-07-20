obj-m := 
obj-m +=   dma_test.o
obj-m +=   async_test.o
#obj-m +=   mmap_alloc_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

