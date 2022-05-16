obj-m := 
#obj-m +=  dma_example.o
obj-m +=  mmap_alloc.o
obj-m +=   cma_example.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

