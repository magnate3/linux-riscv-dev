obj-m := 
#obj-m += test1.o
#obj-m += vmalloc_example.o
#obj-m += kmap_example.o
#obj-m += atomic_example.o
#obj-m +=  kmalloc_example.o
obj-m +=  slab.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

