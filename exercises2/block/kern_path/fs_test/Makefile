obj-m := 
obj-m +=   kern_path_test.o
obj-m +=   dcache_test.o
obj-m +=   super_test.o
obj-m +=   read_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

