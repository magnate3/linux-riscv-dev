obj-m := 
obj-m +=    mmap-test1.o
obj-m +=    mmap-test2.o
obj-m +=    mmap-test3.o
obj-m +=    mmap-test4.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

