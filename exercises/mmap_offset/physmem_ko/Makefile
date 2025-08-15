obj-m += physmem.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
	gcc -Wall $(PWD)/mmap_test.c -o $(PWD)/mmap_test
	gcc -Wall $(PWD)/ioctl_test.c -o $(PWD)/ioctl_test
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm $(PWD)/mmap_test
	rm $(PWD)/ioctl_test
