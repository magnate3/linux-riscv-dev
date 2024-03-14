obj-m := 
obj-m +=   sys_test1.o
obj-m +=   sys_test2.o
obj-m +=   sys_test3.o
obj-m +=   sys_test4.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

