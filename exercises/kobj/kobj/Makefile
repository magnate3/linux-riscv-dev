obj-m := 
obj-m +=   kobj_test.o
obj-m +=   dev_test.o
obj-m +=   dev_test2.o
obj-m +=   dev_test3.o
# core dump
obj-m +=   dev_test4.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

