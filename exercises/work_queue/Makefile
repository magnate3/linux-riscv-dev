obj-m := 
obj-m +=   workqueue_test.o
obj-m +=   workqueue_test2.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

