obj-m := 
obj-m +=   kthread_example.o
obj-m +=   kthread_example2.o
obj-m +=   kthread_example3.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

