obj-m := 
obj-m +=    preempt_count_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

