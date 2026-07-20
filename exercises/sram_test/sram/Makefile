obj-m := 
obj-m +=   sram_test.o
obj-m +=   sram_test1.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

