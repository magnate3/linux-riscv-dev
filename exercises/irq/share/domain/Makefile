obj-m := 
obj-m +=   irq_test.o
obj-m +=   msi_irq_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

