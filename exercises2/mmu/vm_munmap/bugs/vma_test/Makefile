obj-m := 
obj-m +=   vma_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
	gcc app.c -o app
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean
	rm app
