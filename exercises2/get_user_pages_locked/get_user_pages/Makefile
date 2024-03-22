obj-m := 
obj-m +=   get_user_pages_test.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
	gcc test.c -o test
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean
	rm test

