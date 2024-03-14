obj-m := 
obj-m +=   test_hmm.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
	#gcc mmap_test.c  -o mmap_test
	gcc -g user.c  -o user
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean
	rm user

