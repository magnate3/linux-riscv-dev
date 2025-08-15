obj-m := 
obj-m +=   jprobe_test.o
obj-m +=   jprobe_test2.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean
	rm main

