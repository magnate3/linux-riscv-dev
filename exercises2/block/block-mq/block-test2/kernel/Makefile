obj-m := 
obj-m +=   ram-disk.o
#EXTRA_CFLAGS = -m64
EXTRA_CFLAGS = -Wall -g -Wno-unused
EXTRA_CFLAGS += -g -O0
all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

