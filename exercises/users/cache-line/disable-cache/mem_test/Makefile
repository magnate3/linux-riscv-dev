obj-m := 
obj-m +=   disableCache.o
EXTRA_CFLAGS = -m64
EXTRA_CFLAGS += -g -O0
all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

