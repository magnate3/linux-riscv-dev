obj-m := sendUDPWithKernelModule.o packetCaptureWithNetfilter.o sendUDPWithKernelModule_sendmsg.o

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

