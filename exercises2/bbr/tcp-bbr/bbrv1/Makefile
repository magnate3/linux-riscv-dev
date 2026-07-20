obj-m +=  tcp_bbr.o
#obj-m +=  tcp_bbr2.o

all: 
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	rm -f *.c~
	#rm  tcp_bbr.o tcp_bbr.ko

