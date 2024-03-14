obj-m += sample_use2.o
obj-m += sample.o
sample-y += sample_use.o sample_locking.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=`pwd`

clean:
	make -C /lib/modules/$(shell uname -r)/build M=`pwd` clean

