obj-m += sch_prio_dwrr.o
sch_prio_dwrr-y := main.o params.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
