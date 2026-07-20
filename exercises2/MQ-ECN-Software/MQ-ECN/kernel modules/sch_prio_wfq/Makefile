obj-m += sch_prio_wfq.o
sch_prio_wfq-y := main.o params.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
