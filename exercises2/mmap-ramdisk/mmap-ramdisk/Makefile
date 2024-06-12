obj-m += mr.o

all: all-modules mrc

mrc: mrc.c
	$(CC) -o $@ $^
	-sudo chown root.root $@
	-sudo chmod u+s $@

all-modules:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
	@rm -f mrc

.PHONY: load reload unload
reload:
	${MAKE} unload
	${MAKE} load

load: all
	sync
	sync
	sudo insmod mr.ko

unload:
	if lsmod | grep -q '^mr\>' ; then \
		sync ; sync ; \
		sudo rmmod mr ; \
	fi

print-% : ; @echo "$* = $($*)"
