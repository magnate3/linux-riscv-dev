obj-m += kretprobe_nf_rules.o

modules:
	make O=/usr/src/linux-obj/x86_64/default -C /usr/src/linux M=`pwd` modules

clean:
	make O=/usr/src/linux-obj/x86_64/default -C /usr/src/linux M=`pwd` clean
