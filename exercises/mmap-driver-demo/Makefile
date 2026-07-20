BASEINCLUDE ?= /lib/modules/`uname -r`/build

mydemodev-objs	:= my_demodev.o		# <模块名>-objs := <目标文件>.o
obj-m			:= mydemodev.o		# obj-m := <模块名>.o

modules:
	$(MAKE) -C $(BASEINCLUDE) M=$(PWD) modules;

test:
	gcc -g -O0 -o test1 test1.c
	gcc -g -O0 -o test2 test2.c
	gcc -g -O0 -o test3 test3.c
	gcc -g -O0 -o test4 test4.c

clean:
	rm -rf *.o *~ core .depend .*.cmd *.ko *.mod.c .tmp_versions Module.* \
		modules.* *.unsigned *.mod Modules.symvers test1 test2 test3 test4

PHONY: modules clean
