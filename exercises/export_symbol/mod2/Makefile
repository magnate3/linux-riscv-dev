obj-m:=mod2.o
mod2-y:=mod_b.o

KBUILD_EXTRA_SYMBOLS=/home/gxp/code/os/module/export_symbol/mod1/Module.symvers
KERNELDIR := /lib/modules/$(shell uname -r)/build
PWD:=$(shell pwd)

modules:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

modules_install:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
test:
	@echo "------------------------------------熟悉模块之间的引用"
	@echo "------------------------------------进入mod1目录, make 并且加载mod1模块"
	@echo "---------------------------在加载/make mod2模块之前,都要必须保证mod1已经加载进内核"
	cd ../mod1;pwd;make;sudo insmod mod1.ko
	@echo "------------------------------------再make mod2"
	make
	@echo "------------------------------------加载模块mod1以后,才能加载模块mod2"
	sudo insmod mod2.ko
	@echo "------------------------------------查看内核消息"
	dmesg | tail
	@echo "------------------------------------尝试卸载mod1"
	-sudo rmmod mod1
	@echo "------------------------------------可以看到,mod1被mod2所依赖,不能卸载"
	@echo "------------------------------------按依赖关系卸载"
	sudo rmmod mod2
	sudo rmmod mod1
	@echo "clean两个目录"
	make clean
	cd ../mod1;make clean;
