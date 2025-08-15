


```Shell
root@ubuntu:~/vfio/pci# ls
Kconfig       vfio_pci.c         vfio_pci_intrs.o.ur-safe
Makefile      vfio_pci_config.c  vfio_pci_private.h
Makefile.bak  vfio_pci_intrs.c   vfio_pci_rdwr.
```
模块名vfio-pci.ko 依赖vfio_pci.o    
```
root@ubuntu:~/vfio/pci# cat Makefile

vfio-pci-y := vfio_pci.o vfio_pci_intrs.o vfio_pci_rdwr.o vfio_pci_config.o

obj-$(CONFIG_VFIO_PCI) += vfio-pci.o
KERNELDIR := /lib/modules/$(shell uname -r)/build

all:
        make -C $(KERNELDIR) M=$(PWD) modules
clean:
        make -C $(KERNELDIR) M=$(PWD) clean
```

```Shell
root@ubuntu:~/vfio/pci# make
make -C /lib/modules/3.13.0-170-generic/build M=/root/vfio/pci modules
make[1]: Entering directory `/usr/src/linux-headers-3.13.0-170-generic'
  CC [M]  /root/vfio/pci/vfio_pci.o
  CC [M]  /root/vfio/pci/vfio_pci_intrs.o
  CC [M]  /root/vfio/pci/vfio_pci_rdwr.o
  CC [M]  /root/vfio/pci/vfio_pci_config.o
  LD [M]  /root/vfio/pci/vfio-pci.o
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /root/vfio/pci/vfio-pci.mod.o
  LD [M]  /root/vfio/pci/vfio-pci.ko
make[1]: Leaving directory `/usr/src/linux-headers-3.13.0-170-generic'
```

```Shell
root@ubuntu:~/vfio/pci# ls *ko
vfio-pci.ko
root@ubuntu:~/vfio/pci# 
root@ubuntu:~/vfio/pci# ls *o
vfio-pci.ko     vfio-pci.o  vfio_pci_config.o  vfio_pci_rdwr.o
vfio-pci.mod.o  vfio_pci.o  vfio_pci_intrs.o
root@ubuntu:~/vfio/pci# 


# 方法二

```
root@ubuntu:~/vfio/pci# cat Makefile.bak 
MODULE_NAME1 :=  vfio-pci
obj-m := $(MODULE_NAME1).o
OBJ_LIST1 :=  vfio_pci.o vfio_pci_intrs.o vfio_pci_rdwr.o vfio_pci_config.o
$(MODULE_NAME1)-y := $(OBJ_LIST1)

ccflags-y := -O2
#ccflags-y  += -I$(src)
KERNELDIR := /lib/modules/$(shell uname -r)/build

all: dpdk 

dpdk:
        make -C $(KERNELDIR) M=$(PWD) modules
clean:
        make -C $(KERNELDIR) M=$(PWD) clean
root@ubuntu:~/vfio/pci# make -f Makefile.bak 
make -C /lib/modules/3.13.0-170-generic/build M=/root/vfio/pci modules
make[1]: Entering directory `/usr/src/linux-headers-3.13.0-170-generic'
  CC [M]  /root/vfio/pci/vfio_pci.o
  CC [M]  /root/vfio/pci/vfio_pci_intrs.o
  CC [M]  /root/vfio/pci/vfio_pci_rdwr.o
  CC [M]  /root/vfio/pci/vfio_pci_config.o
  LD [M]  /root/vfio/pci/vfio-pci.o
  Building modules, stage 2.
  MODPOST 1 modules
  CC      /root/vfio/pci/vfio-pci.mod.o
  LD [M]  /root/vfio/pci/vfio-pci.ko
make[1]: Leaving directory `/usr/src/linux-headers-3.13.0-170-generic'
root@ubuntu:~/vfio/pci# 
```

# 多个目录

[多个目录Makefile](https://github.com/septemhill/KernelMakefileSample/blob/master/src/Makefile)