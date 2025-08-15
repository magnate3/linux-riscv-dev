# make 参数的说明

```
make 参数的说明：
$(MAKE) -C $(KDIR) M=$(PWD) modules
-C：后面的参数为linux内核的顶层目录
M：后面的参数为需要编译文件的目录
```
#  CONFIG_XXXX

```
root@ubuntu:~/linux-5.15.24-rt/linux-5.15.24/drivers/net/ethernet/cadence# cat Makefile 
# SPDX-License-Identifier: GPL-2.0
#
# Makefile for the Atmel network device drivers.
#
macb-y  := macb_main.o

ifeq ($(CONFIG_MACB_USE_HWSTAMP),y)
macb-y  += macb_ptp.o
endif

obj-$(CONFIG_MACB) += macb.o
obj-$(CONFIG_MACB_PCI) += macb_pci.o
```

# compile directory

```
root@ubuntu:~/linux-5.15.24-rt/linux-5.15.24# pwd
/root/linux-5.15.24-rt/linux-5.15.24
root@ubuntu:~/linux-5.15.24-rt/linux-5.15.24# ls
arch   COPYING  Documentation  include  Kbuild   k_install  localversion-rt  mm                       modules-only.symvers  net                       samples   sifive_rt.patch.20220314  tools  vmlinux
block  CREDITS  drivers        init     Kconfig  lib        MAINTAINERS      modules.builtin          modules.order         patch-5.15.24-rt31.patch  scripts   sound                     usr    vmlinux.o
certs  crypto   fs             ipc      kernel   LICENSES   Makefile         modules.builtin.modinfo  Module.symvers        README                    security  System.map                virt   vmlinux.symvers
root@ubuntu:~/linux-5.15.24-rt/linux-5.15.24# 
```

#  make

```
 make CONFIG_MACB=m -C ~/linux-5.15.24-rt/linux-5.15.24  M=drivers/net/ethernet/cadence ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j64 modules
make: Entering directory '/root/linux-5.15.24-rt/linux-5.15.24'
  CC [M]  drivers/net/ethernet/cadence/macb_main.o
  LD [M]  drivers/net/ethernet/cadence/macb.o
  MODPOST drivers/net/ethernet/cadence/Module.symvers
  CC [M]  drivers/net/ethernet/cadence/macb.mod.o
  LD [M]  drivers/net/ethernet/cadence/macb.ko
make: Leaving directory '/root/linux-5.15.24-rt/linux-5.15.24'
root@ubuntu:~/linux-5.15.24-rt/linux-5.15.24# ls  drivers/net/ethernet/cadence 
built-in.a  Kconfig  macb.h  macb.ko  macb_main.c  macb_main.o  macb.mod  macb.mod.c  macb.mod.o  macb.o  macb_pci.c  macb_ptp.c  Makefile  modules.order  Module.symvers
```

# make clean

```
built-in.a  Kconfig  macb.h  macb.ko  macb_main.c  macb_main.c.bak  macb_main.o  macb.mod  macb.mod.c  macb.mod.o  macb.o  macb_pci.c  macb_poll.c  macb_ptp.c  Makefile  modules.order  Module.symvers
root@ubuntu:~/linux-5.15.24-rt/linux-5.15.24# make CONFIG_MACB=m -C ~/linux-5.15.24-rt/linux-5.15.24  M=drivers/net/ethernet/cadence ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j64 clean
make: Entering directory '/root/linux-5.15.24-rt/linux-5.15.24'
  CLEAN   drivers/net/ethernet/cadence/Module.symvers
make: Leaving directory '/root/linux-5.15.24-rt/linux-5.15.24'
root@ubuntu:~/linux-5.15.24-rt/linux-5.15.24# ls  drivers/net/ethernet/cadence
Kconfig  macb.h  macb_main.c  macb_main.c.bak  macb_pci.c  macb_poll.c  macb_ptp.c  Makefile
root@ubuntu:~/linux-5.15.24-rt/linux-5.15.24#
```


# make CONFIG_E1000E=m  CONFIG_E1000E_HWTS=y  -C/lib/modules/`uname -r`/build  M=./  -j6 modules

```
root@ubuntux86:/work/e1000e# make CONFIG_E1000E=m  CONFIG_E1000E_HWTS=y  -C/lib/modules/`uname -r`/build  M=./  -j6 modules
make: Entering directory '/usr/src/linux-headers-5.13.0-39-generic'
make[1]: *** No rule to make target 'kernel/time/timeconst.bc', needed by 'include/generated/timeconst.h'.  Stop.
make[1]: *** Waiting for unfinished jobs....
make: *** [Makefile:1879: .] Error 2
make: Leaving directory '/usr/src/linux-headers-5.13.0-39-generic'
root@ubuntux86:/work/e1000e# 
```