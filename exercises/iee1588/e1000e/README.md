


# make CONFIG_E1000E=m  CONFIG_E1000E_HWTS=y  -C  /lib/modules/`uname -r`/build  M=./  -j6 modules

```
make CONFIG_E1000E=m  CONFIG_E1000E_HWTS=y  -C  /lib/modules/`uname -r`/build  M=./  -j6 modules
make: Entering directory '/usr/src/linux-headers-5.13.0-39-generic'
make[1]: *** No rule to make target 'kernel/time/timeconst.bc', needed by 'include/generated/timeconst.h'.  Stop.
make[1]: *** Waiting for unfinished jobs....
make: *** [Makefile:1879: .] Error 2
make: Leaving directory '/usr/src/linux-headers-5.13.0-39-generic'
```


#  CONFIG_E1000E_HWTS
```
root@ubuntux86:/work/linux-5.13.1# grep CONFIG_E1000E .config
CONFIG_E1000E=m
CONFIG_E1000E_HWTS=y
```

# build

## make menuconfig
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/e1000e/meuconfig.png)

##  Skipping BTF generation for /work/e1000e/e1000e.ko due to unavailability of vmlinux

```
root@ubuntux86:/work/e1000e# make
make -C /lib/modules/5.13.0-39-generic/build M=/work/e1000e modules
make[1]: Entering directory '/usr/src/linux-headers-5.13.0-39-generic'
  CC [M]  /work/e1000e/82571.o
  CC [M]  /work/e1000e/ich8lan.o
  CC [M]  /work/e1000e/80003es2lan.o
  CC [M]  /work/e1000e/mac.o
  CC [M]  /work/e1000e/manage.o
  CC [M]  /work/e1000e/nvm.o
  CC [M]  /work/e1000e/phy.o
  CC [M]  /work/e1000e/param.o
  CC [M]  /work/e1000e/ethtool.o
  CC [M]  /work/e1000e/netdev.o
/work/e1000e/netdev.c: In function ‘e1000e_rx_hwtstamp’:
/work/e1000e/netdev.c:537:2: warning: ISO C90 forbids mixed declarations and code [-Wdeclaration-after-statement]
  537 |  struct skb_shared_hwtstamps *hwts = skb_hwtstamps(skb);
      |  ^~~~~~
  CC [M]  /work/e1000e/ptp.o
  LD [M]  /work/e1000e/e1000e.o
  MODPOST /work/e1000e/Module.symvers
  CC [M]  /work/e1000e/e1000e.mod.o
  LD [M]  /work/e1000e/e1000e.ko
  BTF [M] /work/e1000e/e1000e.ko
Skipping BTF generation for /work/e1000e/e1000e.ko due to unavailability of vmlinux
make[1]: Leaving directory '/usr/src/linux-headers-5.13.0-39-generic'
```

## ln -sf /usr/lib/modules/$(uname -r)/vmlinux.xz /boot/

```
root@ubuntux86:/work/e1000e# ls /usr/lib/modules/$(uname -r)/vmlinux.xz
ls: cannot access '/usr/lib/modules/5.13.0-39-generic/vmlinux.xz': No such file or directory
root@ubuntux86:/work/e1000e# ln -sf /usr/lib/modules/$(uname -r)/vmlinux.xz /boot/
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/e1000e/boot.png)




# skb_timestamp


## driver

### e1000e_tx_hwtstamp_work

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/e1000e/tx_hw.png)


###  e1000e_rx_hwtstamp

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/e1000e/rx_hw.png)

## /work/tsn/hwtstamp_test# ./stamp_send enp0s31f6 10.11.11.81
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/e1000e/send.png)


## log

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/e1000e/skb_timestamp.png)