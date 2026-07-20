
#  CONFIG_STRICT_DEVMEM

## arm64

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mmap_pci/tree.png)

```
[root@centos7 boot]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 boot]# 

```

```
[root@centos7 boot]# grep CONFIG_STRICT_DEVMEM  config-4.14.0-115.el7a.0.1.aarch64 
[root@centos7 boot]# 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mmap_pci/arm64.png)

## x86

```
root@ubuntux86:/boot#  grep CONFIG_STRICT_DEVMEM   config-5.13.0-39-generic
CONFIG_STRICT_DEVMEM=y
root@ubuntux86:/boot# 
```

```
root@ubuntux86:/boot# uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntux86:/boot# 
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/mmap_pci/x86.png)
