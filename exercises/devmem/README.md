
# arm64

```
[root@centos7 boot]# grep STRICT_DEVMEM config-4.14.0-115.el7a.0.1.aarch64
[root@centos7 boot]# grep CONFIG_DEVMEM   config-4.14.0-115.el7a.0.1.aarch64
# CONFIG_DEVMEM is not set
[root@centos7 boot]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 boot]# 
```


# x86


```
ubuntu@ubuntux86:/boot$ grep STRICT_DEVMEM  config-5.13.0-39-generic
CONFIG_STRICT_DEVMEM=y
# CONFIG_IO_STRICT_DEVMEM is not set
ubuntu@ubuntux86:/boot$ grep CONFIG_DEVMEM config-5.13.0-39-generic 
CONFIG_DEVMEM=y
ubuntu@ubuntux86:/boot$ uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
ubuntu@ubuntux86:/boot$ 
```

# e1000e: addr 70280000, and len 20000 

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/devmem/mem.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/devmem/test1.png)

```
root@ubuntux86:/work/test# ./devmem  0x70280000
/dev/mem opened.
Error at line 86, file devmem.c (1) [Operation not permitted]
root@ubuntux86:/work/test# 

```

影响/dev/mem使用的有一个宏CONFIG_STRICT_DEVMEM，如果打开这个宏的话，user space就只能访问reserver memory和kernel 使用的system Ram 两种，
其他的例如PCI 配置空间等这里IO 空间都会返回错误

#  dd if=/dev/mem | hexdump -C > test.txt

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/devmem/dd.png)

# cat /proc/iomem 

```
root@ubuntux86:/work/test# cat /proc/iomem 
00000000-00000fff : Reserved
00001000-0009efff : System RAM
0009f000-000fffff : Reserved
  000a0000-000bffff : PCI Bus 0000:00
  000f0000-000fffff : System ROM
00100000-5cafb017 : System RAM
5cafb018-5cb1ac57 : System RAM
5cb1ac58-5cb38fff : System RAM
5cb39000-5cc7dfff : Reserved
5cc7e000-5cd49fff : System RAM
5cd4a000-5cd4afff : Reserved
5cd4b000-61c71fff : System RAM
61c72000-65c10fff : Reserved
65c11000-66471fff : ACPI Non-volatile Storage
  663a6000-663a6fff : USBC000:00
66472000-666fefff : ACPI Tables
666ff000-666fffff : System RAM
66700000-6e7fffff : Reserved
  6a800000-6e7fffff : Graphics Stolen Memory
6e800000-dfffffff : PCI Bus 0000:00
  6e800000-6e800fff : 0000:00:1f.5
  6f000000-700fffff : PCI Bus 0000:01
    6f000000-6fffffff : 0000:01:00.0
    70000000-70003fff : 0000:01:00.1
      70000000-70003fff : ICH HD audio
    70004000-70004fff : 0000:01:00.3
    70080000-700fffff : 0000:01:00.0
  70100000-701fffff : PCI Bus 0000:02
    70100000-70103fff : 0000:02:00.0
      70100000-70103fff : nvme
    70104000-701040ff : 0000:02:00.0
      70104000-701040ff : nvme
  70280000-7029ffff : 0000:00:1f.6
    70280000-7029ffff : e1000e
  702a0000-702a1fff : 0000:00:17.0
    702a0000-702a1fff : ahci
  702a3000-702a37ff : 0000:00:17.0
    702a3000-702a37ff : ahci
  702a4000-702a40ff : 0000:00:17.0
    702a4000-702a40ff : ahci
e0000000-efffffff : PCI MMCONFIG 0000 [bus 00-ff]
  e0000000-efffffff : Reserved
    e0000000-efffffff : pnp 00:04
fd000000-fd68ffff : pnp 00:05
fd690000-fd69ffff : INT34C6:00
  fd690000-fd69ffff : INT34C6:00 INT34C6:00
fd6a0000-fd6affff : INT34C6:00
  fd6a0000-fd6affff : INT34C6:00 INT34C6:00
fd6b0000-fd6bffff : INT34C6:00
  fd6b0000-fd6bffff : INT34C6:00 INT34C6:00
fd6c0000-fd6cffff : pnp 00:05
fd6d0000-fd6dffff : INT34C6:00
  fd6d0000-fd6dffff : INT34C6:00 INT34C6:00
fd6e0000-fd6effff : INT34C6:00
  fd6e0000-fd6effff : INT34C6:00 INT34C6:00
fd6f0000-fdffffff : pnp 00:05
fe000000-fe01ffff : pnp 00:05
fe04c000-fe04ffff : pnp 00:05
fe050000-fe0affff : pnp 00:05
fe0d0000-fe0fffff : pnp 00:05
fe200000-fe7fffff : pnp 00:05
fec00000-fec003ff : IOAPIC 0
fed00000-fed003ff : HPET 0
  fed00000-fed003ff : PNP0103:00
fed10000-fed17fff : pnp 00:04
fed40000-fed44fff : INTC6000:00
  fed40000-fed44fff : INTC6000:00
fed45000-fed8ffff : pnp 00:04
fed90000-fed90fff : dmar0
fed91000-fed91fff : dmar1
feda0000-feda0fff : pnp 00:04
feda1000-feda1fff : pnp 00:04
fee00000-feefffff : pnp 00:04
  fee00000-fee00fff : Local APIC
ff000000-ffffffff : Reserved
  ff000000-ffffffff : pnp 00:05
100000000-88d7fffff : System RAM
  5bac00000-5bbc02666 : Kernel code
  5bbe00000-5bc83ffff : Kernel rodata
  5bca00000-5bcd6e1ff : Kernel data
  5bd067000-5bd5fffff : Kernel bss
88d800000-88fffffff : RAM buffer
4000000000-7fffffffff : PCI Bus 0000:00
  4000000000-400fffffff : 0000:00:02.0
  4010000000-4010000fff : 0000:00:15.0
    4010000000-40100001ff : lpss_dev
      4010000000-40100001ff : i2c_designware.0 lpss_dev
    4010000200-40100002ff : lpss_priv
    4010000800-4010000fff : idma64.0
      4010000800-4010000fff : idma64.0 idma64.0
  4010001000-4010001fff : 0000:00:15.1
    4010001000-40100011ff : lpss_dev
      4010001000-40100011ff : i2c_designware.1 lpss_dev
    4010001200-40100012ff : lpss_priv
    4010001800-4010001fff : idma64.1
      4010001800-4010001fff : idma64.1 idma64.1
  6000000000-60120fffff : PCI Bus 0000:01
    6000000000-600fffffff : 0000:01:00.0
    6010000000-6011ffffff : 0000:01:00.0
    6012000000-601203ffff : 0000:01:00.2
      6012000000-601203ffff : xhci-hcd
    6012040000-601204ffff : 0000:01:00.2
  6013000000-6013ffffff : 0000:00:02.0
  6014000000-60140fffff : 0000:00:1f.3
    6014000000-60140fffff : ICH HD audio
  6014100000-601410ffff : 0000:00:14.0
    6014100000-601410ffff : xhci-hcd
  6014110000-6014113fff : 0000:00:1f.3
    6014110000-6014113fff : ICH HD audio
  6014114000-6014117fff : 0000:00:14.2
  6014118000-60141180ff : 0000:00:1f.4
  6014119000-6014119fff : 0000:00:16.0
    6014119000-6014119fff : mei_me
  601411c000-601411cfff : 0000:00:14.2
root@ubuntux86:/work/test#
```
# ./master 

```
root@ubuntux86:/work/test# ./master 
pid = 2649, virtual addr = 4a9e3068 , physical addr = 48bf7068
addr = 0xffffffffffffffff // why 

root@ubuntux86:/work/test# 
```

# references

[linux-mem-device     ](https://hackmd.io/@sysprog/linux-mem-device)