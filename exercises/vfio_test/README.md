
# berfore bind
```
[root@centos7 ethercat_analyze]# ls /dev/vfio/
vfio
[root@centos7 ethercat_analyze]# modprobe vfio-pci
[root@centos7 ethercat_analyze]# 
[root@centos7 igb-uio]# lspci -n -s  0000:05:00.0
05:00.0 0200: 19e5:0200 (rev 45)
[root@centos7 igb-uio]#
[root@centos7 igb-uio]# ./dpdk-devbind.py -s

Network devices using kernel driver
===================================
0000:05:00.0 'Hi1822 Family (2*100GE) 0200' if=enp5s0 drv=hinic unused=igb_uio,vfio-pci 
0000:06:00.0 'Hi1822 Family (2*100GE) 0200' if=enp6s0 drv=hinic unused=igb_uio,vfio-pci *Active*
```
# bind vfio
```
[root@centos7 igb-uio]# lspci -n -s  0000:05:00.0
05:00.0 0200: 19e5:0200 (rev 45)
[root@centos7 igb-uio]#  echo 0000:06:0d.0 > /sys/bus/pci/devices/0000:05:00.0/driver/unbind
-bash: echo: write error: No such device
[root@centos7 igb-uio]#  echo 0000:05:00.0 > /sys/bus/pci/devices/0000:05:00.0/driver/unbind
[root@centos7 igb-uio]#  echo  19e5 0200 > /sys/bus/pci/drivers/vfio-pci/new_id
[root@centos7 igb-uio]# ./dpdk-devbind.py -s

Network devices using DPDK-compatible driver
============================================
0000:05:00.0 'Hi1822 Family (2*100GE) 0200' drv=vfio-pci unused=hinic,igb_uio

Network devices using kernel driver
===================================
0000:06:00.0 'Hi1822 Family (2*100GE) 0200' if=enp6s0 drv=hinic unused=igb_uio,vfio-pci *Active*

[root@centos7 igb-uio]# ls /dev/vfio/
24  vfio
[root@centos7 igb-uio]# 
```
# vfio-test.c
```
[root@centos7 vfio]# gcc vfio-test.c -o test
[root@centos7 vfio]# ls /dev/vfio/
24  vfio
[root@centos7 vfio]# ./test 24 0000:05:00.0 
Using PCI device 0000:05:00.0 in group 24
pre-SET_CONTAINER:
VFIO_CHECK_EXTENSION VFIO_TYPE1_IOMMU: Present
VFIO_CHECK_EXTENSION VFIO_NOIOMMU_IOMMU: Not Present
post-SET_CONTAINER:
VFIO_CHECK_EXTENSION VFIO_TYPE1_IOMMU: Present
VFIO_CHECK_EXTENSION VFIO_NOIOMMU_IOMMU: Not Present
Device supports 9 regions, 5 irqs
Region 0: size 0x20000, offset 0x0, flags 0x7
[]
Region 1: size 0x0, offset 0x10000000000, flags 0x0
Region 2: size 0x8000, offset 0x20000000000, flags 0xf
mmap failed
Region 3: size 0x0, offset 0x30000000000, flags 0x0
Region 4: size 0x100000, offset 0x40000000000, flags 0x7
[]
Region 5: size 0x0, offset 0x50000000000, flags 0x0
Region 6: size 0x100000, offset 0x60000000000, flags 0x1
Region 7: size 0x1000, offset 0x70000000000, flags 0x3
Region 8: Failed to get info
Success
```


```
[root@centos7 demos]# gcc 3_device_info.c  -o test

[root@centos7 demos]# ./test /dev/vfio/24  0000:05:00.0 
pid:50048
group fd:::3
VFIO_GROUP_GET_STATUS status {8:0x1}
container fd:::4
vfio api version:0
vfio type is VFIO_TYPE1_IOMMU
device fd:::5
device_info {num_irqs:5, num_regions:9}
0 region info {argsz:32, flags:0x7, cap_offset:0 size:131072 offset:0x0}
1 region info {argsz:32, flags:0x0, cap_offset:0 size:0 offset:0x10000000000}
2 region info {argsz:64, flags:0xf, cap_offset:0 size:32768 offset:0x20000000000}
2 region info {argsz:64, flags:0xf, cap_offset:32 size:32768 offset:0x20000000000}
  hdr {id:1, version:1 next:0}
    sparse->areas 0 offset:0x0 size:0x0
3 region info {argsz:32, flags:0x0, cap_offset:0 size:0 offset:0x30000000000}
4 region info {argsz:32, flags:0x7, cap_offset:0 size:1048576 offset:0x40000000000}
5 region info {argsz:32, flags:0x0, cap_offset:0 size:0 offset:0x50000000000}
6 region info {argsz:32, flags:0x1, cap_offset:0 size:1048576 offset:0x60000000000}
7 region info {argsz:32, flags:0x3, cap_offset:0 size:4096 offset:0x70000000000}
8 region info {argsz:32, flags:0x0, cap_offset:0 size:0 offset:0x0}
```

##  4_bar_info.c
VFIO_PCI_CONFIG_REGION_INDEX

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vfio_test/pic/4_bar_info.png)

```
[1481070.101672] igb_uio: in igb uio driver attr0 80000000, and attr1 c1502200
```

```
[root@centos7 user]# cat  /sys/bus/pci/devices/0000:05:00.0/resource
0x0000080007b00000 0x0000080007b1ffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000080008a20000 0x0000080008a27fff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000080000200000 0x00000800002fffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x00000000e9200000 0x00000000e92fffff 0x0000000000046200
0x0000080007b20000 0x000008000829ffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x00000800082a0000 0x0000080008a1ffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
0x0000080000300000 0x0000080007afffff 0x000000000014220c
0x0000000000000000 0x0000000000000000 0x0000000000000000
```

# vfio

```
[root@centos7 kernel]# ls /dev/vfio/
24  27  vfio
[root@centos7 kernel]# 
```

# IOMMU

```
ubuntu@ubuntux86:/boot$ uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
ubuntu@ubuntux86:/boot$ grep CONFIG_DMAR_DEFAULT_ON  config-5.13.0-39-generic
ubuntu@ubuntux86:/boot$ dmesg | grep -e DMAR -e IOMMU
[    0.005215] ACPI: DMAR 0x0000000066683000 000088 (v02 INTEL  Dell Inc 00000002      01000013)
[    0.005239] ACPI: Reserving DMAR table memory at [mem 0x66683000-0x66683087]
[    0.102389] DMAR: Host address width 39
[    0.102390] DMAR: DRHD base: 0x000000fed90000 flags: 0x0
[    0.102394] DMAR: dmar0: reg_base_addr fed90000 ver 1:0 cap 1c0000c40660462 ecap 19e2ff0505e
[    0.102395] DMAR: DRHD base: 0x000000fed91000 flags: 0x1
[    0.102397] DMAR: dmar1: reg_base_addr fed91000 ver 1:0 cap d2008c40660462 ecap f050da
[    0.102398] DMAR: RMRR base: 0x0000006a000000 end: 0x0000006e7fffff
[    0.102400] DMAR-IR: IOAPIC id 2 under DRHD base  0xfed91000 IOMMU 1
[    0.102400] DMAR-IR: HPET id 0 under DRHD base 0xfed91000
[    0.102401] DMAR-IR: Queued invalidation will be enabled to support x2apic and Intr-remapping.
[    0.103884] DMAR-IR: Enabled IRQ remapping in x2apic mode
ubuntu@ubuntux86:/boot$ 
```


逻辑上来说，IOMMU group是IOMMU操作的最小对象。某些IOMMU硬件支持将若干IOMMU group组成更大的单元。VFIO据此做出container的概念，可容纳多个IOMMU group。打开/dev/vfio文件即新建一个空的container。在VFIO中，container是IOMMU操作的最小对象。

要使用VFIO，需先将设备与原驱动拨离，并与VFIO绑定。

***用VFIO访问硬件的步骤***

*(1)* 打开设备所在IOMMU group在/dev/vfio/N目录下的文件

vfio_gfd =  open("/dev/vfio/N", O_RDWR)

*(2)* 使用VFIO_GROUP_GET_DEVICE_FD得到表示设备的文件描述 (参数为设备名称，一个典型的PCI设备名形如0000:03.00.01)

vfio_fd = ioctl(vfio_gfd, VFIO_GROUP_GET_DEVICE_FD, pci_addr)

*(3)* 对设备进行read/write/mmap等操作

  ioctl(vfio_fd, VFIO_DEVICE_GET_REGION_INFO, &region_info)
 
 

***用VFIO配置IOMMU的步骤***

*(1)* 打开/dev/vfio，得到container文件描述符

 cfd =  open("/dev/vfio/vfio", O_RDWR)
 
*(2)* 用VFIO_SET_IOMMU绑定一种IOMMU实现层

ioctl(cfd, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU)

*(3)*  打开/dev/vfio/N，得到IOMMU group文件描述符

 vfio_gfd =  open("/dev/vfio/N", O_RDWR)
 
*(4)*  用VFIO_GROUP_SET_CONTAINER将IOMMU group加入container

ioctl(vfio_gfd, VFIO_GROUP_SET_CONTAINER, &cfd)

*(5)* 用VFIO_IOMMU_MAP_DMA将此IOMMU group的DMA地址映射至进程虚拟地址空间

 ioctl(cfd, VFIO_IOMMU_MAP_DMA, &dma_map)