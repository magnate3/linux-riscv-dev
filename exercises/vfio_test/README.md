
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