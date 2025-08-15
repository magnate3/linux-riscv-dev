
#  insmod  iommu_test.ko 

```
./dpdk-devbind.py  -u 0000:05:00.0
[root@centos7 iommu]# rmmod hinic
[root@centos7 iommu]# insmod  iommu_test.ko 
[root@centos7 iommu]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
[root@centos7 iommu]# dmesg | tail -n 10
[  350.777470] ioctl_example close was called
[  840.746228] Goodbye, Linux kernel!
[171851.569810] hinic 0000:05:00.0: IO stopped
[171851.619520] hinic 0000:05:00.0 enp5s0: HINIC_INTF is DOWN
[171851.820353] hinic 0000:05:00.0: HiNIC driver - removed
[172006.877200] hinic 0000:06:00.0: IO stopped
[172006.929619] hinic 0000:06:00.0 enp6s0: HINIC_INTF is DOWN
[172007.057763] hinic 0000:06:00.0: HiNIC driver - removed
[172014.320314] pci iommu test successfully 
[172014.324387] pci iommu test successfully 
```