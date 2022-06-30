
# make and install 
1. make sure you have install this

Fedora
  $sudo yum install kernel-headers
or
Ubuntu   
  $sudo apt-get install linux-headers-$(uname -r)

2. from this dir do this

$make 
$sudo modprobe uio
$sudo insmod igb_uio.ko


#  ./dpdk-devbind.py

```
[root@centos7 igb-uio]# ls /sys/bus/pci/drivers/igb_uio/new_id 
/sys/bus/pci/drivers/igb_uio/new_id
[root@centos7 igb-uio]# cat /sys/bus/pci/drivers/igb_uio/new_id 
cat: /sys/bus/pci/drivers/igb_uio/new_id: Permission denied
[root@centos7 igb-uio]# 
```

```
[root@centos7 igb-uio]# ./dpdk-devbind.py -s

Network devices using kernel driver
===================================
0000:05:00.0 'Hi1822 Family (2*100GE) 0200' if=enp5s0 drv=hinic unused=igb_uio 
0000:06:00.0 'Hi1822 Family (2*100GE) 0200' if=enp6s0 drv=hinic unused=igb_uio 
0000:7d:00.0 'HNS GE/10GE/25GE RDMA Network Controller a222' if=enp125s0f0 drv=hns3 unused=hns_roce_hw_v2,igb_uio *Active*
0000:7d:00.1 'HNS GE/10GE/25GE Network Controller a221' if=enp125s0f1 drv=hns3 unused=igb_uio 
0000:7d:00.2 'HNS GE/10GE/25GE RDMA Network Controller a222' if=enp125s0f2 drv=hns3 unused=hns_roce_hw_v2,igb_uio 
0000:7d:00.3 'HNS GE/10GE/25GE Network Controller a221' if=enp125s0f3 drv=hns3 unused=igb_uio 

No 'Baseband' devices detected
==============================

No 'Crypto' devices detected
============================

No 'Eventdev' devices detected
==============================

No 'Mempool' devices detected
=============================

No 'Compress' devices detected
==============================

No 'Misc (rawdev)' devices detected
===================================
[root@centos7 igb-uio]# ethtool -i enp5s0
driver: hinic
version: 
firmware-version: 
expansion-rom-version: 
bus-info: 0000:05:00.0
supports-statistics: no
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: no
[root@centos7 igb-uio]# ./dpdk-devbind.py -b hinic 0000:05:00.0
Notice: 0000:05:00.0 already bound to driver hinic, skipping
[root@centos7 igb-uio]# ethtool -i enp5s0
driver: hinic
version: 
firmware-version: 
expansion-rom-version: 
bus-info: 0000:05:00.0
supports-statistics: no
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: no
[root@centos7 igb-uio]# ./dpdk-devbind.py -b igb_uio  0000:05:00.0
[root@centos7 igb-uio]# ethtool -i enp5s0
Cannot get driver information: No such device
[root@centos7 igb-uio]# 
```

```
[root@centos7 igb-uio]# ./dpdk-devbind.py -s

Network devices using DPDK-compatible driver
============================================
0000:05:00.0 'Hi1822 Family (2*100GE) 0200' drv=igb_uio unused=hinic

Network devices using kernel driver
===================================
0000:06:00.0 'Hi1822 Family (2*100GE) 0200' if=enp6s0 drv=hinic unused=igb_uio 
0000:7d:00.0 'HNS GE/10GE/25GE RDMA Network Controller a222' if=enp125s0f0 drv=hns3 unused=hns_roce_hw_v2,igb_uio *Active*
0000:7d:00.1 'HNS GE/10GE/25GE Network Controller a221' if=enp125s0f1 drv=hns3 unused=igb_uio 
0000:7d:00.2 'HNS GE/10GE/25GE RDMA Network Controller a222' if=enp125s0f2 drv=hns3 unused=hns_roce_hw_v2,igb_uio 
0000:7d:00.3 'HNS GE/10GE/25GE Network Controller a221' if=enp125s0f3 drv=hns3 unused=igb_uio 

No 'Baseband' devices detected
==============================

No 'Crypto' devices detected
============================

No 'Eventdev' devices detected
==============================

No 'Mempool' devices detected
=============================

No 'Compress' devices detected
==============================

No 'Misc (rawdev)' devices detected
===================================
[root@centos7 igb-uio]# 
```

