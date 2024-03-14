

```
 ./usertools/dpdk-devbind.py  -u 0000:05:00.0
```

#  echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci_scan/hinic/scan.png)

```
[root@centos7 pci_scan]# lspci -n -s  0000:05:00.0
05:00.0 0200: 19e5:0200 (rev 45)
[root@centos7 pci_scan]#  echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
```

```
[79376.447976] ***************** pci bus  info show ************ 
[79376.453788] Vendor: 19e5 Device: 0200, devfun 0, and name 0000:05:00.0 
[79376.460375] bus name : PCI Bus 0000:05, bus ops ffff000008eb6a90 
[79376.466441] Vendor: 0x19e5 Device: 0x371e, devfun 0, and name 0000:04:00.0 
[79376.473374] bus name : PCI Bus 0000:04, bus ops ffff000008eb6a90 
[79376.479444] Vendor: 0x19e5 Device: 0x371e, devfun 0, and name 0000:03:00.0 
[79376.486372] bus name : PCI Bus 0000:03, bus ops ffff000008eb6a90 
[79376.492441] Vendor: 0x19e5 Device: 0xa120, devfun 60, and name 0000:00:0c.0 
[79376.499459] bus name : , bus ops ffff000008eb6a90 
[79376.504226] ***************** pci scan ************ 
[79376.509174] pci_bus_read_dev_vendor_id fail 
[79376.513425] pci_bus_read_dev_vendor_id fail 
[79376.517711] ***************** pci bus  info show ************ 
[79376.523522] Vendor: 19e5 Device: 0200, devfun 0, and name 0000:06:00.0 
[79376.530109] bus name : PCI Bus 0000:06, bus ops ffff000008eb6a90 
[79376.536175] Vendor: 0x19e5 Device: 0x371e, devfun 8, and name 0000:04:01.0 
[79376.543106] bus name : PCI Bus 0000:04, bus ops ffff000008eb6a90 
[79376.549177] Vendor: 0x19e5 Device: 0x371e, devfun 0, and name 0000:03:00.0 
[79376.556105] bus name : PCI Bus 0000:03, bus ops ffff000008eb6a90 
[79376.562174] Vendor: 0x19e5 Device: 0xa120, devfun 60, and name 0000:00:0c.0 
[79376.569194] bus name : , bus ops ffff000008eb6a90 
[79376.573963] ***************** pci scan ************ 
[79376.578908] pci_bus_read_dev_vendor_id fail 
[79376.583159] pci_bus_read_dev_vendor_id fail 
[root@centos7 pci_scan]# 
```

***bus ops 都是ffff000008eb6a90***