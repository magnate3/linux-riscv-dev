# product id
```
[root@centos7 igb-uio]# ./dpdk-devbind.py  -u 0000:05:00.0
Warning: no supported DPDK kernel modules are loaded
[root@centos7 igb-uio]# lspci -n  -s 0000:05:00.0
05:00.0 0200: 19e5:0200 (rev 45)
[root@centos7 igb-uio]# 
```

#  insmod  pci_irq_test.ko 
```
[root@centos7 pci_irq]# insmod  pci_irq_test.ko 
[root@centos7 pci_irq]# dmesg | tail -n 10
[ 1324.949794] hinic 0000:05:00.0 enp5s0: HINIC_INTF is DOWN
[ 1325.101776] hinic 0000:05:00.0: HiNIC driver - removed
[ 1687.317773] 
Hello, world
[ 1687.317874] 
Probed
[ 1687.321874] irq 255
[ 1687.325354] 
 ISR registration failure, 4294967274
[ 1687.327459] Test Driver: probe of 0000:05:00.0 failed with error -1
```

# references

[linux/modules/9_interrupt/module/](https://github.com/gopakumar-thekkedath/linux/tree/master/modules/9_interrupt/module)