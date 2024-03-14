
./usertools/dpdk-devbind.py  -u 0000:05:00.0

#  get_dma_ops
```
static inline const struct dma_map_ops *test_get_dma_ops(struct device *dev)
{
    if (dev->dma_ops)
    {
         pr_info("dev has dma_ops \n");
            return dev->dma_ops;
    }
    return get_arch_dma_ops(dev->bus);
}
```

#  insmod  pci_test.ko 


```
[root@centos7 pci_dma]#  echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
-bash: echo: write error: File exists
```
忽略error   
```
[root@centos7 pci_dma]# dmesg | tail -n 5
[950860.603239] Failed to associated timeout policy `ovs_test_tp'
[3435951.550325] ***************** pci bus  info show ************ 
[3435951.556312] Vendor: 19e5 Device: 0200, devfun 0, and name 0000:05:00.0 
[3435951.563074] dev has dma_ops 
[3435951.566116] dma ops ffff0000088c0010  
You have new mail in /var/spool/mail/root
[root@centos7 pci_dma]# 
```
swiotlb_dma_ops
```
[root@centos7 pci_dma]# cat /proc/kallsyms  | grep 'ffff0000088c0010'
ffff0000088c0010 r swiotlb_dma_ops
```