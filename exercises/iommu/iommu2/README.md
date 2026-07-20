# insmod  pci_test2.ko 
```
[root@centos7 igb-uio]# ./dpdk-devbind.py  -u 0000:05:00.0
Warning: no supported DPDK kernel modules are loaded
Notice: 0000:05:00.0 Hi1822 Family (2*100GE)  is not currently managed by any driver
[root@centos7 igb-uio]# ./dpdk-devbind.py  -u 0000:05:00.0
Warning: no supported DPDK kernel modules are loaded
Notice: 0000:05:00.0 Hi1822 Family (2*100GE)  is not currently managed by any driver
[root@centos7 igb-uio]# ./dpdk-devbind.py  -u 0000:06:00.0
Warning: no supported DPDK kernel modules are loaded
Notice: 0000:06:00.0 Hi1822 Family (2*100GE)  is not currently managed by any driver
[root@centos7 igb-uio]# 
```

```
[root@centos7 iommu]# insmod  pci_test2.ko 
[root@centos7 iommu]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
[root@centos7 iommu]# dmesg | tail -n 20
[ 4113.487966] ***************** pci iommu bus number 5, slot 0, devfn 0  ************ 
[ 4113.495678] group id 25 and group name (null) 
[ 4113.500107] dbg_added iommu_domain_alloc ok
[ 4113.504275] iommu_domain_ops: ffff000008ecec38 
[ 4113.508801] dbg_added iommu_attach_device ok
[ 4113.515080] dbg_added iommu_map ok
[ 4113.518466]  phys_addr: 0000002100000000, my_iova: 0000000050100000 
[ 4113.535121] ***************** pci iommu bus number 6, slot 0, devfn 0  ************ 
[ 4113.542828] group id 31 and group name (null) 
[ 4113.547265] dbg_added iommu_domain_alloc ok
[ 4113.551428] iommu_domain_ops: ffff000008ecec38 
[ 4113.555955] dbg_added iommu_attach_device ok
[ 4113.562164] dbg_added iommu_map ok
[ 4113.565555]  phys_addr: 0000002100000000, my_iova: 0000000050100000 
[root@centos7 iommu]# 
```



#  hns_uio_set_iommu

```
void hns_uio_set_iommu(struct nic_uio_device *priv, unsigned long iova,
               unsigned long paddr, int gfp_order)
{
    struct iommu_domain *domain;
    int ret = 0;

    domain = iommu_domain_alloc(priv->dev->bus); ///////////////

    if (!domain)
        PRINT(KERN_ERR, "domain is null\n");

    ret = iommu_attach_device(domain, priv->dev);
    PRINT(KERN_ERR, "domain is null = %d\n", ret);

    ret =
        iommu_map(domain, iova, (phys_addr_t)paddr, gfp_order,
              (IOMMU_WRITE | IOMMU_READ | IOMMU_CACHE));
    PRINT(KERN_ERR, "domain is null = %d\n", ret);
}

```


# iommu_attach_device
```
int iommu_attach_device(struct iommu_domain *domain, struct device *dev)
{
    struct iommu_group *group;
    int ret;

    group = iommu_group_get(dev);
    /* FIXME: Remove this when groups a mandatory for iommu drivers */
    if (group == NULL)
        return __iommu_attach_device(domain, dev);

    /*
     * We have a group - lock it to make sure the device-count doesn't
     * change while we are attaching
     */
    mutex_lock(&group->mutex);
    ret = -EINVAL;
    if (iommu_group_device_count(group) != 1)
        goto out_unlock;

    ret = __iommu_attach_group(domain, group);

out_unlock:
    mutex_unlock(&group->mutex);
    iommu_group_put(group);

    return ret;
}
```