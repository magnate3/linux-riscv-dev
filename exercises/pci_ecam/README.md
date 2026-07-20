


# 1.PCIe ECAM机制
PCI Express Enhanced Configuration Access Mechanism (ECAM)是访问PCIe配置空间的一种机制。是将PCIe的配置空间映射到MEM空间，使用MEM访问其配置空间的一种实现。可以根据一个PCIe设备的BDF得到其配置空间偏移地址：
```C
#define PCI_ECAM_ADDRESS(Bus,Device,Function,Offset) \
  (((Offset) & 0xfff) | (((Function) & 0x07) << 12) | (((Device) & 0x1f) << 15) | (((Bus) & 0xff) << 20))
```

# 2.PCIe ECAM基地址
查看一台主机是否支持PCIe ECAM机制，可通过以下命令查看：

> ## 2.1.对于x86主机：

sudo cat /proc/iomem | grep MMCONFIG


若支持ECAM，则会出现类似以下字段：

f8000000 - fbffffff : PCI MMCONFIG 0000 [bus 00-3f]


其中0xf8000000就是该主机下PCIe ECAM的基地址。加上PCI_ECAM_ADDRESS(bus, device, func, reg)，就得到了指定PCIe设备的物理地址，可以使用此地址直接访问其配置空间。


```
cat /proc/iomem | grep MMCONFIG
  80000000-8fffffff : PCI MMCONFIG 0000 [bus 00-ff]
```

> ## 2.2.对于ARM64主机：

sudo cat /proc/iomem | grep ECAM
若支持ECAM，则会出现以下字段：

40000000-4fffffff : PCI ECAM

其中0x40000000就是该主机下PCIe ECAM的基地址。加上PCI_ECAM_ADDRESS(bus, device, func, reg)，就得到了指定PCIe设备的物理地址，可以使用此地址直接访问其配置空间。
 
 ```
 [root@centos7 ~]# cat /proc/iomem | grep ECAM
d0000000-d3ffffff : PCI ECAM
d7400000-d76fffff : PCI ECAM
d7800000-d79fffff : PCI ECAM
d7a00000-d7afffff : PCI ECAM
d7b00000-d7bfffff : PCI ECAM
d7c00000-d7dfffff : PCI ECAM
d8000000-d9ffffff : PCI ECAM
db400000-db6fffff : PCI ECAM
db800000-db9fffff : PCI ECAM
dba00000-dbafffff : PCI ECAM
dbb00000-dbbfffff : PCI ECAM
dbc00000-dbdfffff : PCI ECAM
[root@centos7 ~]# 
 ```
 
 
 #  pci_scan_device
 ![image](pci.jpg)
 
 ##  pci_bus_read_dev_vendor_id
 在函数内部，pci_bus_read_config_dword会根据ECAM规范，将bus、devfn和where参数转换为ECAM规范下的地址，并通过ECAM机制从该地址读取32位数据。读取到的数据会存储在val指针指向的内存位置中，以供函数调用者使用。    
总结起来，ECAM机制提供了一种标准的访问PCI设备配置空间的方式，函数pci_bus_read_config_dword则是在ECAM规范下实现的读取32位数据的函数。通过这两者的结合，可以方便地访问和读取PCI设备的配置空间。    
 
 ```
 bool pci_bus_read_dev_vendor_id(struct pci_bus *bus, int devfn, u32 *l,
				int crs_timeout)
{
	int delay = 1;

	if (pci_bus_read_config_dword(bus, devfn, PCI_VENDOR_ID, l))
		return false;

	/* some broken boards return 0 or ~0 if a slot is empty: */
	if (*l == 0xffffffff || *l == 0x00000000 ||
	    *l == 0x0000ffff || *l == 0xffff0000)
		return false;

	/*
	 * Configuration Request Retry Status.  Some root ports return the
	 * actual device ID instead of the synthetic ID (0xFFFF) required
	 * by the PCIe spec.  Ignore the device ID and only check for
	 * (vendor id == 1).
	 */
	while ((*l & 0xffff) == 0x0001) {
		if (!crs_timeout)
			return false;

		msleep(delay);
		delay *= 2;
		if (pci_bus_read_config_dword(bus, devfn, PCI_VENDOR_ID, l))
			return false;
		/* Card hasn't responded in 60 seconds?  Must be stuck. */
		if (delay > crs_timeout) {
			printk(KERN_WARNING "pci %04x:%02x:%02x.%d: not responding\n",
			       pci_domain_nr(bus), bus->number, PCI_SLOT(devfn),
			       PCI_FUNC(devfn));
			return false;
		}
	}

	return true;
}
EXPORT_SYMBOL(pci_bus_read_dev_vendor_id);

/*
 * Read the config data for a PCI device, sanity-check it
 * and fill in the dev structure...
 */
static struct pci_dev *pci_scan_device(struct pci_bus *bus, int devfn)
{
	struct pci_dev *dev;
	u32 l;

	if (!pci_bus_read_dev_vendor_id(bus, devfn, &l, 60*1000))
		return NULL;

	dev = pci_alloc_dev(bus);
	if (!dev)
		return NULL;

	dev->devfn = devfn;
	dev->vendor = l & 0xffff;
	dev->device = (l >> 16) & 0xffff;

	pci_set_of_node(dev);

	if (pci_setup_device(dev)) {
		pci_bus_put(dev->bus);
		kfree(dev);
		return NULL;
	}

	return dev;
}
 ```
 
 #  pci_scan_slot
 ```
 unsigned int pci_scan_child_bus(struct pci_bus *bus)
{
	unsigned int devfn, pass, max = bus->busn_res.start;
	struct pci_dev *dev;

	dev_dbg(&bus->dev, "scanning bus\n");

	/* Go find them, Rover! */
	for (devfn = 0; devfn < 0x100; devfn += 8)
		pci_scan_slot(bus, devfn);
}
 ```
 
 # pci scan test
 
 ```
 ./usertools/dpdk-devbind.py  -u 0000:05:00.0
 insmod  pci_test.ko 
 ```
 PCIe 标准支持在一台 Host 上创建最多 256 个 PCIe Bus，每条 Bus 最多可以支持 32 个 PCIe Device，每个 Device 最多可以支持 8 个 Functions   
 BDF：bus是8位，Ddevice是5位，function是3位    
 ```
/* PCIe device probe */
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
        pr_info("***************** pci bus  info show ************ \n");
        u8 index = 0;
        unsigned int devfn;
        printk("Vendor: %04x Device: %04x, devfun %d, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
        //printk("Vendor: %#x Device: %#x, devfun %x, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
        struct pci_bus *bus = pdev->bus, *bus2;
        show_pci_info(bus);
        pr_info("***************** pci bus************ \n");
        for(index = 0; index != 0xff; index++)
        {
            bus2 = pci_find_bus(pci_domain_nr(bus), index);
            if (bus2)
            {
                   pr_err("bus2 name : %s, bus2 ops %p \n", bus2->name, bus2->ops);
            }
        }
        pr_info("***************** pci scan start ************ \n");
        test_pci_scan_device(bus, pdev->devfn);
        for (devfn = 0; devfn < 0x100; devfn += 8)
        {
            test_pci_scan_device(bus, devfn);
        } 
        return 0;
}
 ```
 
 ```
[ 2132.267558] ***************** pci bus  info show ************ 
[ 2132.273366] Vendor: 19e5 Device: 0200, devfun 0, and name 0000:05:00.0 
[ 2132.279958] bus name : PCI Bus 0000:05, bus ops ffff000008eb6a90 
[ 2132.286027] Vendor: 0x19e5 Device: 0x371e, devfun 0, and name 0000:04:00.0 
[ 2132.292956] bus name : PCI Bus 0000:04, bus ops ffff000008eb6a90 
[ 2132.299024] Vendor: 0x19e5 Device: 0x371e, devfun 0, and name 0000:03:00.0 
[ 2132.305956] bus name : PCI Bus 0000:03, bus ops ffff000008eb6a90 
[ 2132.312021] Vendor: 0x19e5 Device: 0xa120, devfun 60, and name 0000:00:0c.0 
[ 2132.319038] bus name : , bus ops ffff000008eb6a90 
[ 2132.323807] ***************** pci bus************ 
[ 2132.328579] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.333521] bus2 name : PCI Bus 0000:01, bus2 ops ffff000008eb6a90 
[ 2132.339761] bus2 name : PCI Bus 0000:02, bus2 ops ffff000008eb6a90 
[ 2132.346001] bus2 name : PCI Bus 0000:03, bus2 ops ffff000008eb6a90 
[ 2132.352238] bus2 name : PCI Bus 0000:04, bus2 ops ffff000008eb6a90 
[ 2132.358479] bus2 name : PCI Bus 0000:05, bus2 ops ffff000008eb6a90 
[ 2132.364716] bus2 name : PCI Bus 0000:06, bus2 ops ffff000008eb6a90 
[ 2132.370956] bus2 name : PCI Bus 0000:07, bus2 ops ffff000008eb6a90 
[ 2132.377196] bus2 name : PCI Bus 0000:08, bus2 ops ffff000008eb6a90 
[ 2132.383433] bus2 name : PCI Bus 0000:09, bus2 ops ffff000008eb6a90 
[ 2132.389723] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.394666] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.399611] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.404552] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.409496] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.414438] bus2 name : PCI Bus 0000:7d, bus2 ops ffff000008eb6a90 
[ 2132.420679] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.425623] bus2 name : PCI Bus 0000:81, bus2 ops ffff000008eb6a90 
[ 2132.431860] bus2 name : PCI Bus 0000:82, bus2 ops ffff000008eb6a90 
[ 2132.438100] bus2 name : PCI Bus 0000:83, bus2 ops ffff000008eb6a90 
[ 2132.444338] bus2 name : PCI Bus 0000:84, bus2 ops ffff000008eb6a90 
[ 2132.450600] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.455547] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.460488] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.465433] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.470374] bus2 name : , bus2 ops ffff000008eb6a90 
[ 2132.475319] bus2 name : PCI Bus 0000:bd, bus2 ops ffff000008eb6a90 
[ 2132.481584] ***************** pci scan start ************ 
[ 2132.487050] vendor 19e5, deivce 200, fn 0 
[ 2132.491127] vendor 19e5, deivce 200, fn 0 
[ 2132.495206] devfn 8 pci_bus_read_dev_vendor_id fail 
 ```