# lspci -n -s 05:00.0
```
[root@centos7 pci_test]# lspci -n -s 05:00.0
05:00.0 0200: 19e5:0200 (rev 45)
[root@centos7 pci_test]# 
```

# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id

```
[root@centos7 pci_test]#  echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id 
-bash: echo: write error: File exists
[root@centos7 pci_test]# ls  /sys/bus/pci/drivers/PCIe_demo/
0000:05:00.0  bind  module  new_id  remove_id  uevent  unbind
```

# insmod  pci_test.ko 

```
[2353073.299542] Vendor: 0x19e5 Device: 0x200, devfun 0
[2353073.304490] Capability: 0x8fe2
[2353073.307703] Power: 0x0
[2353073.310226] MSI/MSI-X: 0x18a
[2353073.313272] enable MSI-X 
[2353073.313273] MSI-X: 0x801f
[2353073.318835] MSI-X: 0x801f
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/msi/pci-test/pic/b_en.png)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/msi/pci-test/pic/a_en.png)

# pci_msi_set_enable

```
static inline void pci_msi_set_enable(struct pci_dev *dev, int enable)
{
	u16 control;
 
    /*
     *    读取消息控制寄存器.
     */
	pci_read_config_word(dev, dev->msi_cap + PCI_MSI_FLAGS, &control);
    /*
     *    bit[0] 置 1.  见下图.
     */
	control &= ~PCI_MSI_FLAGS_ENABLE;
	if (enable)
		control |= PCI_MSI_FLAGS_ENABLE;
    /*
     *    写入消息控制寄存器
     */
	pci_write_config_word(dev, dev->msi_cap + PCI_MSI_FLAGS, control);
}
```


# pci_read_config_word

```
drivers/pci/access.c:533:int pci_read_config_word(const struct pci_dev *dev, int where, u16 *val)

#define PCI_OP_READ(size, type, len) \
int noinline pci_bus_read_config_##size \
        (struct pci_bus *bus, unsigned int devfn, int pos, type *value) \
{                                                                       \
        int res;                                                        \
        unsigned long flags;                                            \
        u32 data = 0;                                                   \
        if (PCI_##size##_BAD) return PCIBIOS_BAD_REGISTER_NUMBER;       \
        pci_lock_config(flags);                                         \
        res = bus->ops->read(bus, devfn, pos, len, &data);              \
        *value = (type)data;                                            \
        pci_unlock_config(flags);                                       \
        return res;                                                     \
}
```
## bus->ops->write && dev->devfn
```
#define PCI_OP_WRITE(size, type, len) \
int noinline pci_bus_write_config_##size \
        (struct pci_bus *bus, unsigned int devfn, int pos, type value)  \
{                                                                       \
        int res;                                                        \
        unsigned long flags;                                            \
        if (PCI_##size##_BAD) return PCIBIOS_BAD_REGISTER_NUMBER;       \
        pci_lock_config(flags);                                         \
        res = bus->ops->write(bus, devfn, pos, len, value);             \
        pci_unlock_config(flags);                                       \
        return res;                                                     \
}

PCI_OP_READ(byte, u8, 1)
PCI_OP_READ(word, u16, 2)

int pci_read_config_word(const struct pci_dev *dev, int where, u16 *val)
{
        if (pci_dev_is_disconnected(dev)) {
                *val = ~0;
                return PCIBIOS_DEVICE_NOT_FOUND;
        }
        return pci_bus_read_config_word(dev->bus, dev->devfn, where, val);
}
```

# reference
[Linux PCI Express配置空间读写内核实现](http://www.ilinuxkernel.com/files/5/Linux_PCI_Express_Kernel_RW.htm)