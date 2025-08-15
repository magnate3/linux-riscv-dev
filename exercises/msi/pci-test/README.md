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


# test2

```
[2363251.664294] ***************** pci header show ************
[2363251.664298]   vendor ID =                   0x19e5
[2363251.674877]   device ID =                   0x0200
[2363251.679820]   command register =            0x0002
[2363251.684766]   status register =             0x0010
[2363251.689709]   revision ID =                 0x45
[2363251.694482]   programming interface =       0x00
[2363251.699249]   cache line =                  0x08
[2363251.704019]   latency time =                0x00
[2363251.708790]   header type =                 0x00
[2363251.713562]   BIST =                        0x00
[2363251.718332]   base address 0 =              0x07b0000c
[2363251.723623] ** normal pci device **
[2363251.723625]   base address 1 =              0x00000800
[2363251.732560]   base address 2 =              0x08a2000c
[2363251.737846]   base address 3 =              0x00000800
[2363251.743135]   base address 4 =              0x0020000c
[2363251.748422]   base address 5 =              0x00000800
[2363251.753712]   cardBus CIS pointer =         0x00000000
[2363251.758999]   sub system vendor ID =        0x19e5
[2363251.763943]   sub system ID =               0xd139
[2363251.768886]   expansion ROM base address =  0xe9200000
[2363251.774177]   interrupt line =              0xff
[2363251.778948]   interrupt pin =               0x00
[2363251.783718]   min Grant =                   0x00
[2363251.788486]   max Latency =                 0x00
```

# insmod  pci_cmd_test.ko 
```
[2412827.783597] Vendor: 0x19e5 Device: 0x200, devfun 0
[2412827.788546] Capability: 0x8fe2
[2412827.791760] Power: 0x0
[2412827.794282] cmd  : 0x2
[2412827.796805] MSI/MSI-X: 0x18a
[2412827.799850] enable MSI-X 
[2412827.799852] MSI-X: 0x801f
[2412827.805413] MSI-X: 0x801f
[2412827.808199] PCIe_probe: PCI Command = 0x0002
[2412827.812623] PCIe_probe: PCI Status = 0x0010
[2412827.816961] PCIe_probe: PCI BAR0 = 0x07b0000c
```


# reference
[Linux PCI Express配置空间读写内核实现](http://www.ilinuxkernel.com/files/5/Linux_PCI_Express_Kernel_RW.htm)