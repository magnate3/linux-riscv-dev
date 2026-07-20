# e1000e
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/e1000/res.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/e1000.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/e1000/hinic.png)

 e1000e_set_interrupt_capability --> pci_enable_msix_range

```
root@ubuntux86:/home/ubuntu# ethtool -i enp0s31f6
driver: e1000e
version: 5.13.0-39-generic
firmware-version: 0.4-4
expansion-rom-version: 
bus-info: 0000:00:1f.6
supports-statistics: yes
supports-test: yes
supports-eeprom-access: yes
supports-register-dump: yes
supports-priv-flags: yes
root@ubuntux86:/home/ubuntu# lspci -s 0000:00:1f.6 -vv 
00:1f.6 Ethernet controller: Intel Corporation Ethernet Connection (14) I219-LM (rev 11)
        Subsystem: Dell Ethernet Connection (14) I219-LM
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Interrupt: pin A routed to IRQ 125
        Region 0: Memory at 70280000 (32-bit, non-prefetchable) [size=128K]
        Capabilities: [c8] Power Management version 3
                Flags: PMEClk- DSI+ D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold+)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=1 PME-
        Capabilities: [d0] MSI: Enable+ Count=1/1 Maskable- 64bit+
                Address: 00000000fee00298  Data: 0000
        Kernel driver in use: e1000e
        Kernel modules: e1000e
root@ubuntux86:/work/pci_test# lspci -s 0000:00:1f.6 -n
00:1f.6 0200: 8086:15f9 (rev 11)
```

##  PCIe_demo 0000:00:1f.6: BAR 0: can't reserve [mem 0x70280000-0x7029ffff]
**not to release mem region by pci_release_mem_regions(dev);**


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/e1000/release.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/e1000/bar0.png)


```
static void pci_msi_set_enable(struct pci_dev *dev, int enable)
{
        u16 control;

        pci_read_config_word(dev, dev->msi_cap + PCI_MSI_FLAGS, &control);
        control &= ~PCI_MSI_FLAGS_ENABLE;
        if (enable)
                control |= PCI_MSI_FLAGS_ENABLE;
        pci_write_config_word(dev, dev->msi_cap + PCI_MSI_FLAGS, control);
}
```

```
static int msi_capability_init(struct pci_dev *dev, int nvec,
                               struct irq_affinity *affd)
{
        struct msi_desc *entry;
        int ret;
        unsigned mask;

        pci_msi_set_enable(dev, 0);     /* Disable MSI during set up */

        entry = msi_setup_entry(dev, nvec, affd);
        if (!entry)
                return -ENOMEM;

        /* All MSIs are unmasked by default; mask them all */
        mask = msi_mask(entry->msi_attrib.multi_cap);
        msi_mask_irq(entry, mask, mask);

        list_add_tail(&entry->list, dev_to_msi_list(&dev->dev));

        /* Configure MSI capability structure */
        ret = pci_msi_setup_msi_irqs(dev, nvec, PCI_CAP_ID_MSI);
        if (ret) {
                msi_mask_irq(entry, mask, ~mask);
                free_msi_irqs(dev);
                return ret;
        }

        ret = msi_verify_entries(dev);
        if (ret) {
                msi_mask_irq(entry, mask, ~mask);
                free_msi_irqs(dev);
                return ret;
        }

        ret = populate_msi_sysfs(dev);
        if (ret) {
                msi_mask_irq(entry, mask, ~mask);
                free_msi_irqs(dev);
                return ret;
        }
}
```

```
ubuntu@ubuntux86:/boot$ grep CONFIG_PCI_MSI_IRQ_DOMAIN  config-5.13.0-39-generic
CONFIG_PCI_MSI_IRQ_DOMAIN=y
ubuntu@ubuntux86:/boot$ 
#ifdef CONFIG_PCI_MSI_IRQ_DOMAIN
static int pci_msi_setup_msi_irqs(struct pci_dev *dev, int nvec, int type)
{
        struct irq_domain *domain;

        domain = dev_get_msi_domain(&dev->dev);
        if (domain && irq_domain_is_hierarchy(domain))
                return msi_domain_alloc_irqs(domain, &dev->dev, nvec);

        return arch_setup_msi_irqs(dev, nvec, type);
}

static void pci_msi_teardown_msi_irqs(struct pci_dev *dev)
{
        struct irq_domain *domain;

        domain = dev_get_msi_domain(&dev->dev);
        if (domain && irq_domain_is_hierarchy(domain))
                msi_domain_free_irqs(domain, &dev->dev);
        else
                arch_teardown_msi_irqs(dev);
}
#else
#define pci_msi_setup_msi_irqs          arch_setup_msi_irqs
#define pci_msi_teardown_msi_irqs       arch_teardown_msi_irqs
#endif
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/e1000/msi.png)

```
static int msi_domain_alloc(struct irq_domain *domain, unsigned int virq,
                            unsigned int nr_irqs, void *arg)
{
        struct msi_domain_info *info = domain->host_data;
        struct msi_domain_ops *ops = info->ops;
        irq_hw_number_t hwirq = ops->get_hwirq(info, arg);
        int i, ret;

        if (irq_find_mapping(domain, hwirq) > 0)
                return -EEXIST;

        if (domain->parent) {
                ret = irq_domain_alloc_irqs_parent(domain, virq, nr_irqs, arg);
                if (ret < 0)
                        return ret;
        }

        for (i = 0; i < nr_irqs; i++) {
                ret = ops->msi_init(domain, info, virq + i, hwirq + i, arg);
                if (ret < 0) {
                        if (ops->msi_free) {
                                for (i--; i > 0; i--)
                                        ops->msi_free(domain, info, virq + i);
                        }
                        irq_domain_free_irqs_top(domain, virq, nr_irqs);
                        return ret;
                }
        }

        return 0;
}
```

## domain 关联 chip 和 irq_data


 ** irq_domain_set_info --> irq_domain_set_hwirq_and_chip  **
```
int irq_domain_set_hwirq_and_chip(struct irq_domain *domain, unsigned int virq,
                                  irq_hw_number_t hwirq, struct irq_chip *chip,
                                  void *chip_data)
{
        struct irq_data *irq_data = irq_domain_get_irq_data(domain, virq);

        if (!irq_data)
                return -ENOENT;

        irq_data->hwirq = hwirq;
        irq_data->chip = chip ? chip : &no_irq_chip;
        irq_data->chip_data = chip_data;

        return 0;
}
```

## 在哪个函数设置domain->host_data

```
__irq_domain_add
        domain->ops = ops;
        domain->host_data = host_data;
        domain->hwirq_max = hwirq_max;
        domain->revmap_size = size;
        domain->revmap_direct_max_irq = direct_max;       

```
# msi地址

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/e1000/base.png)

```
msi_setup_entry(struct pci_dev *dev, int nvec, struct irq_affinity *affd)
{
        struct irq_affinity_desc *masks = NULL;
        struct msi_desc *entry;
        u16 control;

        if (affd)
                masks = irq_create_affinity_masks(nvec, affd);

        /* MSI Entry Initialization */
        entry = alloc_msi_entry(&dev->dev, nvec, masks);
        if (!entry)
                goto out;

        pci_read_config_word(dev, dev->msi_cap + PCI_MSI_FLAGS, &control);

        entry->msi_attrib.is_msix       = 0;
        entry->msi_attrib.is_64         = !!(control & PCI_MSI_FLAGS_64BIT);
        entry->msi_attrib.is_virtual    = 0;
        entry->msi_attrib.entry_nr      = 0;
        entry->msi_attrib.maskbit       = !!(control & PCI_MSI_FLAGS_MASKBIT);
        entry->msi_attrib.default_irq   = dev->irq;     /* Save IOAPIC IRQ */
        entry->msi_attrib.multi_cap     = (control & PCI_MSI_FLAGS_QMASK) >> 1;
        entry->msi_attrib.multiple      = ilog2(__roundup_pow_of_two(nvec));

        if (control & PCI_MSI_FLAGS_64BIT)
                entry->mask_pos = dev->msi_cap + PCI_MSI_MASK_64;
        else
                entry->mask_pos = dev->msi_cap + PCI_MSI_MASK_32;

        /* Save the initial mask status */
        if (entry->msi_attrib.maskbit)
                pci_read_config_dword(dev, entry->mask_pos, &entry->masked);

out:
        kfree(masks);
        return entry;
}

```

**linux2.6**

entry->mask_base = (void __iomem *)(long)msi_mask_bits_reg(pos,
				is_64bit_address(control));
```
\define msi_mask_bits_reg(base, is64bit) \
	( (is64bit == 1) ? base+PCI_MSI_MASK_BIT : base+PCI_MSI_MASK_BIT-4)
static int msi_capability_init(struct pci_dev *dev)
{
	struct msi_desc *entry;
	int pos, ret;
	u16 control;

	msi_set_enable(dev, 0);	/* Ensure msi is disabled as I set it up */

   	pos = pci_find_capability(dev, PCI_CAP_ID_MSI);
	pci_read_config_word(dev, msi_control_reg(pos), &control);
	/* MSI Entry Initialization */
	entry = alloc_msi_entry();
	if (!entry)
		return -ENOMEM;

	entry->msi_attrib.type = PCI_CAP_ID_MSI;
	entry->msi_attrib.is_64 = is_64bit_address(control);
	entry->msi_attrib.entry_nr = 0;
	entry->msi_attrib.maskbit = is_mask_bit_support(control);
	entry->msi_attrib.masked = 1;
	entry->msi_attrib.default_irq = dev->irq;	/* Save IOAPIC IRQ */
	entry->msi_attrib.pos = pos;
	if (is_mask_bit_support(control)) {
		entry->mask_base = (void __iomem *)(long)msi_mask_bits_reg(pos,
				is_64bit_address(control));
	}
	entry->dev = dev;
	if (entry->msi_attrib.maskbit) {
		unsigned int maskbits, temp;
		/* All MSIs are unmasked by default, Mask them all */
		pci_read_config_dword(dev,
			msi_mask_bits_reg(pos, is_64bit_address(control)),
			&maskbits);
		temp = (1 << multi_msi_capable(control));
		temp = ((temp - 1) & ~temp);
		maskbits |= temp;
		pci_write_config_dword(dev,
			msi_mask_bits_reg(pos, is_64bit_address(control)),
			maskbits);
		entry->msi_attrib.maskbits_mask = temp;
	}
	list_add_tail(&entry->list, &dev->msi_list);

	/* Configure MSI capability structure */
	ret = arch_setup_msi_irqs(dev, 1, PCI_CAP_ID_MSI);
	if (ret) {
		msi_free_irqs(dev);
		return ret;
	}

	/* Set MSI enabled bits	 */
	pci_intx_for_msi(dev, 0);
	msi_set_enable(dev, 1);
	dev->msi_enabled = 1;

	dev->irq = entry->irq;
	return 0;
}
```

## msi和msix的pci_msi_domain_write_msg的区别

```
 
 static void __iomem *pci_msix_desc_addr(struct msi_desc *desc)
{
        if (desc->msi_attrib.is_virtual)
                return NULL;

        return desc->mask_base +
                desc->msi_attrib.entry_nr * PCI_MSIX_ENTRY_SIZE;
}


void __pci_write_msi_msg(struct msi_desc *entry, struct msi_msg *msg)
{
        struct pci_dev *dev = msi_desc_to_pci_dev(entry);

        if (dev->current_state != PCI_D0 || pci_dev_is_disconnected(dev)) {
                /* Don't touch the hardware now */
        } else if (entry->msi_attrib.is_msix) {  /////is_msix
                void __iomem *base = pci_msix_desc_addr(entry); // 通过ioremap

                if (!base)
                        goto skip;

                writel(msg->address_lo, base + PCI_MSIX_ENTRY_LOWER_ADDR);
                writel(msg->address_hi, base + PCI_MSIX_ENTRY_UPPER_ADDR);
                writel(msg->data, base + PCI_MSIX_ENTRY_DATA);
        } else {
                int pos = dev->msi_cap;
                u16 msgctl;

                pci_read_config_word(dev, pos + PCI_MSI_FLAGS, &msgctl);
                msgctl &= ~PCI_MSI_FLAGS_QSIZE;
                msgctl |= entry->msi_attrib.multiple << 4;
                pci_write_config_word(dev, pos + PCI_MSI_FLAGS, msgctl);

                pci_write_config_dword(dev, pos + PCI_MSI_ADDRESS_LO,
                                       msg->address_lo);
                if (entry->msi_attrib.is_64) {
                        pci_write_config_dword(dev, pos + PCI_MSI_ADDRESS_HI,
                                               msg->address_hi);
                        pci_write_config_word(dev, pos + PCI_MSI_DATA_64,
                                              msg->data);
                } else {
                        pci_write_config_word(dev, pos + PCI_MSI_DATA_32,
                                              msg->data);
                }
        }

skip:
        entry->msg = *msg;

        if (entry->write_msi_msg)
                entry->write_msi_msg(entry, entry->write_msi_msg_data);

}

```

# vfio_pci_enable
```
        msix_pos = pdev->msix_cap;
        if (msix_pos) {
                u16 flags;
                u32 table;

                pci_read_config_word(pdev, msix_pos + PCI_MSIX_FLAGS, &flags);
                pci_read_config_dword(pdev, msix_pos + PCI_MSIX_TABLE, &table);

                vdev->msix_bar = table & PCI_MSIX_TABLE_BIR;
                vdev->msix_offset = table & PCI_MSIX_TABLE_OFFSET;
                vdev->msix_size = ((flags & PCI_MSIX_FLAGS_QSIZE) + 1) * 16;
        } else
                vdev->msix_bar = 0xFF;

```