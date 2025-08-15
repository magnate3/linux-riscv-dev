

# dev->msix_cap

```
static void pci_msi_setup_pci_dev(struct pci_dev *dev)
{
        /*
         * Disable the MSI hardware to avoid screaming interrupts
         * during boot.  This is the power on reset default so
         * usually this should be a noop.
         */
        dev->msi_cap = pci_find_capability(dev, PCI_CAP_ID_MSI);
        if (dev->msi_cap)
                pci_msi_set_enable(dev, 0);

        dev->msix_cap = pci_find_capability(dev, PCI_CAP_ID_MSIX);
        if (dev->msix_cap)
                pci_msix_clear_and_set_ctrl(dev, PCI_MSIX_FLAGS_ENABLE, 0);
}

```


# pci_msi_domain_calc_hwirq
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/msi/hwirq.jpg)

```
irq_hw_number_t pci_msi_domain_calc_hwirq(struct pci_dev *dev,
                                          struct msi_desc *desc)
{
        return (irq_hw_number_t)desc->msi_attrib.entry_nr |
                PCI_DEVID(dev->bus->number, dev->devfn) << 11 |
                (pci_domain_nr(dev->bus) & 0xFFFFFFFF) << 27;
}
```


```
static void pci_msi_domain_set_desc(msi_alloc_info_t *arg,
                                    struct msi_desc *desc)
{
        arg->desc = desc;
        arg->hwirq = pci_msi_domain_calc_hwirq(msi_desc_to_pci_dev(desc),
                                               desc);
}

int pci_msi_domain_check_cap(struct irq_domain *domain,
                             struct msi_domain_info *info, struct device *dev)
{
        struct msi_desc *desc = first_pci_msi_entry(to_pci_dev(dev));

        /* Special handling to support __pci_enable_msi_range() */
        if (pci_msi_desc_is_multi_msi(desc) &&
            !(info->flags & MSI_FLAG_MULTI_PCI_MSI))
                return 1;
        else if (desc->msi_attrib.is_msix && !(info->flags & MSI_FLAG_PCI_MSIX))
                return -ENOTSUPP;

        return 0;
}

struct pci_dev *msi_desc_to_pci_dev(struct msi_desc *desc)
{
        return to_pci_dev(desc->dev);
}

int pci_irq_get_node(struct pci_dev *pdev, int vec)
{
        const struct cpumask *mask;

        mask = pci_irq_get_affinity(pdev, vec);
        if (mask)
                return local_memory_node(cpu_to_node(cpumask_first(mask)));
        return dev_to_node(&pdev->dev);
}

struct pci_dev *dev = msi_desc_to_pci_dev(entry);
struct pci_dev *msi_desc_to_pci_dev(struct msi_desc *desc)
{
        return to_pci_dev(desc->dev);
}
EXPORT_SYMBOL(msi_desc_to_pci_dev);

```

# struct msi_controller -- struct irq_domain
```
struct msi_controller {
	struct module *owner;
	struct device *dev;
	struct device_node *of_node;
	struct list_head list;
#ifdef CONFIG_GENERIC_MSI_IRQ_DOMAIN
	struct irq_domain *domain; /////////////////// domain
#endif

	int (*setup_irq)(struct msi_controller *chip, struct pci_dev *dev,
			 struct msi_desc *desc);
	void (*teardown_irq)(struct msi_controller *chip, unsigned int irq);
};

int __weak arch_setup_msi_irq(struct pci_dev *dev, struct msi_desc *desc)
{
	struct msi_controller *chip = pci_msi_controller(dev);
	int err;
	if (!chip || !chip->setup_irq)
		return -EINVAL;
	err = chip->setup_irq(chip, dev, desc);
	if (err < 0)
		return err;
	irq_set_chip_data(desc->irq, chip);
	return 0;
}
```

# irq_find_mapping

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/msi/virq1.jpg)


```
void irq_chip_parent(struct irq_data *data, unsigned long hwirq)
{
        struct irq_chip *chip ;
        struct irq_domain *domain;
        if (!data)
        {
               return ;
        }
        data = data->parent_data;
        if(!data)
        {
               return ;
        }
        chip = data->chip;
        pr_err(" parent chip->name %s  \t", chip->name);
        domain = data->domain;
        if (domain)
        {
                    pr_err("parent domain name :  %s \t", domain->name );
                    pr_err("irq_find_mapping %d \t",irq_find_mapping(domain, hwirq));
        }
        irq_chip_parent(data, hwirq);
}
```


```
[161528.982333] irq info oupt begin ************************ 
 virq:  265, hwirq: 2621456 ,desc->depth 0, parent_irq:  0       
[161528.982334]  leaf chip->name ITS-MSI  
[161528.994626] leaf domain name :  irqchip@0000000202100000-2 and irq_find_mapping 265 
[161528.998530]  parent chip->name ITS  
[161529.006413] parent domain name :  irqchip@0000000202100000-4 
[161529.010145] irq_find_mapping 0 
[161529.016039]  parent chip->name GICv3  
[161529.019339] parent domain name :  irqchip@ffff00000a8e0000 
[161529.023245] irq_find_mapping 0 

[161529.033836] devfn 0,  vendor 19e5 ,device 200 
[161529.038344]  IRQ 265 dev_id find 
```


# msix 和 ioreamp

[pci msi中断](https://zhuanlan.zhihu.com/p/542566695)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/msi/iomem.jpg)

读取msix中断表的地址，table地址位于第bir个bar的table_offset位置
```
pci_read_config_dword(dev, dev->msix_cap + PCI_MSIX_TABLE, &table_offset);
bir = (u8)(table_offset & PCI_MSIX_TABLE_BIR); //读取bir
table_offset &= PCI_MSIX_TABLE_OFFSET; //读取table偏移
phys_addr = pci_resource_start(dev, bir) + table_offset;
```
将msix表做地址映射，后续以内存读写方式进行访问
```
ioremap(phys_addr, nr_entries * PCI_MSIX_ENTRY_SIZE);
```
## iounmap

```
[ 2036.490438] irq info oupt begin ************************ 
 virq:  265, hwirq: 2621456 ,desc->depth 0, parent_irq:  0       
[ 2036.490439]  leaf chip->name ITS-MSI  
[ 2036.502566] leaf domain name :  irqchip@0000000202100000-2 and irq_find_mapping 265 
[ 2036.506384]  parent chip->name ITS  
[ 2036.514182] parent domain name :  irqchip@0000000202100000-4 
[ 2036.517828] irq_find_mapping 0 
[ 2036.523636]  parent chip->name GICv3  
[ 2036.526850] parent domain name :  irqchip@ffff00000a8e0000 
[ 2036.530670] irq_find_mapping 0 

[ 2036.541003] devfn 0,  vendor 19e5 ,device 200 
[ 2036.545426] irq domain name irqchip@0000000202100000-2,  hwirq:  2621456  
[ 2036.552273] pdev phys_addr 8a20000 
[ 2036.555745] ******bar name : BAR0
[ 2036.555746] flags  200,  and addr 80007b00000, and len 20000 
[ 2036.564766] ******bar name : BAR1
[ 2036.564767] ******bar name : BAR2
[ 2036.568067] flags  200,  and addr 80008a20000, and len 8000 
[ 2036.577001] ******bar name : BAR3
[ 2036.577001] ******bar name : BAR4
[ 2036.580300] flags  200,  and addr 80000200000, and len 100000 
[ 2036.589406] ******bar name : BAR5
[ 2036.589407] pdev entry mask 44310000 
[ 2036.596354] pdev entry mask 44310000 
[ 2036.600000] pdev entry mask 44310000 
[ 2036.603648] pdev entry mask 44310000 
[ 2036.607295] pdev entry mask 44310000 
[ 2036.610943] pdev entry mask 44310000 
[ 2036.614589] pdev entry mask 44310000 
[ 2036.618234] pdev entry mask 44310000 
[ 2036.621884] pdev entry mask 44310000 
[ 2036.625530] pdev entry mask 44310000 
[ 2036.629175] pdev entry mask 44310000 
[ 2036.632824] pdev entry mask 44310000 
[ 2036.636470] pdev entry mask 44310000 
[ 2036.640116] pdev entry mask 44310000 
[ 2036.643764] pdev entry mask 44310000 
[ 2036.647411] pdev entry mask 44310000 
[ 2036.651059] pdev entry mask 44310000 
[ 2036.654705] pdev entry mask 44310000 
[ 2036.658351] pdev entry mask 44310000 
[ 2036.661999] pdev entry mask 44310000 
[ 2036.665646] pdev entry mask 44310000 
[ 2036.669291] pdev entry mask 44310000 
[ 2036.672940] pdev entry mask 44310000 
[ 2036.676587] pdev entry mask 44310000 
[ 2036.680232] pdev entry mask 44310000 
[ 2036.683881] pdev entry mask 44310000 
[ 2036.687526] pdev entry mask 44310000 
[ 2036.691175] pdev entry mask 44310000 
[ 2036.694821] pdev entry mask 44310000 
[ 2036.698466] pdev entry mask 44310000 
[ 2036.702115] pdev entry mask 44310000 
[ 2036.705762] pdev entry mask 44310000 
[ 2036.709407]  IRQ 265 dev_id find 
```

```
static void free_msi_irqs(struct pci_dev *dev)
{
	struct msi_desc *entry, *tmp;
	struct attribute **msi_attrs;
	struct device_attribute *dev_attr;
	int i, count = 0;
	list_for_each_entry(entry, &dev->msi_list, list)
		if (entry->irq)
			for (i = 0; i < entry->nvec_used; i++)
				BUG_ON(irq_has_action(entry->irq + i));
	pci_msi_teardown_msi_irqs(dev);
	list_for_each_entry_safe(entry, tmp, &dev->msi_list, list) {
		if (entry->msi_attrib.is_msix) {
			if (list_is_last(&entry->list, &dev->msi_list))
				iounmap(entry->mask_base); ///////////////////
		}
		list_del(&entry->list);
		kfree(entry);
	}

}
static int msix_setup_entries(struct pci_dev *dev, void __iomem *base,
			      struct msix_entry *entries, int nvec)
{
	struct msi_desc *entry;
	int i;
	for (i = 0; i < nvec; i++) {
		entry = alloc_msi_entry(dev);
		if (!entry) {
			if (!i)
				iounmap(base); ///////////////
			else
				free_msi_irqs(dev);
			/* No enough memory. Don't try again */
			return -ENOMEM;
		}
		entry->msi_attrib.is_msix	= 1;
		entry->msi_attrib.is_64		= 1;
		entry->msi_attrib.entry_nr	= entries[i].entry;
		entry->msi_attrib.default_irq	= dev->irq;
		entry->mask_base		= base;
		entry->nvec_used		= 1;
		list_add_tail(&entry->list, &dev->msi_list);
	}
	return 0;
}
```
# __pci_read_msi_msg

```
void __pci_read_msi_msg(struct msi_desc *entry, struct msi_msg *msg)
{
        struct pci_dev *dev = msi_desc_to_pci_dev(entry);

        BUG_ON(dev->current_state != PCI_D0);

        if (entry->msi_attrib.is_msix) {
                void __iomem *base = pci_msix_desc_addr(entry);

                msg->address_lo = readl(base + PCI_MSIX_ENTRY_LOWER_ADDR);
                msg->address_hi = readl(base + PCI_MSIX_ENTRY_UPPER_ADDR);
                msg->data = readl(base + PCI_MSIX_ENTRY_DATA);
        } else {
                int pos = dev->msi_cap;
                u16 data;

                pci_read_config_dword(dev, pos + PCI_MSI_ADDRESS_LO,
                                      &msg->address_lo);
                if (entry->msi_attrib.is_64) {
                        pci_read_config_dword(dev, pos + PCI_MSI_ADDRESS_HI,
                                              &msg->address_hi);
                        pci_read_config_word(dev, pos + PCI_MSI_DATA_64, &data);
                } else {
                        msg->address_hi = 0;
                        pci_read_config_word(dev, pos + PCI_MSI_DATA_32, &data);
                }
                msg->data = data;
        }
}
  
  
void __pci_write_msi_msg(struct msi_desc *entry, struct msi_msg *msg)
{
        struct pci_dev *dev = msi_desc_to_pci_dev(entry);

        if (dev->current_state != PCI_D0 || pci_dev_is_disconnected(dev)) {
                /* Don't touch the hardware now */
        } else if (entry->msi_attrib.is_msix) {
                void __iomem *base = pci_msix_desc_addr(entry);

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
        entry->msg = *msg;
}
```