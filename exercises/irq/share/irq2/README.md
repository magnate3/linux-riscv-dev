
# virq_debug_show
```
static int virq_debug_show(struct seq_file *m, void *private)
{
	unsigned long flags;
	struct irq_desc *desc;
	const char *p;
	static const char none[] = "none";
	void *data;
	int i;
	seq_printf(m, "%-5s  %-7s  %-15s  %-*s  %s\n", "irq", "hwirq",
		      "chip name", (int)(2 * sizeof(void *) + 2), "chip data",
		      "domain name");
	for (i = 1; i < nr_irqs; i++) {
		desc = irq_to_desc(i);
		if (!desc)
			continue;
		raw_spin_lock_irqsave(&desc->lock, flags);
		if (desc->action && desc->action->handler) {
			struct irq_chip *chip;
			seq_printf(m, "%5d  ", i);
			seq_printf(m, "0x%05lx  ", desc->irq_data.hwirq);
			chip = irq_desc_get_chip(desc);
			if (chip && chip->name)
				p = chip->name;
			else
				p = none;
			seq_printf(m, "%-15s  ", p);
			data = irq_desc_get_chip_data(desc);
			seq_printf(m, data ? "0x%p  " : "  %p  ", data);
			if (desc->irq_data.domain && desc->irq_data.domain->of_node)
				p = desc->irq_data.domain->of_node->full_name;
			else
				p = none;
			seq_printf(m, "%s\n", p);
		}
		raw_spin_unlock_irqrestore(&desc->lock, flags);
	}
	return 0;
}
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/irq2/irq.jpg)


# 设置MSI-X

```
 
if ((rc = pci_enable_msix_exact(pdev, irqs, NUM_IRQS)) < 0)
    printk(KERN_ERR "MSIX ENABLE FAILED: %d\n" rc);

...

for (i = 0; i < NUM_IRQS; ++i) {
    irqs[i].entry = i;
    rc = devm_request_irq(&pdev->dev, irqs[i].vector, irq_handler,
                          0, "my_driver", NULL);

    if (rc)
        printk(KERN_ERR "Failed to request irq: %d\n", i); 
}
```

```
static int init_msix(struct hinic_hwdev *hwdev)
{
        struct hinic_hwif *hwif = hwdev->hwif;
        struct pci_dev *pdev = hwif->pdev;
        int nr_irqs, num_aeqs, num_ceqs;
        size_t msix_entries_size;
        int i, err;

        num_aeqs = HINIC_HWIF_NUM_AEQS(hwif);
        num_ceqs = HINIC_HWIF_NUM_CEQS(hwif);
        nr_irqs = MAX_IRQS(HINIC_MAX_QPS, num_aeqs, num_ceqs);
        if (nr_irqs > HINIC_HWIF_NUM_IRQS(hwif))
                nr_irqs = HINIC_HWIF_NUM_IRQS(hwif);

        msix_entries_size = nr_irqs * sizeof(*hwdev->msix_entries);
        hwdev->msix_entries = devm_kzalloc(&pdev->dev, msix_entries_size,
                                           GFP_KERNEL);
        if (!hwdev->msix_entries)
                return -ENOMEM;

        for (i = 0; i < nr_irqs; i++)
                hwdev->msix_entries[i].entry = i;

        err = pci_enable_msix_exact(pdev, hwdev->msix_entries, nr_irqs);
        if (err) {
                dev_err(&pdev->dev, "Failed to enable pci msix\n");
                return err;
        }

        return 0;
}

static inline int pci_enable_msix_exact(struct pci_dev *dev,
                                        struct msix_entry *entries, int nvec)
{
        int rc = pci_enable_msix_range(dev, entries, nvec, nvec);
        if (rc < 0)
                return rc;
        return 0;
}
```

# references

[MSI-X （二）](https://www.pcietech.com/313.html/)