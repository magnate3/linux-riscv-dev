/*
 * PCIe Configuration Space
 *
 * (C) 2019.10.24  <>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/version.h>
#include <linux/slab.h>

#define DEV_NAME		"PCIe_demo"
//8086:15f9
static const struct pci_device_id PCIe_ids[] = {
	{ PCI_DEVICE(0x8086, 0x15f9), },
};

int print_bars(struct pci_dev *pdev)
{
        int i, iom, iop;
        unsigned long flags;
        unsigned long addr, len;
        static const char *bar_names[PCI_STD_RESOURCE_END + 1]  = {
                "BAR0",
                "BAR1",
                "BAR2",
                "BAR3",
                "BAR4",
                "BAR5",
        };

        iom = 0;
        iop = 0;

        for (i = 0; i < ARRAY_SIZE(bar_names); i++) {
             pr_err("******bar name : %s", bar_names[i]);
                addr = pci_resource_start(pdev, i);
                len = pci_resource_len(pdev, i);
                if (len != 0 && addr != 0) {
                        flags = pci_resource_flags(pdev, i);
                        if (flags & IORESOURCE_MEM) {
                                iom++;
                        } else if (flags & IORESOURCE_IO) {
                                iop++;
                        }
                        pr_info("flags  %lx,  and addr %lx, and len %lx \n", flags & IORESOURCE_MEM,  addr, len);
                }

        }
        return 0;
}
void show_pci_info(struct pci_bus *bus)
{
     struct pci_dev *pdev, *dev;
     if (!bus)
     {
         return ;
     }
     pr_err("bus name : %s, bus ops %p \n", bus->name, bus->ops);
     pdev = bus->self;
     if (pdev)
     {
         printk("Vendor: %#x Device: %#x, devfun %x, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
#if 0
         print_bars(pdev);
#endif
     }

#if 0
    list_for_each_entry(dev, &bus->devices, bus_list) {
                int i;
                if (pci_is_bridge(dev))
                {
                    pr_err("pci is bridge \t ");
                } 
                for (i = 0; i < PCI_NUM_RESOURCES; i++) {
                        struct resource *r = &dev->resource[i];

                        if (r->parent || !r->start || !r->flags)
                                continue;

                        pr_err("PCI: Claiming %s: ", pci_name(dev));
                        pr_err("Resource %d: %016llx..%016llx [%x]\n",
                                 i, (unsigned long long)r->start,
                                 (unsigned long long)r->end,
                                 (unsigned int)r->flags);
                }
    }
#endif
     show_pci_info(bus->parent);
}
#define HINIC_PCI_CFG_REGS_BAR          0
#define HINIC_PCI_DB_BAR                4
static void test_msix_map_region(struct pci_dev *dev, unsigned nr_entries)
{
        resource_size_t phys_addr;
        u32 table_offset;
        unsigned long flags;
        u8 bir;

        pci_read_config_dword(dev, dev->msix_cap + PCI_MSIX_TABLE,
                              &table_offset);
        bir = (u8)(table_offset & PCI_MSIX_TABLE_BIR);
        flags = pci_resource_flags(dev, bir);
        if (!flags || (flags & IORESOURCE_UNSET))
                return ;

        table_offset &= PCI_MSIX_TABLE_OFFSET;
        phys_addr = pci_resource_start(dev, bir) + table_offset;
        pr_info("msix phys_addr : %p \n", phys_addr);
        //return ioremap_nocache(phys_addr, nr_entries * PCI_MSIX_ENTRY_SIZE);
}
void test_pci_ioremap_bar(struct pci_dev *pdev, int bar)
{
        struct resource *res = &pdev->resource[bar];

        /*
 *          * Make sure the BAR is actually a memory resource, not an IO resource
 *                   */
        if (res->flags & IORESOURCE_UNSET || !(res->flags & IORESOURCE_MEM)) {
                dev_warn(&pdev->dev, "can't ioremap BAR %d: %pR\n", bar, res);
                return ;
        }
        pr_info("bar %d phys_addr : %p, len : %ld \n",bar, res->start, resource_size(res));
        //return ioremap_nocache(res->start, resource_size(res));
}
/* PCIe device probe */
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
        pr_info("***************** pci info show ************ \n");
        struct pci_bus *bus = pdev->bus;
        print_bars(pdev);
        test_msix_map_region(pdev, 64);
        test_pci_ioremap_bar(pdev, HINIC_PCI_DB_BAR);
        test_pci_ioremap_bar(pdev, HINIC_PCI_CFG_REGS_BAR);
        pr_info("***************** pci bus  info show ************ \n");
        show_pci_info(bus);
	return 0;
}

/* PCIe device remove */
static void PCIe_remove(struct pci_dev *pdev)
{
}

static struct pci_driver PCIe_demo_driver = {
	.name		= DEV_NAME,
	.id_table	= PCIe_ids,
	.probe		= PCIe_probe,
	.remove		= PCIe_remove,
};

static int __init PCIe_demo_init(void)
{
	return pci_register_driver(&PCIe_demo_driver);
}

static void __exit PCIe_demo_exit(void)
{
	pci_unregister_driver(&PCIe_demo_driver);
}

module_init(PCIe_demo_init);
module_exit(PCIe_demo_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("dda");
MODULE_DESCRIPTION("PCIe Configuration Space Module");
