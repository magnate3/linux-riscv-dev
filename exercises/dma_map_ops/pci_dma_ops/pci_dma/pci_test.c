/*
 * PCIe Configuration Space
 *
 * (C) 2019.10.24 BuddyZhang1 <>
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
#include <linux/aer.h>
#include <linux/interrupt.h>
#include <linux/msi.h>
#include <linux/irq.h>
#include <linux/irqdesc.h>
#include <linux/irqdomain.h>
#include <linux/delay.h>
#define DEV_NAME		"PCIe_demo"
//8086:15f9
static const struct pci_device_id PCIe_ids[] = {
	{ PCI_DEVICE(0x19e5,0x0200 ), },
};
static inline const struct dma_map_ops *test_get_dma_ops(struct device *dev)
{
    if (dev->dma_ops)
    {
         pr_info("dev has dma_ops \n");
            return dev->dma_ops;
    }
    return get_arch_dma_ops(dev->bus);
}
/* PCIe device probe */
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	pr_info("***************** pci bus  info show ************ \n");
        printk("Vendor: %04x Device: %04x, devfun %d, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
        pr_info("dma ops %p  \n",test_get_dma_ops(&pdev->dev));
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
