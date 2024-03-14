/*
 * PCIe Configuration Space
 *
 * (C) 2019.10.24  <@aliyun.com>
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
// 19e5:0200
static const struct pci_device_id PCIe_ids[] = {
	{ PCI_DEVICE(0x19e5, 0x0200), },
};

/* PCIe device probe */
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	u16 device, vendor;
	u16 value;
	u16 power;
	u16 msi;
	u16 msix;
	u16 val;
	u32 bar0;
        u16 cmd; 

	/* PCI/PCI-compatible Type0/1 header read/write */
	pci_read_config_word(pdev, PCI_DEVICE_ID, &device);
	pci_read_config_word(pdev, PCI_VENDOR_ID, &vendor);
	printk("Vendor: %#x Device: %#x, devfun %x\n", vendor, device, pdev->devfn);

	/* PCI Express Capability */
	pcie_capability_read_word(pdev, PCI_EXP_DEVCAP, &value);
	printk("Capability: %#x\n", value);

	/* PCIe Power Management */
	pci_read_config_word(pdev, pdev->pm_cap + PCI_PM_CTRL, &power);
	printk("Power: %#x\n", power);

        pci_read_config_word(pdev, PCI_COMMAND, &cmd);
	printk("cmd  : %#x\n", cmd );
	/* MSI/MSI-X */
	pci_read_config_word(pdev, pdev->msi_cap + PCI_MSI_FLAGS, &msi);
	printk("MSI/MSI-X: %#x\n", msi);
	printk("enable MSI-X ");
	pci_read_config_word(pdev, pdev->msix_cap + PCI_MSIX_FLAGS, &msix);
	printk("MSI-X: %#x\n", msix);
        msix |=  PCI_MSIX_FLAGS_ENABLE; 
        pci_write_config_word(pdev, pdev->msix_cap + PCI_MSIX_FLAGS, msix);
	pci_read_config_word(pdev, pdev->msix_cap + PCI_MSIX_FLAGS, &msix);
	printk("MSI-X: %#x\n", msix);
        pci_read_config_word(pdev, PCI_COMMAND, &val);
	pr_err("%s: PCI Command = 0x%04x\n", __func__, val);
       	pci_read_config_word(pdev, PCI_STATUS, &val);
	pr_err("%s: PCI Status = 0x%04x\n", __func__, val);
	pci_read_config_dword(pdev, PCI_BASE_ADDRESS_0, &bar0);
	pr_err("%s: PCI BAR0 = 0x%08x\n", __func__, bar0);
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
MODULE_AUTHOR(" <@aliyun.com>");
MODULE_DESCRIPTION("PCIe Configuration Space Module");
