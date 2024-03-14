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
//8086:15f9
static const struct pci_device_id PCIe_ids[] = {
	{ PCI_DEVICE(0x8086, 0x15f9), },
};
void pci_header_show(struct pci_dev * dev)
{
	u8 _byte, header_type;
	u16 _word;
	u32 _dword;

#define PRINT(msg, type, reg) \
	pci_read_config_##type(dev, reg, &_##type); \
	pr_info(msg, _##type)

#define PRINT2(msg, type, reg, func) \
	pci_read_config_##type(dev, reg, &_##type); \
	pr_info(msg, _##type, func(_##type))

	pci_read_config_byte(dev, PCI_HEADER_TYPE, &header_type);

	PRINT ("  vendor ID =                   0x%.4x\n", word, PCI_VENDOR_ID);
	PRINT ("  device ID =                   0x%.4x\n", word, PCI_DEVICE_ID);
	PRINT ("  command register =            0x%.4x\n", word, PCI_COMMAND);
	PRINT ("  status register =             0x%.4x\n", word, PCI_STATUS);
	PRINT ("  revision ID =                 0x%.2x\n", byte, PCI_REVISION_ID);
	//PRINT2("  class code =                  0x%.2x (%s)\n", byte, PCI_CLASS_CODE,
	//							pci_classes_str);
	//PRINT ("  sub class code =              0x%.2x\n", byte, PCI_CLASS_SUB_CODE);
	PRINT ("  programming interface =       0x%.2x\n", byte, PCI_CLASS_PROG);
	PRINT ("  cache line =                  0x%.2x\n", byte, PCI_CACHE_LINE_SIZE);
	PRINT ("  latency time =                0x%.2x\n", byte, PCI_LATENCY_TIMER);
	PRINT ("  header type =                 0x%.2x\n", byte, PCI_HEADER_TYPE);
	PRINT ("  BIST =                        0x%.2x\n", byte, PCI_BIST);
	PRINT ("  base address 0 =              0x%.8x\n", dword, PCI_BASE_ADDRESS_0);

	switch (header_type & 0x03) {
	case PCI_HEADER_TYPE_NORMAL:	/* "normal" PCI device */
                pr_info("** normal pci device **");
		PRINT ("  base address 1 =              0x%.8x\n", dword, PCI_BASE_ADDRESS_1);
		PRINT ("  base address 2 =              0x%.8x\n", dword, PCI_BASE_ADDRESS_2);
		PRINT ("  base address 3 =              0x%.8x\n", dword, PCI_BASE_ADDRESS_3);
		PRINT ("  base address 4 =              0x%.8x\n", dword, PCI_BASE_ADDRESS_4);
		PRINT ("  base address 5 =              0x%.8x\n", dword, PCI_BASE_ADDRESS_5);
		PRINT ("  cardBus CIS pointer =         0x%.8x\n", dword, PCI_CARDBUS_CIS);
		PRINT ("  sub system vendor ID =        0x%.4x\n", word, PCI_SUBSYSTEM_VENDOR_ID);
		PRINT ("  sub system ID =               0x%.4x\n", word, PCI_SUBSYSTEM_ID);
		PRINT ("  expansion ROM base address =  0x%.8x\n", dword, PCI_ROM_ADDRESS);
		PRINT ("  interrupt line =              0x%.2x\n", byte, PCI_INTERRUPT_LINE);
		PRINT ("  interrupt pin =               0x%.2x\n", byte, PCI_INTERRUPT_PIN);
		PRINT ("  min Grant =                   0x%.2x\n", byte, PCI_MIN_GNT);
		PRINT ("  max Latency =                 0x%.2x\n", byte, PCI_MAX_LAT);
		break;

	case PCI_HEADER_TYPE_BRIDGE:	/* PCI-to-PCI bridge */

                pr_info("**  pci to pci  bridge **");
		PRINT ("  base address 1 =              0x%.8x\n", dword, PCI_BASE_ADDRESS_1);
		PRINT ("  primary bus number =          0x%.2x\n", byte, PCI_PRIMARY_BUS);
		PRINT ("  secondary bus number =        0x%.2x\n", byte, PCI_SECONDARY_BUS);
		PRINT ("  subordinate bus number =      0x%.2x\n", byte, PCI_SUBORDINATE_BUS);
		PRINT ("  secondary latency timer =     0x%.2x\n", byte, PCI_SEC_LATENCY_TIMER);
		PRINT ("  IO base =                     0x%.2x\n", byte, PCI_IO_BASE);
		PRINT ("  IO limit =                    0x%.2x\n", byte, PCI_IO_LIMIT);
		PRINT ("  secondary status =            0x%.4x\n", word, PCI_SEC_STATUS);
		PRINT ("  memory base =                 0x%.4x\n", word, PCI_MEMORY_BASE);
		PRINT ("  memory limit =                0x%.4x\n", word, PCI_MEMORY_LIMIT);
		PRINT ("  prefetch memory base =        0x%.4x\n", word, PCI_PREF_MEMORY_BASE);
		PRINT ("  prefetch memory limit =       0x%.4x\n", word, PCI_PREF_MEMORY_LIMIT);
		PRINT ("  prefetch memory base upper =  0x%.8x\n", dword, PCI_PREF_BASE_UPPER32);
		PRINT ("  prefetch memory limit upper = 0x%.8x\n", dword, PCI_PREF_LIMIT_UPPER32);
		PRINT ("  IO base upper 16 bits =       0x%.4x\n", word, PCI_IO_BASE_UPPER16);
		PRINT ("  IO limit upper 16 bits =      0x%.4x\n", word, PCI_IO_LIMIT_UPPER16);
		PRINT ("  expansion ROM base address =  0x%.8x\n", dword, PCI_ROM_ADDRESS1);
		PRINT ("  interrupt line =              0x%.2x\n", byte, PCI_INTERRUPT_LINE);
		PRINT ("  interrupt pin =               0x%.2x\n", byte, PCI_INTERRUPT_PIN);
		PRINT ("  bridge control =              0x%.4x\n", word, PCI_BRIDGE_CONTROL);
		break;

	case PCI_HEADER_TYPE_CARDBUS:	/* PCI-to-CardBus bridge */

                pr_info("**  pci to cardbus bridge **");
		PRINT ("  capabilities =                0x%.2x\n", byte, PCI_CB_CAPABILITY_LIST);
		PRINT ("  secondary status =            0x%.4x\n", word, PCI_CB_SEC_STATUS);
		PRINT ("  primary bus number =          0x%.2x\n", byte, PCI_CB_PRIMARY_BUS);
		PRINT ("  CardBus number =              0x%.2x\n", byte, PCI_CB_CARD_BUS);
		PRINT ("  subordinate bus number =      0x%.2x\n", byte, PCI_CB_SUBORDINATE_BUS);
		PRINT ("  CardBus latency timer =       0x%.2x\n", byte, PCI_CB_LATENCY_TIMER);
		PRINT ("  CardBus memory base 0 =       0x%.8x\n", dword, PCI_CB_MEMORY_BASE_0);
		PRINT ("  CardBus memory limit 0 =      0x%.8x\n", dword, PCI_CB_MEMORY_LIMIT_0);
		PRINT ("  CardBus memory base 1 =       0x%.8x\n", dword, PCI_CB_MEMORY_BASE_1);
		PRINT ("  CardBus memory limit 1 =      0x%.8x\n", dword, PCI_CB_MEMORY_LIMIT_1);
		PRINT ("  CardBus IO base 0 =           0x%.4x\n", word, PCI_CB_IO_BASE_0);
		PRINT ("  CardBus IO base high 0 =      0x%.4x\n", word, PCI_CB_IO_BASE_0_HI);
		PRINT ("  CardBus IO limit 0 =          0x%.4x\n", word, PCI_CB_IO_LIMIT_0);
		PRINT ("  CardBus IO limit high 0 =     0x%.4x\n", word, PCI_CB_IO_LIMIT_0_HI);
		PRINT ("  CardBus IO base 1 =           0x%.4x\n", word, PCI_CB_IO_BASE_1);
		PRINT ("  CardBus IO base high 1 =      0x%.4x\n", word, PCI_CB_IO_BASE_1_HI);
		PRINT ("  CardBus IO limit 1 =          0x%.4x\n", word, PCI_CB_IO_LIMIT_1);
		PRINT ("  CardBus IO limit high 1 =     0x%.4x\n", word, PCI_CB_IO_LIMIT_1_HI);
		PRINT ("  interrupt line =              0x%.2x\n", byte, PCI_INTERRUPT_LINE);
		PRINT ("  interrupt pin =               0x%.2x\n", byte, PCI_INTERRUPT_PIN);
		PRINT ("  bridge control =              0x%.4x\n", word, PCI_CB_BRIDGE_CONTROL);
		PRINT ("  subvendor ID =                0x%.4x\n", word, PCI_CB_SUBSYSTEM_VENDOR_ID);
		PRINT ("  subdevice ID =                0x%.4x\n", word, PCI_CB_SUBSYSTEM_ID);
		PRINT ("  PC Card 16bit base address =  0x%.8x\n", dword, PCI_CB_LEGACY_MODE_BASE);
		break;

	default:
		pr_info("unknown header\n");
		break;
    }

#undef PRINT
#undef PRINT2
}

/* PCIe device probe */
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
        pr_info("***************** pci header show ************");
        pci_header_show(pdev);
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
