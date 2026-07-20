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
#include <linux/iommu.h>
#define DEV_NAME		"PCIe_demo"

struct iommu_group {
        struct kobject kobj;
        struct kobject *devices_kobj;
        struct list_head devices;
        struct mutex mutex;
        struct blocking_notifier_head notifier;
        void *iommu_data;
        void (*iommu_data_release)(void *iommu_data);
        char *name;
        int id;
        struct iommu_domain *default_domain;
        struct iommu_domain *domain;
};

//8086:15f9
static const struct pci_device_id PCIe_ids[] = {
	{ PCI_DEVICE(0x8086, 0x15f9), },
};

#define PCI_DEVFN(slot, func)	((((slot) & 0x1f) << 3) | ((func) & 0x07))
#define PCI_SLOT(devfn)		(((devfn) >> 3) & 0x1f)
#define PCI_FUNC(devfn)		((devfn) & 0x07)
#define RESV_PHY_SIZE 0x80000000
#define RESV_PHY_ADDR 0x2100000000
unsigned long my_iova = 0x50100000;
/* PCIe device probe */
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
        int ret, order;
        u64 phys_addr;
        size_t size;
        struct iommu_domain *mydomain;
        struct page *pages;
        struct iommu_group *group;
        pr_info("***************** pci iommu bus number %d, slot %d, devfn %d  ************ \n", pdev->bus->number, PCI_SLOT(pdev->devfn), PCI_FUNC(pdev->devfn));
        group = iommu_group_get(&pdev->dev);
        //pr_info("group id %d and group name %s \n", group->id , group->name);
        pr_info("group id %d and group name %s \n", group->id, group->name);
        size = RESV_PHY_SIZE;
        phys_addr = RESV_PHY_ADDR;
        order = get_order(PAGE_SIZE * 2);
        pages = alloc_pages(GFP_KERNEL | __GFP_ZERO, order);
        if (!pages)
                goto err0;
        mydomain = iommu_domain_alloc(&pci_bus_type);
        if (mydomain == NULL) {
                pr_info("dbg_added iommu_domain_alloc error\n");
                return -1;
        } else {
                pr_info("dbg_added iommu_domain_alloc ok\n");
                pr_info("iommu_domain_ops: %p and iommu_ops->domain_alloc %p, iommu_ops->map %p \n", mydomain->ops, (pdev->dev.bus)->iommu_ops->domain_alloc,  (pdev->dev.bus)->iommu_ops->map);
        }

        ret = iommu_attach_device(mydomain, &pdev->dev);
        if (ret) {
                        pr_info("dbg_added iommu_attach_device error\n");
                       goto err1;
        } else {
                pr_info("dbg_added iommu_attach_device ok\n");
        }

        //ret = iommu_map(mydomain, my_iova, page_to_phys(pages), PAGE_SIZE * 2,  IOMMU_READ | IOMMU_WRITE);
        ret = iommu_map(mydomain, my_iova, phys_addr, size, IOMMU_READ | IOMMU_WRITE);
        if (ret < 0) {
                pr_info("dbg_added iommu_map error\n");
                goto err2;
        } else {
                pr_info("dbg_added iommu_map ok\n");
                pr_info(" phys_addr: %016lx, my_iova: %016lx \n", phys_addr, my_iova);
                iommu_unmap(mydomain, my_iova, size);
        }
err2:

        iommu_detach_device(mydomain, &pdev->dev);
err1:

        iommu_domain_free(mydomain);
err0:
        __free_pages(pages, order);
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
