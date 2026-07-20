/*
 * PCIe Configuration Space
 *
 * (C) 2019.10.24 <.com>
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
#include <linux/iova.h>
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
#if 0
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
        //ret = arm_smmu_map(mydomain, my_iova, phys_addr, size, IOMMU_READ | IOMMU_WRITE);
        ret = (pdev->dev.bus)->iommu_ops->map(mydomain, my_iova, phys_addr, size, IOMMU_READ | IOMMU_WRITE);
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

#else

#define PAGE_NR			4
#define VBASE			0x00000000
static int test_iommu_pf(struct iommu_domain *domain,
		struct device *dev, unsigned long iova, int flags, void *token)
{
	return -ENOSYS;
}

enum iommu_dma_cookie_type {
        IOMMU_DMA_IOVA_COOKIE,
        IOMMU_DMA_MSI_COOKIE,
};

struct iommu_dma_cookie {
        enum iommu_dma_cookie_type      type;
        union {
                /* Full allocator for IOMMU_DMA_IOVA_COOKIE */
                struct iova_domain      iovad;
                /* Trivial linear page allocator for IOMMU_DMA_MSI_COOKIE */
                dma_addr_t              msi_iova;
        };
        struct list_head                msi_page_list;
        spinlock_t                      msi_lock;
};
static bool test_valid_domain(struct iommu_domain *domain)
{
        struct iommu_dma_cookie *cookie = domain->iova_cookie;
        struct iova_domain *iovad;
        if(NULL == cookie){
            pr_err("%s cookie is NULL \n",__func__);
            return false;
        }
        iovad = &cookie->iovad;
        if(NULL == iovad)
        {
             pr_err("%s cookie is NULL \n",__func__);
             return false;
        }
        pr_info("domain is valid \n");
        return true;
}
static dma_addr_t iommu_dma_alloc_iova(struct iommu_domain *domain,
                size_t size, dma_addr_t dma_limit, struct device *dev)
{
        struct iommu_dma_cookie *cookie = domain->iova_cookie;
        struct iova_domain *iovad;
        unsigned long shift, iova_len, iova = 0;
        if(NULL == cookie){
            pr_info("cookie is NULL \n");
            return 0;
        }
        iovad = &cookie->iovad;
        if(NULL == iovad)
        {
             pr_err("iovad is NULL \n");
             return 0;
        }
        if (cookie->type == IOMMU_DMA_MSI_COOKIE) {
                cookie->msi_iova += size;
                return cookie->msi_iova - size;
        }

        shift = iova_shift(iovad);
        iova_len = size >> shift;
        /*
 *          * Freeing non-power-of-two-sized allocations back into the IOVA caches
 *                   * will come back to bite us badly, so we have to waste a bit of space
 *                            * rounding up anything cacheable to make sure that can't happen. The
 *                                     * order of the unadjusted size will still match upon freeing.
 *                                              */
        if (iova_len < (1 << (IOVA_RANGE_CACHE_MAX_SIZE - 1)))
                iova_len = roundup_pow_of_two(iova_len);

        if (domain->geometry.force_aperture)
                dma_limit = min(dma_limit, domain->geometry.aperture_end);

        /* Try to get PCI devices a SAC address */
        if (dma_limit > DMA_BIT_MASK(32) && dev_is_pci(dev))
                iova = alloc_iova_fast(iovad, iova_len, DMA_BIT_MASK(32) >> shift, false);

        if (!iova)
                iova = alloc_iova_fast(iovad, iova_len, dma_limit >> shift, false);

        return (dma_addr_t)iova << shift;
}

static void iommu_dma_free_iova(struct iommu_domain *domain,
                dma_addr_t iova, size_t size)
{
        struct iommu_dma_cookie *cookie = domain->iova_cookie;
        struct iova_domain *iovad = &cookie->iovad;

        /* The MSI case is only ever cleaning up its most recent allocation */
        if (cookie->type == IOMMU_DMA_MSI_COOKIE)
                cookie->msi_iova -= size;
        else
                free_iova_fast(iovad, iova_pfn(iovad, iova),
                                size >> iova_shift(iovad));
}
static void *pva_dma_alloc_and_map_at( struct iommu_domain *domain ,struct device *dev, size_t size,
				      dma_addr_t iova, gfp_t flags,
				      unsigned long attrs)
{
	//struct iommu_domain *domain;
	unsigned long shift, pg_size, mp_size;
	dma_addr_t tmp_iova, offset;
	phys_addr_t pa, pa_new;
	void *cpu_va;
	int ret;

#if 0
	domain = iommu_get_domain_for_dev(dev);
	if (!domain) {
		dev_err(dev, "IOMMU domain not found");
		return NULL;
	}
#endif
	shift = __ffs(domain->pgsize_bitmap);
	pg_size = 1UL << shift;
	mp_size = pg_size;
	pr_info("start to alloc iova \n");

	/* Reserve iova range */
	tmp_iova = iommu_dma_alloc_iova(domain, size,  dma_get_mask(dev), dev);
	//tmp_iova = iommu_dma_alloc_iova(domain, size, iova + size - pg_size);
	if (tmp_iova != iova) {
		dev_err(dev, "failed to reserve iova at 0x%llx size 0x%lx, new iova 0x%llx \n",
			iova, size, tmp_iova);
		//return NULL;
	}
	if (!tmp_iova){

		dev_err(dev, "failed to alloc iova \n");
		return NULL;
	}

	pr_info("start to alloc dma attr\n");
	/* Allocate a memory first and get a tmp_iova */
	cpu_va = dma_alloc_attrs(dev, size, &tmp_iova, flags, attrs);
	if (!cpu_va)
		goto fail_dma_alloc;

	/* Use tmp_iova to remap non-contiguous pages to the desired iova */
	for (offset = 0; offset < size; offset += mp_size) {
		dma_addr_t cur_iova = tmp_iova + offset;

		mp_size = pg_size;
		pa = iommu_iova_to_phys(domain, cur_iova);
		/* Checking if next physical addresses are contiguous */
		for ( ; offset + mp_size < size; mp_size += pg_size) {
			pa_new = iommu_iova_to_phys(domain, cur_iova + mp_size);
			if (pa + mp_size != pa_new)
				break;
		}

		/* Remap the contiguous physical addresses together */
		ret = iommu_map(domain, iova + offset, pa, mp_size,
				IOMMU_READ | IOMMU_WRITE);
		if (ret) {
			dev_err(dev, "failed to map pa %llx va %llx size %lx\n",
				pa, iova + offset, mp_size);
			goto fail_map;
		}

		/* Verify if the new iova is correctly mapped */
		if (pa != iommu_iova_to_phys(domain, iova + offset)) {
			dev_err(dev, "mismatched pa 0x%llx <-> 0x%llx\n",
				pa, iommu_iova_to_phys(domain, iova + offset));
			goto fail_map;
		}
	}

	/* Unmap and free the tmp_iova since target iova is linked */
	iommu_unmap(domain, tmp_iova, size);
        dma_free_attrs(dev,
			size, cpu_va,
			tmp_iova,
			DMA_ATTR_SKIP_CPU_SYNC);
	iommu_dma_free_iova(domain, tmp_iova, size);

	return cpu_va;

fail_map:
	iommu_unmap(domain, iova, offset);
	dma_free_attrs(dev, size, cpu_va, tmp_iova, attrs);
fail_dma_alloc:
	iommu_dma_free_iova(domain, iova, size);

	return NULL;
}
static struct iommu_domain *test_iommu_domain_alloc(struct bus_type *bus,
                                                 unsigned type)
{
        struct iommu_domain *domain;

        if (bus == NULL || bus->iommu_ops == NULL)
                return NULL;

        domain = bus->iommu_ops->domain_alloc(type);
        if (!domain)
                return NULL;

        domain->ops  = bus->iommu_ops;
        domain->type = type;
        /* Assume all sizes by default; the driver may override this later */
        domain->pgsize_bitmap  = bus->iommu_ops->pgsize_bitmap;

        return domain;
}

static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    int r=0;
    struct iommu_domain *domain;
    struct device *dev = &pdev->dev; 
    	/* Enable PCI device */
	r = pci_enable_device(pdev);
	if (r < 0) {
		printk("%s ERROR: PCI Device Enable failed.\n", DEV_NAME);
		goto out;
	}
	pci_set_master(pdev);
    domain = iommu_domain_alloc(&pci_bus_type);
#if 0
    // cause coredump
    domain = test_iommu_domain_alloc(&pci_bus_type,IOMMU_DOMAIN_DMA);
    if (!domain) {
		printk("System Error: Domain failed.\n");
		r = -ENOMEM;
		goto err_domain;
     }
     ///if (domain->type == IOMMU_DOMAIN_DMA) {
     ///           if (iommu_dma_init_domain(domain, dma_base, size, dev))
     ///                   goto out_err;

     ///           dev->dma_ops = &iommu_dma_ops;
     ///}
#endif
     if(!test_valid_domain(domain)) 
     {
	  goto err_domain;
           
     }
/* IOMMU PageFault */
   iommu_set_fault_handler(domain, test_iommu_pf, &pdev);	

	/* Attach Device */
    r = iommu_attach_device(domain, &pdev->dev);
    if (r) {
		printk("System Error: Can't attach iommu device.\n");
		goto err_domain;
	}
     pva_dma_alloc_and_map_at(domain,&pdev->dev,PAGE_SIZE*PAGE_NR, 0, GFP_KERNEL | __GFP_ZERO,DMA_ATTR_SKIP_CPU_SYNC);
err_attach:
     iommu_detach_device(domain, &pdev->dev); 
err_domain:
     iommu_domain_free(domain);
err_enable_pci:
    pci_disable_device(pdev);
out:
    return 0;
}
#endif
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
MODULE_AUTHOR(" <.com>");
MODULE_DESCRIPTION("PCIe Configuration Space Module");
