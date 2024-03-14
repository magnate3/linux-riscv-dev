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
#include <linux/iommu.h>
#include "arm64.h"
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

struct arm_smmu_s1_cfg {
        __le64                          *cdptr;
        dma_addr_t                      cdptr_dma;

        struct arm_smmu_ctx_desc {
                u16     asid;
                u64     ttbr;
                u64     tcr;
                u64     mair;
        }                               cd;
};

struct arm_smmu_s2_cfg {
        u16                             vmid;
        u64                             vttbr;
        u64                             vtcr;
};

/* SMMU private data for an IOMMU domain */
enum arm_smmu_domain_stage {
        ARM_SMMU_DOMAIN_S1 = 0,
        ARM_SMMU_DOMAIN_S2,
        ARM_SMMU_DOMAIN_NESTED,
        ARM_SMMU_DOMAIN_BYPASS,
};
struct io_pgtable_ops {
        int (*map)(struct io_pgtable_ops *ops, unsigned long iova,
                   phys_addr_t paddr, size_t size, int prot);
        int (*unmap)(struct io_pgtable_ops *ops, unsigned long iova,
                     size_t size);
        phys_addr_t (*iova_to_phys)(struct io_pgtable_ops *ops,
                                    unsigned long iova);
};
struct arm_smmu_domain {
        struct arm_smmu_device          *smmu;
        struct mutex                    init_mutex; /* Protects smmu pointer */

        struct io_pgtable_ops           *pgtbl_ops;

        enum arm_smmu_domain_stage      stage;
        union {
                struct arm_smmu_s1_cfg  s1_cfg;
                struct arm_smmu_s2_cfg  s2_cfg;
        };

        struct iommu_domain             domain;
};
enum io_pgtable_fmt {
        ARM_32_LPAE_S1,
        ARM_32_LPAE_S2,
        ARM_64_LPAE_S1,
        ARM_64_LPAE_S2,
        ARM_V7S,
        IO_PGTABLE_NUM_FMTS,
};
struct io_pgtable_cfg {
        #define IO_PGTABLE_QUIRK_ARM_NS         BIT(0)
        #define IO_PGTABLE_QUIRK_NO_PERMS       BIT(1)
        #define IO_PGTABLE_QUIRK_TLBI_ON_MAP    BIT(2)
        #define IO_PGTABLE_QUIRK_ARM_MTK_4GB    BIT(3)
        #define IO_PGTABLE_QUIRK_NO_DMA         BIT(4)
        unsigned long                   quirks;
        unsigned long                   pgsize_bitmap;
        unsigned int                    ias;
        unsigned int                    oas;
        const struct iommu_gather_ops   *tlb;
        struct device                   *iommu_dev;

        /* Low-level data specific to the table format */
        union {
                struct {
                        u64     ttbr[2];
                        u64     tcr;
                        u64     mair[2];
                } arm_lpae_s1_cfg;

                struct {
                        u64     vttbr;
                        u64     vtcr;
                } arm_lpae_s2_cfg;

                struct {
                        u32     ttbr[2];
                        u32     tcr;
                        u32     nmrr;
                        u32     prrr;
                } arm_v7s_cfg;
        };
};
struct io_pgtable {
        enum io_pgtable_fmt     fmt;
        void                    *cookie;
        struct io_pgtable_cfg   cfg;
        struct io_pgtable_ops   ops;
};
struct arm_lpae_io_pgtable {
        struct io_pgtable       iop;

        int                     levels;
        size_t                  pgd_size;
        unsigned long           pg_shift;
        unsigned long           bits_per_level;

        void                    *pgd;
};
/* Struct accessors */
#define io_pgtable_to_data(x)                                           \
        container_of((x), struct arm_lpae_io_pgtable, iop)
#define io_pgtable_ops_to_pgtable(x) container_of((x), struct io_pgtable, ops)
#define io_pgtable_ops_to_data(x)                                       \
        io_pgtable_to_data(io_pgtable_ops_to_pgtable(x))
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


static struct arm_smmu_domain *to_smmu_domain(struct iommu_domain *dom)
{
        return container_of(dom, struct arm_smmu_domain, domain);
}
static void test_arm_64_lpae_alloc_pgtable_s1(struct arm_lpae_io_pgtable *data,struct io_pgtable_cfg *cfg)
{
	u64 reg;
	/* TCR */
	reg = (ARM_LPAE_TCR_SH_IS << ARM_LPAE_TCR_SH0_SHIFT) |
	      (ARM_LPAE_TCR_RGN_WBWA << ARM_LPAE_TCR_IRGN0_SHIFT) |
	      (ARM_LPAE_TCR_RGN_WBWA << ARM_LPAE_TCR_ORGN0_SHIFT);

	switch (ARM_LPAE_GRANULE(data)) {
	case SZ_4K:
		reg |= ARM_LPAE_TCR_TG0_4K;
		break;
	case SZ_16K:
		reg |= ARM_LPAE_TCR_TG0_16K;
		break;
	case SZ_64K:
		reg |= ARM_LPAE_TCR_TG0_64K;
		break;
	}

	switch (cfg->oas) {
	case 32:
		reg |= (ARM_LPAE_TCR_PS_32_BIT << ARM_LPAE_TCR_IPS_SHIFT);
		break;
	case 36:
		reg |= (ARM_LPAE_TCR_PS_36_BIT << ARM_LPAE_TCR_IPS_SHIFT);
		break;
	case 40:
		reg |= (ARM_LPAE_TCR_PS_40_BIT << ARM_LPAE_TCR_IPS_SHIFT);
		break;
	case 42:
		reg |= (ARM_LPAE_TCR_PS_42_BIT << ARM_LPAE_TCR_IPS_SHIFT);
		break;
	case 44:
		reg |= (ARM_LPAE_TCR_PS_44_BIT << ARM_LPAE_TCR_IPS_SHIFT);
		break;
	case 48:
		reg |= (ARM_LPAE_TCR_PS_48_BIT << ARM_LPAE_TCR_IPS_SHIFT);
		break;
	default:
		goto out_free_data;
	}

	reg |= (64ULL - cfg->ias) << ARM_LPAE_TCR_T0SZ_SHIFT;
        	/* Disable speculative walks through TTBR1 */
	reg |= ARM_LPAE_TCR_EPD1;
	pr_err("cfg arm_lpae_s1_cfg.tcr  equals ?  reg %d",cfg->arm_lpae_s1_cfg.tcr == reg);
out_free_data:
	return ;
}

static phys_addr_t test_arm_lpae_iova_to_phys(struct io_pgtable_ops *ops,
                                         unsigned long iova)
{
        struct arm_lpae_io_pgtable *data;
        arm_lpae_iopte pte, *ptep = NULL;
        int lvl ;
        if(NULL == ops)
               return 0;
        data = io_pgtable_ops_to_data(ops);
        ptep = data->pgd;
        lvl = ARM_LPAE_START_LVL(data);

        do {
                /* Valid IOPTE pointer? */
                if (!ptep)
                        return 0;

                /* Grab the IOPTE we're interested in */
                ptep += ARM_LPAE_LVL_IDX(iova, lvl, data);
                pte = READ_ONCE(*ptep);

                /* Valid entry? */
                if (!pte)
                        return 0;

                /* Leaf entry? */
                if (iopte_leaf(pte,lvl))
                        goto found_translation;

                /* Take it to the next level */
                ptep = iopte_deref(pte, data);
        } while (++lvl < ARM_LPAE_MAX_LEVELS);

        /* Ran out of page tables to walk */
        return 0;

found_translation:
        iova &= (ARM_LPAE_BLOCK_SIZE(lvl, data) - 1);
        return ((phys_addr_t)iopte_to_pfn(pte,data) << data->pg_shift) | iova;
}
static void test_arm_lpae_dump_ops(struct iommu_domain *domain)
{
        struct io_pgtable_ops *ops = to_smmu_domain(domain)->pgtbl_ops;
        struct arm_lpae_io_pgtable *data = NULL;
        struct io_pgtable_cfg *cfg = NULL;
        struct io_pgtable *iop =NULL; 
        if (!ops)
                return ;
        data = io_pgtable_ops_to_data(ops);
        iop = io_pgtable_ops_to_pgtable(ops);
        cfg = &data->iop.cfg;
        
        switch(iop->fmt){
          case  ARM_64_LPAE_S1 :
              pr_err("ARM_64_LPAE_S1 iop fmt \n");
              break;
          case  ARM_64_LPAE_S2 :
              pr_err("ARM_64_LPAE_S2 iop fmt \n");
              break;
          default:
              pr_err("unknow iop fmt \n");
        }
        test_arm_64_lpae_alloc_pgtable_s1(data,cfg);
        pr_err("cfg: pgsize_bitmap 0x%lx, ias %u-bit\n",
                cfg->pgsize_bitmap, cfg->ias);
        pr_err("data: %d levels, 0x%zx pgd_size, %lu pg_shift, %lu bits_per_level, pgd @ %p\n",
                data->levels, data->pgd_size, data->pg_shift,
                data->bits_per_level, data->pgd);
}
static phys_addr_t
arm_smmu_iova_to_phys(struct iommu_domain *domain, dma_addr_t iova)
{
        struct io_pgtable_ops *ops = to_smmu_domain(domain)->pgtbl_ops;
       
        if (domain->type == IOMMU_DOMAIN_IDENTITY)
                return iova;

        if (!ops)
                return 0;
        return ops->iova_to_phys(ops, iova);
}
phys_addr_t test_iommu_iova_to_phys(struct iommu_domain *domain,
			       unsigned long iova)
{
	if (unlikely(domain->ops->iova_to_phys == NULL))
		return 0;

#if 0
	return domain->ops->iova_to_phys(domain, iova);
#else
	return arm_smmu_iova_to_phys(domain, iova);
#endif
}
#if 0
static void test2_arm_lpae_dump_ops(struct io_pgtable_ops *ops)
{
        struct arm_lpae_io_pgtable *data = io_pgtable_ops_to_data(ops);
        struct io_pgtable_cfg *cfg = &data->iop.cfg;

        pr_err("cfg: pgsize_bitmap 0x%lx, ias %u-bit\n",
                cfg->pgsize_bitmap, cfg->ias);
        pr_err("data: %d levels, 0x%zx pgd_size, %lu pg_shift, %lu bits_per_level, pgd @ %p\n",
                data->levels, data->pgd_size, data->pg_shift,
                data->bits_per_level, data->pgd);
}

#define __FAIL(ops, i)  ({                                              \
                WARN(1, "selftest: test failed for fmt idx %d\n", (i)); \
                test2_arm_lpae_dump_ops(ops);                                 \
                -EFAULT;                                                \
})
static int arm_lpae_run_tests(struct io_pgtable_cfg *cfg)
{
        static const enum io_pgtable_fmt fmts[] = {
                ARM_64_LPAE_S1,
                ARM_64_LPAE_S2,
        };

        int i;
        //int  j;
        //unsigned long iova;
        //size_t size;
        struct io_pgtable_ops *ops;
        struct io_pgtable_cfg *cfg_cookie;

        for (i = 0; i < ARRAY_SIZE(fmts); ++i) {
                cfg_cookie = cfg;
                ops = alloc_io_pgtable_ops(fmts[i], cfg, cfg);
                if (!ops) {
                        pr_err("selftest: failed to allocate io pgtable ops\n");
                        return -ENOMEM;
                }

                /*
 *                  * Initial sanity checks.
 *                                   * Empty page tables shouldn't provide any translations.
 *                                                    */
                if (ops->iova_to_phys(ops, 42))
                        return __FAIL(ops, i);

                if (ops->iova_to_phys(ops, SZ_1G + 42))
                        return __FAIL(ops, i);

                if (ops->iova_to_phys(ops, SZ_2G + 42))
                        return __FAIL(ops, i);
         }
         return 0;
}
#endif
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct page *pages[PAGE_NR];
    int i = 0, j = 0;
    int r=0;
    phys_addr_t pa;
    dma_addr_t iova;
    struct iommu_domain *domain;
    	/* Enable PCI device */
	r = pci_enable_device(pdev);
	if (r < 0) {
		printk("%s ERROR: PCI Device Enable failed.\n", DEV_NAME);
		goto err_enable_pci;
	}
	pci_set_master(pdev);
    /* Page Buffer */
    for (i = 0; i < PAGE_NR; i++) {
		pages[i] = alloc_page(GFP_KERNEL);
		if (!pages[i])
			goto err_page;
    }
    domain = iommu_domain_alloc(&pci_bus_type);
    if (!domain) {
		printk("System Error: Domain failed.\n");
		r = -ENOMEM;
		goto err_page;
     }
/* IOMMU PageFault */
   iommu_set_fault_handler(domain, test_iommu_pf, &pdev);	

	/* Attach Device */
	r = iommu_attach_device(domain, &pdev->dev);
	if (r) {
		printk("System Error: Can't attach iommu device.\n");
		goto err_domain;
	}
        test_arm_lpae_dump_ops(domain);
   	/* IOMMU Mapping */
	for (j = 0; j < PAGE_NR; j++) {
		unsigned long pfn = page_to_pfn(pages[j]);

                iova = VBASE + j * PAGE_SIZE;
                pa = pfn << PAGE_SHIFT;
		r = iommu_map(domain, iova, pa, PAGE_SIZE, IOMMU_READ);
		if (r) {
			printk("System Error: Faild to iommu map.\n");
			goto err_iommu_map;
		}
                /* Verify if the new iova is correctly mapped */
		if (pa != iommu_iova_to_phys(domain, iova)) {
			pr_err("mismatched pa 0x%llx <-> 0x%llx \n",pa, iommu_iova_to_phys(domain, iova)); 
		}
                else {
			pr_err("matched pa 0x%llx <-> 0x%llx, test_arm_lpae_iova_to_phys: 0x%llx\n", pa, iommu_iova_to_phys(domain, iova),\
                                test_arm_lpae_iova_to_phys(to_smmu_domain(domain)->pgtbl_ops, iova));
		}
	}
    pr_info("pci iommu test successfully \n");
err_iommu_map:
     	while (j--)
		iommu_unmap(domain, VBASE + j * PAGE_SIZE, PAGE_SIZE);
err_attach:
     iommu_detach_device(domain, &pdev->dev); 
err_domain:
     iommu_domain_free(domain);
err_page:
    while (i--)
    {
       __free_page(pages[i]);

    }
    pci_disable_device(pdev);
err_enable_pci:
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
