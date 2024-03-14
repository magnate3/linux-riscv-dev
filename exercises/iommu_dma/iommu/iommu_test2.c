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
#include <linux/iova.h>
#include "arm64.h"
#define DEV_NAME		"PCIe_demo"
#define PVA_SELF_TESTMODE_ADDR_SIZE	0x00000800
#define PVA_SELF_TESTMODE_START_ADDR	0x90000000
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
#if 1
static dma_addr_t iommu_dma_alloc_iova(struct iommu_domain *domain,
                size_t size, dma_addr_t dma_limit, struct device *dev)
{
        struct iommu_dma_cookie *cookie = domain->iova_cookie;
        struct iova_domain *iovad ;
        unsigned long shift, iova_len, iova = 0;

        if(NULL == cookie)
        {
             pr_err("cookie is NULL \n");
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
                iova = alloc_iova_fast(iovad, iova_len, DMA_BIT_MASK(32) >> shift,false);

        if (!iova)
                iova = alloc_iova_fast(iovad, iova_len, dma_limit >> shift,false);

        return (dma_addr_t)iova << shift;
}
#else
static dma_addr_t __iommu_dma_alloc_iova(struct iommu_domain *domain,
		size_t size, u64 dma_limit, struct device *dev)
{
        struct iommu_dma_cookie *cookie = domain->iova_cookie;
        struct iova_domain *iovad ;
        unsigned long shift, iova_len, iova = 0;

        if(NULL == cookie)
        {
             pr_err("cookie is NULL \n");
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
 * 	 * Freeing non-power-of-two-sized allocations back into the IOVA caches
 * 	 	 * will come back to bite us badly, so we have to waste a bit of space
 * 	 	 	 * rounding up anything cacheable to make sure that can't happen. The
 * 	 	 	 	 * order of the unadjusted size will still match upon freeing.
 * 	 	 	 	 	 */
	if (iova_len < (1 << (IOVA_RANGE_CACHE_MAX_SIZE - 1)))
		iova_len = roundup_pow_of_two(iova_len);


	if (domain->geometry.force_aperture)
		dma_limit = min(dma_limit, (u64)domain->geometry.aperture_end);

	/* Try to get PCI devices a SAC address */
	if (dma_limit > DMA_BIT_MASK(32) && dev_is_pci(dev))
		iova = alloc_iova_fast(iovad, iova_len,
				       DMA_BIT_MASK(32) >> shift, false);

	if (!iova)
		iova = alloc_iova_fast(iovad, iova_len, dma_limit >> shift,
				       true);

	return (dma_addr_t)iova << shift;
}
static dma_addr_t iommu_dma_alloc_iova(struct device *dev, size_t size,
				u64 dma_limit)
{
	struct iommu_domain *domain = iommu_get_domain_for_dev(dev);

	return __iommu_dma_alloc_iova(domain, size, dma_limit, dev);
}
#endif
static void iommu_dma_free_iova(struct device *dev,
                dma_addr_t iova, size_t size)
{
        struct iommu_domain *domain = iommu_get_domain_for_dev(dev);
	struct iommu_dma_cookie *cookie = domain->iova_cookie;
        struct iova_domain *iovad = &cookie->iovad;

        /* The MSI case is only ever cleaning up its most recent allocation */
        if (cookie->type == IOMMU_DMA_MSI_COOKIE)
                cookie->msi_iova -= size;
        else
                free_iova_fast(iovad, iova_pfn(iovad, iova),
                                size >> iova_shift(iovad));
}
static void __iommu_dma_free_pages(struct page **pages, int count)
{
        while (count--)
                __free_page(pages[count]);
        kvfree(pages);
}
static struct page **__iommu_dma_alloc_pages(unsigned int count,
                unsigned long order_mask, gfp_t gfp)
{
        struct page **pages;
        unsigned int i = 0, array_size = count * sizeof(*pages);

        order_mask &= (2U << MAX_ORDER) - 1;
        if (!order_mask)
                return NULL;

        if (array_size <= PAGE_SIZE)
                pages = kzalloc(array_size, GFP_KERNEL);
        else
                pages = vzalloc(array_size);
        if (!pages)
                return NULL;

        /* IOMMU can map any pages, so himem can also be used here */
        gfp |= __GFP_NOWARN | __GFP_HIGHMEM;

        while (count) {
                struct page *page = NULL;
                unsigned int order_size;

                /*
 *                  * Higher-order allocations are a convenience rather
 *                                   * than a necessity, hence using __GFP_NORETRY until
 *                                                    * falling back to minimum-order allocations.
 *                                                                     */
                for (order_mask &= (2U << __fls(count)) - 1;
                     order_mask; order_mask &= ~order_size) {
                        unsigned int order = __fls(order_mask);

                        order_size = 1U << order;
                        page = alloc_pages((order_mask - order_size) ?
                                           gfp | __GFP_NORETRY : gfp, order);
                        if (!page)
                                continue;
                        if (!order)
                                break;
                        if (!PageCompound(page)) {
                                split_page(page, order);
                                break;
                        } else if (!split_huge_page(page)) {
                                break;
                        }
                        __free_pages(page, order);
                }
                if (!page) {
                        __iommu_dma_free_pages(pages, i);
                        return NULL;
                }
                count -= order_size;
                while (order_size--)
                        pages[i++] = page++;
        }
        return pages;
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
    int ret;
    phys_addr_t pa, pa_new;
    dma_addr_t iova=PVA_SELF_TESTMODE_START_ADDR;
    unsigned long shift, pg_size, mp_size;
    dma_addr_t tmp_iova, offset;
    struct iommu_domain *domain;
    struct device *dev = &pdev->dev;
    size_t size=PVA_SELF_TESTMODE_ADDR_SIZE;
    void *cpu_va;
    //struct page **pages;
    gfp_t flags=GFP_KERNEL | __GFP_ZERO;
    //unsigned int count;
    unsigned long attrs=DMA_ATTR_SKIP_CPU_SYNC;
    struct iommu_group *group;
     unsigned type;
    	/* Enable PCI device */
	ret = pci_enable_device(pdev);
	if (ret < 0) {
		printk("%s ERROR: PCI Device Enable failed.\n", DEV_NAME);
		goto err_enable_pci;
	}
	pci_set_master(pdev);
    group = iommu_group_get(&pdev->dev);
    if(group){
        pr_info("group id %d and group name %s, default_domain == NULL ? %d \n", group->id, group->name, NULL == group->default_domain);
        if(group->default_domain){
            type = group->default_domain->type; 
        switch(type){
          case  IOMMU_DOMAIN_BLOCKED:
              pr_err("domain IOMMU_DOMAIN_BLOCKED \n");
              break;
          case  IOMMU_DOMAIN_IDENTITY:
              pr_err("domain IOMMU_DOMAIN_IDENTITY\n");
              break;
          case  IOMMU_DOMAIN_UNMANAGED:
              pr_err("domain IOMMU_DOMAIN_UNMANAGED\n");
              break;
          case  IOMMU_DOMAIN_DMA:
              pr_err("domain IOMMU_DOMAIN_DMA\n");
              break;
          default:
              pr_err("unknow iop fmt \n");
        }
        }
    }
    domain = test_iommu_domain_alloc(&pci_bus_type,IOMMU_DOMAIN_UNMANAGED);
    //domain = test_iommu_domain_alloc(&pci_bus_type,IOMMU_DOMAIN_DMA);
    if (!domain) {
		printk("System Error: Domain failed.\n");
		ret = -ENOMEM;
		goto err_page;
     }
/* IOMMU PageFault */
   iommu_set_fault_handler(domain, test_iommu_pf, &pdev);	

	/* Attach Device */
	ret = iommu_attach_device(domain, &pdev->dev);
	if (ret) {
		printk("System Error: Can't attach iommu device.\n");
		goto err_domain;
	}
     test_arm_lpae_dump_ops(domain);
     shift = __ffs(domain->pgsize_bitmap);
     pg_size = 1UL << shift;
     mp_size = pg_size;
     /* Reserve iova range */
#if 1
     tmp_iova = iommu_dma_alloc_iova(domain, size,  DMA_BIT_MASK(64), dev);
     //tmp_iova = iommu_dma_alloc_iova(domain, size, dma_get_mask(dev), dev);
     if(0 == tmp_iova){
		dev_err(dev, "iommu_dma_alloc_iova failed");
		goto err_attach;
	}
#else
      //count = PAGE_ALIGN(size) >> PAGE_SHIFT;
      //pages = __iommu_dma_alloc_pages(count, size  >> PAGE_SHIFT,flags);
      //if (!pages)
      //		goto err_attach;
     tmp_iova = iommu_dma_alloc_iova(dev, size, iova + size - pg_size);
     if(0 == tmp_iova){
		dev_err(dev, "iommu_dma_alloc_iova failed");
		goto err_attach;
	}
#endif
     if (tmp_iova != iova) {
		dev_err(dev, "failed to reserve iova at 0x%llx size 0x%lx\n",
			iova, size);
		goto err_attach;
	}

	/* Allocate a memory first and get a tmp_iova */
      cpu_va = dma_alloc_attrs(dev, size, &tmp_iova, flags, attrs);
      if (!cpu_va)
		goto fail_dma_alloc;
#if 1
   	/* IOMMU Mapping */
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
#endif
#if 1
	dma_free_attrs(dev, size, cpu_va, tmp_iova, attrs);
	iommu_dma_free_iova(dev, tmp_iova, size);
#endif
         pr_info("pci iommu test successfully \n");
         iommu_detach_device(domain, &pdev->dev); 
         iommu_domain_free(domain);
         pci_disable_device(pdev);
         return 0;
fail_map:
	iommu_unmap(domain, iova, offset);
	dma_free_attrs(dev, size, cpu_va, tmp_iova, attrs);
fail_dma_alloc:
	iommu_dma_free_iova(dev, iova, size);
err_attach:
     iommu_detach_device(domain, &pdev->dev); 
err_domain:
     iommu_domain_free(domain);
err_page:
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
