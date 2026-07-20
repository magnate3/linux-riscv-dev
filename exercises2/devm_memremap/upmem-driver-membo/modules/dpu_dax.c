/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/acpi.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/idr.h>
#include <linux/kernel.h>
#include <linux/memremap.h>
#include <linux/mm.h>
#include <linux/pagemap.h>
#include <linux/pfn_t.h>
#include <linux/platform_device.h>
#include <linux/version.h>

#include "dpu_region.h"
#include "dpu_region_constants.h"

#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 17, 0)
typedef int vm_fault_t;
#endif

struct class *dpu_dax_class;

static ssize_t size_show(struct device *dev, struct device_attribute *attr,
			 char *buf)
{
	struct dpu_region *region = dev_get_drvdata(dev);

	return sprintf(buf, "%llu\n", region->size);
}

static ssize_t numa_node_show(struct device *dev, struct device_attribute *attr,
			      char *buf)
{
	return sprintf(buf, "%d\n", dev->numa_node);
}

static DEVICE_ATTR_RO(size);
static DEVICE_ATTR_RO(numa_node);

static struct attribute *dpu_dax_region_attrs[] = {
	&dev_attr_size.attr,
	&dev_attr_numa_node.attr,
	NULL,
};

static struct attribute_group dpu_dax_region_attrs_group = {
	.attrs = dpu_dax_region_attrs,
};

const struct attribute_group *dpu_dax_region_attrs_groups[] = {
	&dpu_dax_region_attrs_group, NULL
};

#ifdef __x86_64__
static vm_fault_t dpu_dax_pud_huge_fault(struct vm_fault *vmf, void *vaddr)
{
	struct file *filp = vmf->vma->vm_file;
	struct dpu_region *region = filp->private_data;
	phys_addr_t paddr;
	unsigned long pud_addr = (unsigned long)vaddr & PUD_MASK;
	unsigned long pgoff;
	pfn_t pfn;

	pgoff = linear_page_index(vmf->vma, pud_addr);
	paddr = ((phys_addr_t)__pa(region->base) + pgoff * PAGE_SIZE) &
		PUD_MASK;
	pfn = phys_to_pfn_t(paddr, PFN_DEV | PFN_MAP);

#if LINUX_VERSION_CODE == KERNEL_VERSION(3, 10, 0)
	return vmf_insert_pfn_pud(vmf->vma, (unsigned long)vaddr, vmf->pud, pfn,
				  vmf->flags & FAULT_FLAG_WRITE);
#else
	return vmf_insert_pfn_pud(vmf, pfn, vmf->flags & FAULT_FLAG_WRITE);
#endif
}
#endif

static vm_fault_t dpu_dax_pmd_huge_fault(struct vm_fault *vmf, void *vaddr)
{
	struct file *filp = vmf->vma->vm_file;
	struct dpu_region *region = filp->private_data;
	phys_addr_t paddr;
	unsigned long pmd_addr = (unsigned long)vaddr & PMD_MASK;
	unsigned long pgoff;
	pfn_t pfn;

	pgoff = linear_page_index(vmf->vma, pmd_addr);
	paddr = ((phys_addr_t)__pa(region->base) + pgoff * PAGE_SIZE) &
		PMD_MASK;
	pfn = phys_to_pfn_t(paddr, PFN_DEV | PFN_MAP);

	pr_debug("Mapping pages of size %lx at @v=%llx to @p=%llx\n", PMD_SIZE,
		 (uint64_t)vaddr, paddr);

#if LINUX_VERSION_CODE == KERNEL_VERSION(3, 10, 0)
	return vmf_insert_pfn_pmd(vmf->vma, (unsigned long)vaddr, vmf->pmd, pfn,
				  vmf->flags & FAULT_FLAG_WRITE);
#else
	return vmf_insert_pfn_pmd(vmf, pfn, vmf->flags & FAULT_FLAG_WRITE);
#endif
}

static vm_fault_t dpu_dax_pte_huge_fault(struct vm_fault *vmf, void *vaddr)
{
	struct file *filp = vmf->vma->vm_file;
	struct dpu_region *region = filp->private_data;
	phys_addr_t paddr;
	unsigned long pte_addr = (unsigned long)vaddr & PAGE_MASK;
	unsigned long pgoff;
	pfn_t pfn;

	pgoff = linear_page_index(vmf->vma, pte_addr);
	paddr = ((phys_addr_t)__pa(region->base) + pgoff * PAGE_SIZE) &
		PAGE_MASK;
	pfn = phys_to_pfn_t(paddr, PFN_DEV | PFN_MAP);

	pr_debug("Mapping pages of size %lx at @v=%llx to @p=%llx\n", PAGE_SIZE,
		 (uint64_t)vaddr, paddr);

#if LINUX_VERSION_CODE == KERNEL_VERSION(4, 15, 18) ||                         \
	LINUX_VERSION_CODE == KERNEL_VERSION(3, 10, 0)
	return vm_insert_mixed(vmf->vma, (unsigned long)vaddr, pfn);
#else
	return vmf_insert_mixed(vmf->vma, (unsigned long)vaddr, pfn);
#endif
}

static vm_fault_t dpu_dax_huge_fault(struct vm_fault *vmf,
				     enum page_entry_size pe_size)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 10, 0)
	void *vaddr = (void *)vmf->address;
#else
	void *vaddr = vmf->virtual_address;
#endif

	pr_debug("%s: %s (%#lx - %#lx) size = %d\n", current->comm,
		 (vmf->flags & FAULT_FLAG_WRITE) ? "write" : "read",
		 vmf->vma->vm_start, vmf->vma->vm_end, pe_size);

	switch (pe_size) {
	case PE_SIZE_PTE:
		return dpu_dax_pte_huge_fault(vmf, vaddr);
	case PE_SIZE_PMD:
		return dpu_dax_pmd_huge_fault(vmf, vaddr);
	case PE_SIZE_PUD:
#ifdef __x86_64__
		return dpu_dax_pud_huge_fault(vmf, vaddr);
#elif defined __powerpc64__
		return VM_FAULT_FALLBACK;
#endif
	}

	return VM_FAULT_SIGBUS;
}

static vm_fault_t dpu_dax_fault(
#if LINUX_VERSION_CODE == KERNEL_VERSION(3, 10, 0)
	struct vm_area_struct *vma,
#endif
	struct vm_fault *vmf)
{
	return dpu_dax_huge_fault(vmf, PE_SIZE_PTE);
}

static const struct vm_operations_struct dpu_dax_vm_ops = {
	.huge_fault = dpu_dax_huge_fault,
	.fault = dpu_dax_fault,
};

static int dpu_dax_open(struct inode *inode, struct file *filp)
{
	struct dpu_region *region =
		container_of(inode->i_cdev, struct dpu_region, cdev_dax);

	filp->private_data = region;
	inode->i_flags = S_DAX;

	return 0;
}

static int dpu_dax_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct dpu_region *region = filp->private_data;
	struct dpu_region_address_translation *tr = &region->addr_translate;
	int ret = 0;

	dpu_region_lock(region);

	switch (region->mode) {
	case DPU_REGION_MODE_UNDEFINED:
		if ((tr->capabilities & CAP_PERF) == 0) {
			ret = -EINVAL;
			goto end;
		}

		region->mode = DPU_REGION_MODE_PERF;
		break;
	case DPU_REGION_MODE_HYBRID:
	case DPU_REGION_MODE_SAFE:
		/* TODO: Can we return a value that is not correct
			 * regarding man mmap ?
			 */
		pr_err("device is in safe mode, can't open"
		       " it in perf mode.\n");
		ret = -EPERM;
		goto end;
	case DPU_REGION_MODE_PERF:
		break;
	}

	vma->vm_ops = &dpu_dax_vm_ops;
	/* Caller must set VM_MIXEDMAP on vma if it wants to call this
	 * function [vm_insert_page] from other places, for example from page-fault handler
	 */
	vma->vm_flags |= VM_HUGEPAGE | VM_MIXEDMAP;
#if LINUX_VERSION_CODE == KERNEL_VERSION(3, 10, 0)
	vma->vm_flags2 |= VM_PFN_MKWRITE | VM_HUGE_FAULT;
#endif

end:
	dpu_region_unlock(region);
	return ret;
}

/* Always aligned on 1G page */
static unsigned long dpu_dax_get_unmapped_area(struct file *filp,
					       unsigned long addr,
					       unsigned long len,
					       unsigned long pgoff,
					       unsigned long flags)
{
	unsigned long addr_align = 0;

	pr_debug("%s: Looking for region of size %lu", __func__, len);

	addr_align = current->mm->get_unmapped_area(filp, addr, len + SZ_1G,
						    pgoff, flags);
	if (!IS_ERR_VALUE(addr_align)) {
		/* If the address is already aligned on 1G */
		if (!(addr_align & (SZ_1G - 1)))
			return addr_align;
		return (addr_align + SZ_1G) & ~(SZ_1G - 1);
	}

	pr_err("%s: Failed to align mmap region on 1G, perf will be degraded\n",
	       __func__);

	return current->mm->get_unmapped_area(filp, addr, len, pgoff, flags);
}

static const struct file_operations dpu_dax_fops = {
	.owner = THIS_MODULE,
	.open = dpu_dax_open,
	.mmap = dpu_dax_mmap,
	.get_unmapped_area = dpu_dax_get_unmapped_area,
};

static void dpu_dax_percpu_exit(void *data)
{
	struct percpu_ref *ref = data;
	struct dpu_dax_device *dpu_dax_dev =
		container_of(ref, struct dpu_dax_device, ref);

	wait_for_completion(&dpu_dax_dev->cmp);
	percpu_ref_exit(ref);
}

static void dpu_dax_percpu_release(struct percpu_ref *ref)
{
	struct dpu_dax_device *dpu_dax_dev =
		container_of(ref, struct dpu_dax_device, ref);

	complete(&dpu_dax_dev->cmp);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 3, 0)
static void dpu_dax_percpu_kill(struct dev_pagemap *pgmap)
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(4, 19, 37)
static void dpu_dax_percpu_kill(struct percpu_ref *ref)
#else
static void dpu_dax_percpu_kill(void *data)
#endif
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 3, 0)
	struct percpu_ref *ref = pgmap->ref;
#elif LINUX_VERSION_CODE < KERNEL_VERSION(4, 19, 37)
	struct percpu_ref *ref = data;
#endif

	percpu_ref_kill(ref);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 3, 0)
static void dpu_dax_percpu_cleanup(struct dev_pagemap *pgmap)
{
}

static void dpu_dax_percpu_page_free(struct page *page)
{
}
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 3, 0)
static const struct dev_pagemap_ops dpu_dax_dev_pagemap_ops = {
	.kill = dpu_dax_percpu_kill,
	.cleanup = dpu_dax_percpu_cleanup,
	.page_free = dpu_dax_percpu_page_free,
};
#endif

int dpu_dax_init_device(struct platform_device *pdev, struct dpu_region *region)
{
	struct resource *res;
	struct dpu_dax_device *dpu_dax_dev = &region->dpu_dax_dev;
	struct device *dev = &pdev->dev;
	void *addr;
	int node, pxm;
	int ret;

	res = devm_request_mem_region(dev, pdev->resource->start,
				      resource_size(pdev->resource),
				      "dpu_region");
	if (!res) {
		dev_err(&pdev->dev, "unable to request DPU memory region.\n");
		ret = -EBUSY;
		goto error;
	}

	/* We cannot get the node ID using memory_add_physaddr_to_nid
	 * as our memory regions are not in the numa_meminfo structure
	 */
	pxm = dpu_region_srat_get_pxm(pdev->resource->start);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
	node = pxm_to_online_node(pxm);
#else
	node = acpi_map_pxm_to_online_node(pxm);
#endif
	set_dev_node(&pdev->dev, node);

	init_completion(&dpu_dax_dev->cmp);

	memset(&dpu_dax_dev->ref, 0, sizeof(struct percpu_ref));
	ret = percpu_ref_init(&dpu_dax_dev->ref, dpu_dax_percpu_release, 0,
			      GFP_KERNEL);
	if (ret)
		goto error;

	ret = devm_add_action_or_reset(dev, dpu_dax_percpu_exit,
				       &dpu_dax_dev->ref);
	if (ret)
		goto ref_error;

	/* vmem_altmap is used only if memmap must be stored in our
         * memory region, which we clearly do NOT want.
         * The function returns __va(pdev->resource->start) (which is kernel
         * logical address :)
         */
	dpu_dax_dev->pgmap.ref = &dpu_dax_dev->ref;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 3, 0)
	dpu_dax_dev->pgmap.ops = &dpu_dax_dev_pagemap_ops;
#elif LINUX_VERSION_CODE >= KERNEL_VERSION(4, 19, 37)
	dpu_dax_dev->pgmap.kill = dpu_dax_percpu_kill;
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
	dpu_dax_dev->pgmap.range.start = pdev->resource->start;
	dpu_dax_dev->pgmap.range.end = pdev->resource->end;
	dpu_dax_dev->pgmap.nr_range = 1;
	dpu_dax_dev->pgmap.type = MEMORY_DEVICE_FS_DAX;
	addr = devm_memremap_pages(dev, &dpu_dax_dev->pgmap);
#elif LINUX_VERSION_CODE == KERNEL_VERSION(3, 10, 0) ||                        \
	LINUX_VERSION_CODE >= KERNEL_VERSION(4, 16, 0)
	memcpy(&dpu_dax_dev->pgmap.res, pdev->resource,
	       sizeof(struct resource));
	dpu_dax_dev->pgmap.type = MEMORY_DEVICE_FS_DAX;
	addr = devm_memremap_pages(dev, &dpu_dax_dev->pgmap);
#else
	dpu_dax_dev->pgmap.res = pdev->resource;
	addr = devm_memremap_pages(dev, pdev->resource, &dpu_dax_dev->ref,
				   NULL);
#endif
	if (IS_ERR(addr)) {
		dev_err(&pdev->dev, "%s: devm_memremap_pages failed\n",
			__func__);
		ret = PTR_ERR(addr);
		goto ref_error;
	}

#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 19, 37)
	ret = devm_add_action_or_reset(dev, dpu_dax_percpu_kill,
				       &dpu_dax_dev->ref);
	if (ret)
		goto ref_error;
#endif

	ret = alloc_chrdev_region(&region->devt_dax, 0, 1, "dax");
	if (ret)
		goto ref_error;

	cdev_init(&region->cdev_dax, &dpu_dax_fops);
	region->cdev_dax.owner = THIS_MODULE;

	memset(&region->dev_dax, 0, sizeof(struct device));
	device_initialize(&region->dev_dax);

	region->dev_dax.devt = region->devt_dax;
	region->dev_dax.class = dpu_dax_class;
	region->dev_dax.parent = &pdev->dev;
	dev_set_drvdata(&region->dev_dax, region);
	region->rank.id = ida_simple_get(&dpu_region_ida, 0, 0, GFP_KERNEL);
    region->rank.nid = node;
	dev_set_name(&region->dev_dax, "dax%d.%d", region->rank.id,
		     region->rank.id);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 10, 0)
	ret = cdev_device_add(&region->cdev_dax, &region->dev_dax);
	if (ret)
		goto cdev_error;
#else
	ret = cdev_add(&region->cdev_dax, region->dev_dax.devt, 1);
	if (ret)
		goto cdev_error;

	ret = device_add(&region->dev_dax);
	if (ret)
		goto free_cdev;
#endif

	region->size = resource_size(pdev->resource);
	region->base = addr;

	dev_dbg(&pdev->dev, "DAX region and device allocated\n");

	return 0;
#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 10, 0)
free_cdev:
	cdev_del(&region->cdev_dax);
#endif
cdev_error:
	unregister_chrdev_region(region->devt_dax, 1);
ref_error:
	percpu_ref_kill(&dpu_dax_dev->ref);
error:
	pr_err("%s failed\n", __func__);
	return ret;
}

void dpu_dax_release_device(struct dpu_region *region)
{
	pr_debug("dpu_dax: releasing DAX resources\n");

	cdev_del(&region->cdev_dax);
	device_del(&region->dev_dax);
	unregister_chrdev_region(region->devt_dax, 1);
}
