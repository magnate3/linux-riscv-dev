


#   alloc_iova


+ 1  alloc_iova    

+ 2  dma_map_page   

+ 3  dma_unmap_page   

+ 4 __free_iova    

```
static int allocate_trash_buffer(struct ipu6_mmu *mmu)
{
	unsigned int n_pages = PHYS_PFN(PAGE_ALIGN(IPU6_MMUV2_TRASH_RANGE));
	struct iova *iova;
	unsigned int i;
	dma_addr_t dma;
	unsigned long iova_addr;
	int ret;

	/* Allocate 8MB in iova range */
	iova = alloc_iova(&mmu->dmap->iovad, n_pages,
			  PHYS_PFN(mmu->dmap->mmu_info->aperture_end), 0);
	if (!iova) {
		dev_err(mmu->dev, "cannot allocate iova range for trash\n");
		return -ENOMEM;
	}

	dma = dma_map_page(mmu->dmap->mmu_info->dev, mmu->trash_page, 0,
			   PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (dma_mapping_error(mmu->dmap->mmu_info->dev, dma)) {
		dev_err(mmu->dmap->mmu_info->dev, "Failed to map trash page\n");
		ret = -ENOMEM;
		goto out_free_iova;
	}

	mmu->pci_trash_page = dma;

	/*
	 * Map the 8MB iova address range to the same physical trash page
	 * mmu->trash_page which is already reserved at the probe
	 */
	iova_addr = iova->pfn_lo;
	for (i = 0; i < n_pages; i++) {
		ret = ipu6_mmu_map(mmu->dmap->mmu_info, PFN_PHYS(iova_addr),
				   mmu->pci_trash_page, PAGE_SIZE);
		if (ret) {
			dev_err(mmu->dev,
				"mapping trash buffer range failed\n");
			goto out_unmap;
		}

		iova_addr++;
	}

	mmu->iova_trash_page = PFN_PHYS(iova->pfn_lo);
	dev_dbg(mmu->dev, "iova trash buffer for MMUID: %d is %u\n",
		mmu->mmid, (unsigned int)mmu->iova_trash_page);
	return 0;

out_unmap:
	ipu6_mmu_unmap(mmu->dmap->mmu_info, PFN_PHYS(iova->pfn_lo),
		       PFN_PHYS(iova_size(iova)));
	dma_unmap_page(mmu->dmap->mmu_info->dev, mmu->pci_trash_page,
		       PAGE_SIZE, DMA_BIDIRECTIONAL);
out_free_iova:
	__free_iova(&mmu->dmap->iovad, iova);
	return ret;
}
```
**demo2 iommu_map**

+ 1  __get_free_pages   

+ 2  alloc_iova

+ 3   iova_dma_addr   

+ 4  iommu_map

```
void *tegra_drm_alloc(struct tegra_drm *tegra, size_t size, dma_addr_t *dma)
{
	struct iova *alloc;
	void *virt;
	gfp_t gfp;
	int err;

	if (tegra->domain)
		size = iova_align(&tegra->carveout.domain, size);
	else
		size = PAGE_ALIGN(size);

	gfp = GFP_KERNEL | __GFP_ZERO;
	if (!tegra->domain) {
		/*
		 * Many units only support 32-bit addresses, even on 64-bit
		 * SoCs. If there is no IOMMU to translate into a 32-bit IO
		 * virtual address space, force allocations to be in the
		 * lower 32-bit range.
		 */
		gfp |= GFP_DMA;
	}

	virt = (void *)__get_free_pages(gfp, get_order(size));
	if (!virt)
		return ERR_PTR(-ENOMEM);

	if (!tegra->domain) {
		/*
		 * If IOMMU is disabled, devices address physical memory
		 * directly.
		 */
		*dma = virt_to_phys(virt);
		return virt;
	}

	alloc = alloc_iova(&tegra->carveout.domain,
			   size >> tegra->carveout.shift,
			   tegra->carveout.limit, true);
	if (!alloc) {
		err = -EBUSY;
		goto free_pages;
	}

	*dma = iova_dma_addr(&tegra->carveout.domain, alloc);
	err = iommu_map(tegra->domain, *dma, virt_to_phys(virt),
			size, IOMMU_READ | IOMMU_WRITE, GFP_KERNEL);
	if (err < 0)
		goto free_iova;

	return virt;

free_iova:
	__free_iova(&tegra->carveout.domain, alloc);
free_pages:
	free_pages((unsigned long)virt, get_order(size));

	return ERR_PTR(err);
}
```