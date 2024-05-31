static void *pva_dma_alloc_and_map_at(struct device *dev, size_t size,
				      dma_addr_t iova, gfp_t flags,
				      unsigned long attrs)
{
	struct iommu_domain *domain;
	unsigned long shift, pg_size, mp_size;
	dma_addr_t tmp_iova, offset;
	phys_addr_t pa, pa_new;
	void *cpu_va;
	int ret;

	domain = iommu_get_domain_for_dev(dev);
	if (!domain) {
		dev_err(dev, "IOMMU domain not found");
		return NULL;
	}

	shift = __ffs(domain->pgsize_bitmap);
	pg_size = 1UL << shift;
	mp_size = pg_size;

	/* Reserve iova range */
	tmp_iova = iommu_dma_alloc_iova(dev, size, iova + size - pg_size);
	if (tmp_iova != iova) {
		dev_err(dev, "failed to reserve iova at 0x%llx size 0x%lx\n",
			iova, size);
		return NULL;
	}

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
	iommu_dma_free_iova(dev, tmp_iova, size);

	return cpu_va;

fail_map:
	iommu_unmap(domain, iova, offset);
	dma_free_attrs(dev, size, cpu_va, tmp_iova, attrs);
fail_dma_alloc:
	iommu_dma_free_iova(dev, iova, size);

	return NULL;
}