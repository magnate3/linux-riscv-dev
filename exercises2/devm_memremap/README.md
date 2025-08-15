#  ZONE_DEVICE
ZONE_DEVICE设施建立在SPARSEMEM_VMEMMAP之上，为设备驱动程序标识的物理地址范围提供struct page mem_map服务。ZONE_DEVICE的"device"方面与这样一个事实相关，即这些地址范围的页面对象从不被标记为在线，并且必须针对设备而不仅仅是页面进行引用，以保持内存固定以供活动使用。ZONE_DEVICE通过devm_memremap_pages()执行足够的内存热插拔，以打开pfn_to_page()、page_to_pfn()和get_user_pages()服务，以供给定范围的pfns使用。由于页面引用计数永远不会低于1，因此页面永远不会被跟踪为空闲内存，并且页面的struct list_head lru空间被重新用于反向引用到映射内存的主机设备/驱动程序。    

虽然SPARSEMEM将内存表示为一组节，可选地收集到内存块中，但ZONE_DEVICE用户需要更小的粒度来填充mem_map。鉴于ZONE_DEVICE内存从未标记为在线，因此它随后永远不会通过sysfs内存热插拔API在内存块边界上公开其内存范围。实现依赖于这种缺乏用户API约束，以允许指定子节大小的内存范围给arch_add_memory()，即内存热插拔的上半部分。子节支持允许2MB作为devm_memremap_pages()的跨架构通用对齐粒度。    

ZONE_DEVICE的用户有：   

pmem：将平台持久性内存映射为通过DAX映射进行直接I/O的目标。    

hmm：通过->page_fault()和->page_free()事件回调扩展ZONE_DEVICE，以允许设备驱动程序协调与设备内存相关的内存管理事件，通常是GPU内存。参见异构内存管理（HMM）。    

p2pdma：创建struct page对象，允许PCI/-E拓扑中的对等设备之间协调直接DMA操作，即绕过主机内存。    

# hmm   memremap_pages

```
struct resource *res;
struct dev_pagemap pagemap;

res = request_free_mem_region(&iomem_resource, /* number of bytes */,
                              "name of driver resource");
pagemap.type = MEMORY_DEVICE_PRIVATE;
pagemap.range.start = res->start;
pagemap.range.end = res->end;
pagemap.nr_range = 1;
pagemap.ops = &device_devmem_ops;
memremap_pages(&pagemap, numa_node_id());

memunmap_pages(&pagemap);
release_mem_region(pagemap.range.start, range_len(&pagemap.range));
```

## kgd2kfd_init_zone_device


```

static const struct dev_pagemap_ops svm_migrate_pgmap_ops = {
	.page_free		= svm_migrate_page_free,
	.migrate_to_ram		= svm_migrate_to_ram,
};
```

```
int kgd2kfd_init_zone_device(struct amdgpu_device *adev)
{
	struct amdgpu_kfd_dev *kfddev = &adev->kfd;
	struct dev_pagemap *pgmap;
	struct resource *res = NULL;
	unsigned long size;
	void *r;

	/* Page migration works on gfx9 or newer */
	if (adev->ip_versions[GC_HWIP][0] < IP_VERSION(9, 0, 1))
		return -EINVAL;

	if (adev->gmc.is_app_apu)
		return 0;

	pgmap = &kfddev->pgmap;
	memset(pgmap, 0, sizeof(*pgmap));

	/* TODO: register all vram to HMM for now.
	 * should remove reserved size
	 */
	size = ALIGN(adev->gmc.real_vram_size, 2ULL << 20);
	if (adev->gmc.xgmi.connected_to_cpu) {
		pgmap->range.start = adev->gmc.aper_base;
		pgmap->range.end = adev->gmc.aper_base + adev->gmc.aper_size - 1;
		pgmap->type = MEMORY_DEVICE_COHERENT;
	} else {
		res = devm_request_free_mem_region(adev->dev, &iomem_resource, size);
		if (IS_ERR(res))
			return -ENOMEM;
		pgmap->range.start = res->start;
		pgmap->range.end = res->end;
		pgmap->type = MEMORY_DEVICE_PRIVATE;
	}

	pgmap->nr_range = 1;
	pgmap->ops = &svm_migrate_pgmap_ops;
	pgmap->owner = SVM_ADEV_PGMAP_OWNER(adev);
	pgmap->flags = 0;
	/* Device manager releases device-specific resources, memory region and
	 * pgmap when driver disconnects from device.
	 */
	r = devm_memremap_pages(adev->dev, pgmap);
	if (IS_ERR(r)) {
		pr_err("failed to register HMM device memory\n");
		/* Disable SVM support capability */
		pgmap->type = 0;
		if (pgmap->type == MEMORY_DEVICE_PRIVATE)
			devm_release_mem_region(adev->dev, res->start, resource_size(res));
		return PTR_ERR(r);
	}

	pr_debug("reserve %ldMB system memory for VRAM pages struct\n",
		 SVM_HMM_PAGE_STRUCT_SIZE(size) >> 20);

	amdgpu_amdkfd_reserve_system_mem(SVM_HMM_PAGE_STRUCT_SIZE(size));

	pr_info("HMM registered %ldMB device memory\n", size >> 20);

	return 0;
}
```

+ devm_memremap_pages的返回地址没有被使用    



```
static void
svm_migrate_get_vram_page(struct svm_range *prange, unsigned long pfn)
{
	struct page *page;

	page = pfn_to_page(pfn);
	svm_range_bo_ref(prange->svm_bo);
	page->zone_device_data = prange->svm_bo;
	zone_device_page_init(page);
}
```

# devm_memremap_pages

```
[   26.107902]  vmemmap_populate+0x20/0x34
[   26.111726]  __populate_section_memmap+0x1a4/0x1d8
[   26.116506]  sparse_add_section+0x138/0x1f4
[   26.120678]  __add_pages+0xd8/0x180
[   26.124155]  pagemap_range+0x324/0x41c
[   26.127893]  memremap_pages+0x184/0x2b4
[   26.131717]  devm_memremap_pages+0x30/0x7c
[   26.135802]  svm_migrate_init+0xd8/0x18c [amdgpu]
[   26.140993]  kgd2kfd_device_init+0x39c/0x5e0 [amdgpu]
[   26.146525]  amdgpu_amdkfd_device_init+0x13c/0x1d4 [amdgpu]
[   26.152576]  amdgpu_device_ip_init+0x53c/0x588 [amdgpu]
[   26.158276]  amdgpu_device_init+0x828/0xc60 [amdgpu]
[   26.163714]  amdgpu_driver_load_kms+0x28/0x1a0 [amdgpu]
[   26.169412]  amdgpu_pci_probe+0x1b0/0x420 [amdgpu]
[   26.174675]  local_pci_probe+0x48/0xa0
[   26.178416]  work_for_cpu_fn+0x24/0x40
[   26.178418]  process_one_work+0x1ec/0x470
[   26.178420]  worker_thread+0x200/0x410
[   26.178422]  kthread+0xec/0x100
[   26.178424]  ret_from_fork+0x10/0x20
```


#  p2pmem devm_memremap_pages
采用gen_pool_alloc_algo_owner进行分配
genalloc 是 linux 内核提供的通用内存分配器，源码位于 lib/genalloc.c。这个分配器为独立于内核以外的内存块提供分配方法，采用的是最先适配原则，android 最新的 ION 内存管理器对 ION_HEAP_TYPE_CARVEOUT 类型的内存就是采用的这个分配器。

```
static const struct dev_pagemap_ops p2pdma_pgmap_ops = {
	.page_free = p2pdma_page_free,
};
```


```
	pgmap = &p2p_pgmap->pgmap;
	pgmap->range.start = pci_resource_start(pdev, bar) + offset;
	pgmap->range.end = pgmap->range.start + size - 1;
	pgmap->nr_range = 1;
	pgmap->type = MEMORY_DEVICE_PCI_P2PDMA;
	pgmap->ops = &p2pdma_pgmap_ops;
```

```
        addr = devm_memremap_pages(&pdev->dev, pgmap);
        if (IS_ERR(addr)) {
                error = PTR_ERR(addr);
                goto pgmap_free;
        }

        p2pdma = rcu_dereference_protected(pdev->p2pdma, 1);
        error = gen_pool_add_owner(p2pdma->pool, (unsigned long)addr,
                        pci_bus_address(pdev, bar) + offset,
                        range_len(&pgmap->range), dev_to_node(&pdev->dev),
                        &pgmap->ref);
```
 

nvmet_req_alloc_p2pmem_sgls --> pci_p2pmem_alloc_sgl -->  pci_alloc_p2pmem
```
static int nvmet_req_alloc_p2pmem_sgls(struct pci_dev *p2p_dev,
		struct nvmet_req *req)
{
	req->sg = pci_p2pmem_alloc_sgl(p2p_dev, &req->sg_cnt,
			nvmet_data_transfer_len(req));
	if (!req->sg)
		goto out_err;

	if (req->metadata_len) {
		req->metadata_sg = pci_p2pmem_alloc_sgl(p2p_dev,
				&req->metadata_sg_cnt, req->metadata_len);
		if (!req->metadata_sg)
			goto out_free_sg;
	}

	req->p2p_dev = p2p_dev;

	return 0;
out_free_sg:
	pci_p2pmem_free_sgl(req->p2p_dev, req->sg);
out_err:
	return -ENOMEM;
}

struct scatterlist *pci_p2pmem_alloc_sgl(struct pci_dev *pdev,
					 unsigned int *nents, u32 length)
{
	struct scatterlist *sg;
	void *addr;

	sg = kmalloc(sizeof(*sg), GFP_KERNEL);
	if (!sg)
		return NULL;

	sg_init_table(sg, 1);

	addr = pci_alloc_p2pmem(pdev, length);
	if (!addr)
		goto out_free_sg;

	sg_set_buf(sg, addr, length);
	*nents = 1;
	return sg;

out_free_sg:
	kfree(sg);
	return NULL;
}
void *pci_alloc_p2pmem(struct pci_dev *pdev, size_t size)
{
	void *ret = NULL;
	struct percpu_ref *ref;
	struct pci_p2pdma *p2pdma;

	/*
	 * Pairs with synchronize_rcu() in pci_p2pdma_release() to
	 * ensure pdev->p2pdma is non-NULL for the duration of the
	 * read-lock.
	 */
	rcu_read_lock();
	p2pdma = rcu_dereference(pdev->p2pdma);
	if (unlikely(!p2pdma))
		goto out;

	ret = (void *)gen_pool_alloc_owner(p2pdma->pool, size, (void **) &ref);
	if (!ret)
		goto out;

	if (unlikely(!percpu_ref_tryget_live(ref))) {
		gen_pool_free(p2pdma->pool, (unsigned long) ret, size);
		ret = NULL;
		goto out;
	}
out:
	rcu_read_unlock();
	return ret;
}
```