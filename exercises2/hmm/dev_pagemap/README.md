
#   struct dev_pagemap、struct vm_fault *vmf
```
int svm_migrate_init(struct amdgpu_device *adev)
{
        struct kfd_dev *kfddev = adev->kfd.dev;
        struct dev_pagemap *pgmap;
        struct resource *res = NULL;
        unsigned long size;
        void *r;

        /* Page migration works on Vega10 or newer */
        if (!KFD_IS_SOC15(kfddev))
                return -EINVAL;

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
                        devm_release_mem_region(adev->dev, res->start,
                                                res->end - res->start + 1);
                return PTR_ERR(r);
        }
        pr_debug("reserve %ldMB system memory for VRAM pages struct\n",
                 SVM_HMM_PAGE_STRUCT_SIZE(size) >> 20);

        amdgpu_amdkfd_reserve_system_mem(SVM_HMM_PAGE_STRUCT_SIZE(size));

        svm_range_set_max_pages(adev);

        pr_info("HMM registered %ldMB device memory\n", size >> 20);

        return 0;
}
int
svm_migrate_to_vram(struct svm_range *prange, uint32_t best_loc,
                    struct mm_struct *mm, uint32_t trigger)
{
        if  (!prange->actual_loc)
                return svm_migrate_ram_to_vram(prange, best_loc, mm, trigger);
        else
                return svm_migrate_vram_to_vram(prange, best_loc, mm, trigger);

}
```

+ devm_memremap_pages   


```
[ 2964.810650] PKRU: 55555554
[ 2964.810651] Call Trace:
[ 2964.810652]  <TASK>
[ 2964.810655]  add_pages+0x17/0x70
[ 2964.810659]  arch_add_memory+0x45/0x60
[ 2964.810661]  memremap_pages+0x2ff/0x6a0
[ 2964.810665]  devm_memremap_pages+0x23/0x70
[ 2964.810667]  pci_p2pdma_add_resource+0x19c/0x580
[ 2964.810671]  p2pmem_pci_probe+0x2c/0xb0 [p2pmem_pci]
[ 2964.810676]  local_pci_probe+0x48/0xb0
[ 2964.810679]  work_for_cpu_fn+0x17/0x30
[ 2964.810681]  process_one_work+0x21c/0x430
[ 2964.810683]  worker_thread+0x1fa/0x3c0
[ 2964.810684]  ? __pfx_worker_thread+0x10/0x10
[ 2964.810685]  kthread+0xee/0x120
[ 2964.810688]  ? __pfx_kthread+0x10/0x10
[ 2964.810690]  ret_from_fork+0x29/0x50
[ 2964.810694]  </TASK>
[ 2964.810695] ---[ end trace 0000000000000000 ]---
[ 2964.818531] p2pmem_pci 0000:00:04.0: unable to add p2p resource
```

## page map --> free

```
static const struct dev_pagemap_ops dmirror_devmem_ops = {
        .page_free      = dmirror_devmem_free,
        .migrate_to_ram = dmirror_devmem_fault,
};
```
```
[  217.845933]  dmirror_devmem_free+0x18/0x3e [mmu_test]
[  217.845937]  free_devmap_managed_page+0x59/0x60
[  217.845940]  put_devmap_managed_page+0x53/0x60
[  217.845943]  memunmap_pages+0x104/0x
```

```
[  196.593364]  dump_stack+0x7d/0x9c
[  196.593365]  dmirror_devmem_fault+0x2d/0x1a1 [mmu_test]
[  196.593367]  do_swap_page+0x569/0x730
[  196.593369]  __handle_mm_fault+0x882/0x8e0
[  196.593371]  handle_mm_fault+0xda/0x2b0
[  196.593372]  do_user_addr_fault+0x1bb/0x650
[  196.593373]  exc_page_fault+0x7d/0x170
[  196.593374]  ? asm_exc_page_fault+0x8/0x30
[  196.593375]  asm_exc_page_fault+0x1e/0x30
[  196.593377] RIP: 0033:0x7fefb63079ae
```


```
vm_fault_t do_swap_page(struct vm_fault *vmf)
{

        entry = pte_to_swp_entry(vmf->orig_pte);
        if (unlikely(non_swap_entry(entry))) {

                } else if (is_device_private_entry(entry)) {
                        vmf->page = pfn_swap_entry_to_page(entry);
                        ret = vmf->page->pgmap->ops->migrate_to_ram(vmf);
                }
```


#  pci_alloc_p2pmem

采用gen_pool_alloc_algo_owner进行分配   
genalloc 是 linux 内核提供的通用内存分配器，源码位于 lib/genalloc.c。这个分配器为独立于内核以外的内存块提供分配方法，采用的是最先适配原则，android 最新的 ION 内存管理器对 ION_HEAP_TYPE_CARVEOUT 类型的内存就是采用的这个分配器。    

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
#  svm_migrate_vma_to_vram    
```
static long
svm_migrate_vma_to_vram(struct kfd_node *node, struct svm_range *prange,
			struct vm_area_struct *vma, uint64_t start,
			uint64_t end, uint32_t trigger, uint64_t ttm_res_offset)
{
	struct kfd_process *p = container_of(prange->svms, struct kfd_process, svms);
	uint64_t npages = (end - start) >> PAGE_SHIFT;
	struct amdgpu_device *adev = node->adev;
	struct kfd_process_device *pdd;
	struct dma_fence *mfence = NULL;
	struct migrate_vma migrate = { 0 };
	unsigned long cpages = 0;
	unsigned long mpages = 0;
	dma_addr_t *scratch;
	void *buf;
	int r = -ENOMEM;

	memset(&migrate, 0, sizeof(migrate));
	migrate.vma = vma;
	migrate.start = start;
	migrate.end = end;
	migrate.flags = MIGRATE_VMA_SELECT_SYSTEM;
	migrate.pgmap_owner = SVM_ADEV_PGMAP_OWNER(adev);

	buf = kvcalloc(npages,
		       2 * sizeof(*migrate.src) + sizeof(uint64_t) + sizeof(dma_addr_t),
		       GFP_KERNEL);
	if (!buf)
		goto out;

	migrate.src = buf;
	migrate.dst = migrate.src + npages;
	scratch = (dma_addr_t *)(migrate.dst + npages);

	kfd_smi_event_migration_start(node, p->lead_thread->pid,
				      start >> PAGE_SHIFT, end >> PAGE_SHIFT,
				      0, node->id, prange->prefetch_loc,
				      prange->preferred_loc, trigger);

	r = migrate_vma_setup(&migrate);
	if (r) {
		dev_err(adev->dev, "%s: vma setup fail %d range [0x%lx 0x%lx]\n",
			__func__, r, prange->start, prange->last);
		goto out_free;
	}

	cpages = migrate.cpages;
	if (!cpages) {
		pr_debug("failed collect migrate sys pages [0x%lx 0x%lx]\n",
			 prange->start, prange->last);
		goto out_free;
	}
	if (cpages != npages)
		pr_debug("partial migration, 0x%lx/0x%llx pages collected\n",
			 cpages, npages);
	else
		pr_debug("0x%lx pages collected\n", cpages);

	r = svm_migrate_copy_to_vram(node, prange, &migrate, &mfence, scratch, ttm_res_offset);
	migrate_vma_pages(&migrate);

	svm_migrate_copy_done(adev, mfence);
	migrate_vma_finalize(&migrate);

	mpages = cpages - svm_migrate_unsuccessful_pages(&migrate);
	pr_debug("successful/cpages/npages 0x%lx/0x%lx/0x%lx\n",
			 mpages, cpages, migrate.npages);

	kfd_smi_event_migration_end(node, p->lead_thread->pid,
				    start >> PAGE_SHIFT, end >> PAGE_SHIFT,
				    0, node->id, trigger);

	svm_range_dma_unmap_dev(adev->dev, scratch, 0, npages);

out_free:
	kvfree(buf);
out:
	if (!r && mpages) {
		pdd = svm_range_get_pdd_by_node(prange, node);
		if (pdd)
			WRITE_ONCE(pdd->page_in, pdd->page_in + mpages);

		return mpages;
	}
	return r;
}

```

# KFD_IOCTL_SVM_OP_SET_ATTR