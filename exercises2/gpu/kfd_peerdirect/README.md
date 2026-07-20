
#  AMD的kfd_peerdirect


[看看源码](https://github.com/ROCm/ROCK-Kernel-Driver/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_peerdirect.c)  


#   svm_migrate_to_ram - CPU page fault handler



```
static const struct dev_pagemap_ops svm_migrate_pgmap_ops = {
        .page_free              = svm_migrate_page_free,
        .migrate_to_ram         = svm_migrate_to_ram,
};
``` 

# svm_migrate_to_vram - GPU page fault handler

```
static const struct amdgpu_irq_src_funcs gmc_v10_0_irq_funcs = {
        .set = gmc_v10_0_vm_fault_interrupt_state,
        .process = gmc_v10_0_process_interrupt,
};

```
gmc_v10_0_process_interrupt -->
amdgpu_vm_handle_fault -->  svm_range_restore_pages --> svm_migrate_to_vram    
    KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU
	
	
# pri fault


```
struct pri_queue {
        atomic_t inflight;
        bool finish;
        int status;
};
static struct notifier_block ppr_nb = {
        .notifier_call = ppr_notifier,
};
```


```
static int ppr_notifier(struct notifier_block *nb, unsigned long e, void *data)
{
        struct amd_iommu_fault *iommu_fault;
        struct pasid_state *pasid_state;
        struct device_state *dev_state;
        struct pci_dev *pdev = NULL;
        unsigned long flags;
        struct fault *fault;
        bool finish;
        u16 tag, devid, seg_id;
        int ret;

        iommu_fault = data;
        tag         = iommu_fault->tag & 0x1ff;
        finish      = (iommu_fault->tag >> 9) & 1;
		
		        fault = kzalloc(sizeof(*fault), GFP_ATOMIC);
        if (fault == NULL) {
                /* We are OOM - send success and let the device re-fault */
                finish_pri_tag(dev_state, pasid_state, tag);
                goto out_drop_state;
        }

        fault->dev_state = dev_state;
        fault->address   = iommu_fault->address;
        fault->state     = pasid_state;
        fault->tag       = tag;
        fault->finish    = finish;
        fault->pasid     = iommu_fault->pasid;
        fault->flags     = iommu_fault->flags;
        INIT_WORK(&fault->work, do_fault);
```
+ handle_mm_fault
```
static void do_fault(struct work_struct *work)
{
        struct fault *fault = container_of(work, struct fault, work);
        struct vm_area_struct *vma;
        vm_fault_t ret = VM_FAULT_ERROR;
        unsigned int flags = 0;
        struct mm_struct *mm;
        u64 address;

        mm = fault->state->mm;
        address = fault->address;

        if (fault->flags & PPR_FAULT_USER)
                flags |= FAULT_FLAG_USER;
        if (fault->flags & PPR_FAULT_WRITE)
                flags |= FAULT_FLAG_WRITE;
        flags |= FAULT_FLAG_REMOTE;

        mmap_read_lock(mm);
        vma = find_extend_vma(mm, address);
        if (!vma || address < vma->vm_start)
                /* failed to get a vma in the right range */
                goto out;

        /* Check if we have the right permissions on the vma */
        if (access_error(vma, fault))
                goto out;

        ret = handle_mm_fault(vma, address, flags, NULL);
out:
        mmap_read_unlock(mm);

        if (ret & VM_FAULT_ERROR)
                /* failed to service fault */
                handle_fault_error(fault);

        finish_pri_tag(fault->dev_state, fault->state, fault->tag);

        put_pasid_state(fault->state);

        kfree(fault);
}
```

#   page->zone_device_data  和  struct svm_range_bo

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

# iommu

```
 amd_iommu_bind_pasid(dev->adev->pdev, p->pasid, p->lead_thread)
amd_iommu_set_invalidate_ctx_cb(kfd->adev->pdev, NULL);
amd_iommu_set_invalid_ppr_cb(kfd->adev->pdev, NULL);
```

+   amd_iommu_bind_pasid
```
        mm                        = get_task_mm(task);
        pasid_state->mm           = mm;
        pasid_state->device_state = dev_state;
        pasid_state->pasid        = pasid;
        pasid_state->invalid      = true; /* Mark as valid only if we are
                                             done with setting up the pasid */
        pasid_state->mn.ops       = &iommu_mn;

        if (pasid_state->mm == NULL)
                goto out_free;

        ret = mmu_notifier_register(&pasid_state->mn, mm);
		
```


```
static const struct mmu_notifier_ops iommu_mn = {
        .release                = mn_release,
        .invalidate_range       = mn_invalidate_range,
};
```

```
static void mn_invalidate_range(struct mmu_notifier *mn,
                                struct mm_struct *mm,
                                unsigned long start, unsigned long end)
{
        struct pasid_state *pasid_state;
        struct device_state *dev_state;

        pasid_state = mn_to_state(mn);
        dev_state   = pasid_state->device_state;

        if ((start ^ (end - 1)) < PAGE_SIZE)
                amd_iommu_flush_page(dev_state->domain, pasid_state->pasid,
                                     start);
        else
                amd_iommu_flush_tlb(dev_state->domain, pasid_state->pasid);
}
```

# sdma
This is a multi-purpose DMA engine. The kernel driver uses it for various things including paging and GPU page table updates. It’s also exposed to userspace for use by user mode drivers (OpenGL, Vulkan, etc.)

amdgpu_ttm_access_memory_sdma   
```
        amdgpu_res_first(abo->tbo.resource, offset, len, &src_mm);
        src_addr = amdgpu_ttm_domain_start(adev, bo->resource->mem_type) +
                src_mm.start;
        dst_addr = amdgpu_bo_gpu_offset(adev->mman.sdma_access_bo);
        if (write)
                swap(src_addr, dst_addr);

        amdgpu_emit_copy_buffer(adev, &job->ibs[0], src_addr, dst_addr,
                                PAGE_SIZE, false);

        amdgpu_ring_pad_ib(adev->mman.buffer_funcs_ring, &job->ibs[0]);
        WARN_ON(job->ibs[0].length_dw > num_dw);

        fence = amdgpu_job_submit(job);
```
# references

[AMD GPU ](https://xxlnx.github.io/page/2/)   
[CPU视角下的GPU物理内存 (VRAM/GTT)](https://xxlnx.github.io/2020/05/25/amdgpu/GPU_Physical_Memory/)   
[AMD GPU 虚拟内存](https://xxlnx.github.io/2020/07/05/amdgpu/AMD_GPU_Virtual_Memory/)   