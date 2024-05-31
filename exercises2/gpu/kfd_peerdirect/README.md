
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