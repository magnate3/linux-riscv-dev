

# open-gpu-kernel-modules

[open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/3bf16b890caa8fd6b5db08b5c2437b51c758ac9d/kernel-open/nvidia-uvm/uvm.c#L47)


# uvm_get_cpu_addr

```
unsigned long int uvm_get_cpu_addr(struct vm_area_struct *vma, unsigned long start, unsigned long end)
{
	struct uvm_va_mappings_struct *map = uvm_get_cpu_mapping(get_shared_mem_va_space(), start, end); 

	if (!map)
		return 0;
//UCM_DBG("returning 0x%llx\n", map->cpu_vma->vm_start);
	return map->cpu_vma->vm_start;
}
static uvm_va_space_t *uvm_va_space = NULL;
uvm_va_space_t *get_shared_mem_va_space(void)
{
	return uvm_va_space;
}
```
##  va_space_create

```
NV_STATUS uvm_va_space_create(struct inode *inode, struct file *filp)
{
    NV_STATUS status;
    uvm_va_space_t *va_space;
    va_space = uvm_kvmalloc_zero(sizeof(*va_space));

    if (!va_space)
        return NV_ERR_NO_MEMORY;

    if (!uvm_va_space) {
	uvm_va_space = va_space;
	//UCM_DBG("Saving uvm_va_space %p that is handling shared memory\n", uvm_va_space);
    } 

    uvm_init_rwsem(&va_space->lock, UVM_LOCK_ORDER_VA_SPACE);
    uvm_mutex_init(&va_space->serialize_writers_lock, UVM_LOCK_ORDER_VA_SPACE_SERIALIZE_WRITERS);
    uvm_mutex_init(&va_space->read_acquire_write_release_lock, UVM_LOCK_ORDER_VA_SPACE_READ_ACQUIRE_WRITE_RELEASE_LOCK);
    uvm_range_tree_init(&va_space->va_range_tree);

	memset((void *)va_space->mmap_array, 0, 
			sizeof(struct uvm_va_mappings_struct) *	UVM_MAX_SUPPORTED_MMAPS);
	va_space->mmap_arr_idx = 0;
	va_space->skip_cache = false;

    // By default all struct files on the same inode share the same
    // address_space structure (the inode's) across all processes. This means
    // unmap_mapping_range would unmap virtual mappings across all processes on
    // that inode.
    //
    // Since the UVM driver uses the mapping offset as the VA of the file's
    // process, we need to isolate the mappings to each process.
    address_space_init_once(&va_space->mapping);
    va_space->mapping.host = inode;

    // Some paths in the kernel, for example force_page_cache_readahead which
    // can be invoked from user-space via madvise MADV_WILLNEED and fadvise
    // POSIX_FADV_WILLNEED, check the function pointers within
    // file->f_mapping->a_ops for validity. However, those paths assume that a_ops
    // itself is always valid. Handle that by using the inode's a_ops pointer,
    // which is what f_mapping->a_ops would point to anyway if we weren't re-
    // assigning f_mapping.
    va_space->mapping.a_ops = inode->i_mapping->a_ops;

#if defined(NV_ADDRESS_SPACE_HAS_BACKING_DEV_INFO)
    va_space->mapping.backing_dev_info = inode->i_mapping->backing_dev_info;
#endif

    // Init to 0 since we rely on atomic_inc_return behavior to return 1 as the first ID
    atomic64_set(&va_space->range_group_id_counter, 0);

    INIT_RADIX_TREE(&va_space->range_groups, NV_UVM_GFP_FLAGS);
    uvm_range_tree_init(&va_space->range_group_ranges);

    bitmap_zero(va_space->enabled_peers, UVM8_MAX_UNIQUE_GPU_PAIRS);

    // CPU is not explicitly registered in the va space
    uvm_processor_mask_set(&va_space->can_access[UVM_CPU_ID], UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->accessible_from[UVM_CPU_ID], UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->can_copy_from[UVM_CPU_ID], UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->has_native_atomics[UVM_CPU_ID], UVM_CPU_ID);
    // CPU always participates in system-wide atomics
    uvm_processor_mask_set(&va_space->system_wide_atomics_enabled_processors, UVM_CPU_ID);
    uvm_processor_mask_set(&va_space->faultable_processors, UVM_CPU_ID);


    #if defined(NV_PNV_NPU2_INIT_CONTEXT_PRESENT)
        if (uvm8_ats_mode) {
            // TODO: Bug 1896767: Be as retrictive as possible when using
            //       unsafe_mm. See the comments on unsafe_mm in
            //       uvm8_va_space.h.
            va_space->unsafe_mm = current->mm;
        }
    #endif


    filp->private_data = va_space;
    filp->f_mapping = &va_space->mapping;

    va_space->test_page_prefetch_enabled = true;

    init_tools_data(va_space);

    uvm_va_space_down_write(va_space);

    status = uvm_perf_init_va_space_events(va_space, &va_space->perf_events);
    if (status != NV_OK)
        goto fail;

    status = uvm_perf_heuristics_load(va_space);
    if (status != NV_OK)
        goto fail;

    status = uvm_gpu_init_va_space(va_space);
    if (status != NV_OK)
        goto fail;

    uvm_va_space_up_write(va_space);

    return NV_OK;

fail:
    uvm_perf_heuristics_unload(va_space);
    uvm_perf_destroy_va_space_events(&va_space->perf_events);
    uvm_va_space_up_write(va_space);

    uvm_kvfree(va_space);

    return status;
}
```

+  address_space_init_once   



#  uvm 设备   

> ## struct file_operations uvm_fops

```
static const struct file_operations uvm_fops =
{
    .open            = uvm_open_entry,
    .release         = uvm_release_entry,
    .mmap            = uvm_mmap_entry,
    .unlocked_ioctl  = uvm_unlocked_ioctl_entry,
#if NVCPU_IS_X86_64
    .compat_ioctl    = uvm_unlocked_ioctl_entry,
#endif
    .owner           = THIS_MODULE,
};
```

```
static inline void uvm_init_character_device(struct cdev *cdev, const struct file_operations *fops)
{
    cdev_init(cdev, fops);
    cdev->owner = THIS_MODULE;
}
```

>  ## uvm_open 字符设备   

```
static int uvm_open(struct inode *inode, struct file *filp)
{
    NV_STATUS status = uvm_global_get_status();

    if (status == NV_OK) {
        if (!uvm_down_read_trylock(&g_uvm_global.pm.lock))
            return -EAGAIN;

        status = uvm_va_space_create(inode, filp);

        uvm_up_read(&g_uvm_global.pm.lock);
    }

    return -nv_status_to_errno(status);
}
```


# references

[GPU的UM实现](https://zhuanlan.zhihu.com/p/679635240)   

[cuda统一内存优化DeepUM](https://zhuanlan.zhihu.com/p/672931240)  

[CUDA中的UM机制与GDR实现](https://www.ctyun.cn/developer/article/465119451353157)   