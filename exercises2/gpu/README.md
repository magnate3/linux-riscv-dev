

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

```
struct vm_ucm_operations_struct uvm_ucm_ops_managed =
{
	.get_mmaped_file = uvm_vm_get_mmaped_file,
	.get_cpu_addr = uvm_get_cpu_addr,
	.invalidate_cached_page = uvm_vm_invalidate_cached_page,
	.retrive_cached_page = uvm_vm_retrive_cached_page,
	.retrive_16cached_pages = uvm_vm_retrive_16cached_pages_multi, //uvm_vm_retrive_16cached_pages2,
	.is_gpu_page_dirty = uvm_vm_is_gpu_page_dirty,
};
```

```
   vma->vm_ops = &uvm_vm_ops_managed;
   vma->ucm_vm_ops = &uvm_ucm_ops_managed;
```

>  ## page分配   

```
static NV_STATUS phys_mem_allocate(uvm_page_tree_t *tree, NvLength size, uvm_aperture_t location, uvm_pmm_alloc_flags_t pmm_flags, uvm_mmu_page_table_alloc_t *out)
{
    memset(out, 0, sizeof(*out));

    if (location == UVM_APERTURE_SYS)
        return phys_mem_allocate_sysmem(tree, size, out);
    else
        return phys_mem_allocate_vidmem(tree, size, pmm_flags, out);
}
```

> ## struct vm_ucm_operations_struct 

[参考](https://github.com/acsl-technion/gaia_linux/blob/74ead19e60f17dc5f8a93134b23568badf368daa/include/linux/mm.h#L248)   

```
struct vm_ucm_operations_struct {
	/* vma - current vma, start - virtual address the mapping starts at (==shared_addr)*/
	struct file *(*get_mmaped_file)(struct vm_area_struct *vma, unsigned long start, unsigned long end);
	unsigned long int (*get_cpu_addr)(struct vm_area_struct *vma, unsigned long start, unsigned long end);
	int (*invalidate_cached_page)(struct vm_area_struct *vma, unsigned long virt_addr);
	int (*retrive_cached_page)(unsigned long virt_addr, struct page *cpu_page);
	int (*retrive_16cached_pages)(unsigned long virt_addr, struct page *pages_arr[]);
	/* Recived the struct page of the gpu_page that was foundin page cahe and returns true if this
	 * page is dirty on the GPU. Note that this means that 16 cpu pages corresponding to this gpu_page
	 * have to be invalidated on CPU*/
	int (*is_gpu_page_dirty)(struct vm_area_struct *vma, struct page *gpu_page);
};
```
+ 1 get_page_from_gpu   
```
struct page *get_page_from_gpu(struct address_space *mapping, pgoff_t offset,
		struct page *page)
{
	struct ucm_page_data *pdata;
	struct page *gpu_page;
	bool allocated = false;
	int err;
	//UCM_ERR("Found page @offst %lld in cache but not latest version i_ino=%ld. page=0x%llx\n",offset, mapping->host->i_ino, page);

	/* If I got here then latest version is on GPU and I need a CPU page.
	 * This is because if gpu version was needed and latest is on cpu
	 * I would have taken careof this in aquire.
	 * so - need to bring it from gpu to cpu */
	if (!page) {
		UCM_ERR("Can't call this func without a backing up cpu page\n");
		return NULL;
	}
	gpu_page = pagecache_get_gpu_page(mapping, offset, GPU_NVIDIA, true);
	if (!gpu_page) {
		return NULL;
	}

	pdata = (struct ucm_page_data *)gpu_page->private;
	if (!pdata) {
		UCM_ERR("pdata = NULL\n");
		return NULL;
	}
	if (!pdata->gpu_maped_vma) {
		UCM_ERR("!pdata->gpu_maped_vma\n");
		return NULL;
	} else if(!pdata->gpu_maped_vma->ucm_vm_ops) {
		UCM_ERR("!pdata->gpu_maped_vma->ucm_vm_ops\n");
		return NULL;
	}
	if (!pdata->gpu_maped_vma->ucm_vm_ops->retrive_cached_page) {
		UCM_ERR("vma->ucm_vm_ops->retrive_cached_page cb is not provided. cant update page\n");
		return NULL;
	}

	if (pdata->gpu_maped_vma->ucm_vm_ops->retrive_cached_page(pdata->shared_addr + (offset % 16)*PAGE_SIZE, page)) {
		UCM_ERR("FAIL ++++ retrive_cached_page for offset = %lld FAILED\n", offset);
		if (allocated) {
			page_cache_release(page);
			page = NULL;
			return NULL;
		}
	}
	(void)set_page_version_as_on(mapping, offset,SYS_CPU, GPU_NVIDIA);
	PageUptodate(page);
	return page;
}
struct page *pagecache_get_gpu_page(struct address_space *mapping, pgoff_t offset,
	int gpu_id, bool ignore_version)
{
	void **pagep;
	struct page *page;
	struct radix_tree_node *node = NULL;

	rcu_read_lock();
repeat:
	page = NULL;
	pagep = radix_tree_lookup_slot_node(&mapping->page_tree, offset, &node);
	if (node) {
		int gpu_page_idx = ucm_get_gpu_idx(offset % RADIX_TREE_MAP_SIZE);

		if (mapping->page_tree.rnode == node)
			UCM_ERR("got root: num_gpu = %ld node->count = %ld\n", node->count_gpu, node->count);
		if (node->gpu_pages[gpu_id][gpu_page_idx]) {
			page = node->gpu_pages[gpu_id][gpu_page_idx];
		}
	}

out:
	rcu_read_unlock();

	if (radix_tree_exceptional_entry(page))
		page = NULL;
	if (!page)
		return NULL;

	if (!PageonGPU(page)) {
		UCM_ERR("Got some page but its not marked on gpu\n");
		return NULL;
	}

	return page;
}
EXPORT_SYMBOL(pagecache_get_gpu_page);

```
+ pagecache_get_gpu_page   
+ uvm_vm_retrive_cached_page  


```
#define AQUIRE_PAGES_THREASHOLD 1024
SYSCALL_DEFINE3(maquire, unsigned long, start, size_t, len, int, flags)
{
	unsigned long end_byte;
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *shared_vma, *cpu_vma;
	int unmapped_error = 0;
	int error = -EINVAL;
	struct file *mfile = NULL;
	int nr_pages;
 	pgoff_t index, end; 
	unsigned i;
	int gpu_page_idx = -1;
	int gpu_missing = 0;
	int *pages_idx;
	unsigned long tagged, tagged_gpu;

	struct page **cached_pages = NULL;

	int stat_gpu_deleted = -1;
	int stat_cpu_marked = 0;

	struct timespec tstart, tend;
	int ret;


	if (flags & ~(MA_PROC_NVIDIA | MA_PROC_AMD)) {
		UCM_ERR("What GPU should I aquire for??? Exit\n");
		goto out;
	}
	if (offset_in_page(start))
			goto out;
	if ((flags & MS_ASYNC) && (flags & MS_SYNC))
			goto out;

	error = -ENOMEM;
	if ( start + len <= start)
			goto out;
	error = 0;

	/*
	 * If the interval [start,end) covers some unmapped address ranges,
	 * just ignore them, but return -ENOMEM at the end.
	 */
	down_read(&mm->mmap_sem);
	shared_vma = find_vma(mm, start);
	if (!shared_vma) {
		UCM_ERR("no shared_vma found starting at 0x%llx\n", start);
		goto out_unlock;
	}
	if (!shared_vma->gpu_mapped_shared) {
		UCM_ERR("Aquire is not supported for a NOT-GPU maped vma\n");
		error = -EINVAL;
		goto out_unlock;
	}
	if (!shared_vma->ucm_vm_ops || !shared_vma->ucm_vm_ops->get_mmaped_file || !shared_vma->ucm_vm_ops->invalidate_cached_page) {
		UCM_ERR("ucm_vm_ops not povided\n");
		error = -EINVAL;
		goto out_unlock;
	}


	unsigned long int cpu_addr = shared_vma->ucm_vm_ops->get_cpu_addr(shared_vma, start, end);
	if (!cpu_addr) {
		UCM_ERR("Didn't find CPU addr?!?!?\n");
		error = -EINVAL;
                goto out_unlock;
	}	
	cpu_vma = find_vma(mm, cpu_addr);
	if (!cpu_vma) {
		UCM_ERR("no cpu_vma found starting at 0x%llx\n", cpu_addr);
		goto out_unlock;
	}

	index = (cpu_addr - cpu_vma->vm_start) / PAGE_SIZE;
	end = (cpu_addr + len - cpu_vma->vm_start) / PAGE_SIZE + 1;


	 //Do the msync here
	 get_file(cpu_vma->vm_file);
	up_read(&mm->mmap_sem);

	if (vfs_fsync_range(cpu_vma->vm_file, index, end, 0))
		UCM_ERR("vfs sync failed\n");
	fput(cpu_vma->vm_file);
	down_read(&mm->mmap_sem);

	 while ((index <= end) ) {
			/*(nr_pages = pagevec_lookup_tag(&pvec, cpu_vma->vm_file->f_mapping, &index,
				PAGECACHE_TAG_DIRTY,
				min(end - index, (pgoff_t)PAGEVEC_SIZE-1) + 1)) != 0)*/
		int num_pages = min(AQUIRE_PAGES_THREASHOLD +1, end-index);
		pgoff_t start = index;
		pages_idx = (int*)kzalloc(sizeof(int)*(num_pages + 1), GFP_KERNEL);
		if (!pages_idx) {
			UCM_ERR("Error allocating memory!\n");
			error = -ENOMEM;
         	       goto out_unlock;
		}
		nr_pages = find_get_taged_pages_idx(cpu_vma->vm_file->f_mapping, &index,
			num_pages, pages_idx, PAGECACHE_TAG_CPU_DIRTY);
		index += nr_pages;
		stat_cpu_marked += nr_pages;
		if (!nr_pages) {
			//UCM_DBG("No pages taged as DIRTY ON CPU!!! index=%d end - %d\n", index, end);
			kfree(pages_idx);
			break;
		} 
		for (i = 0; i < nr_pages; i++) {
			int page_idx = pages_idx[i];
			/* until radix tree lookup accepts end_index */
			if (page_idx > end) {
				UCM_DBG("page_idx (%d) > end (%d). continue..\n", page_idx, end);
				continue;
			}
			if (1/*(gpu_page_idx == -1) || (gpu_page_idx != page_idx % 16)*/) {
				struct page *gpu_page = pagecache_get_gpu_page(cpu_vma->vm_file->f_mapping, page_idx, GPU_NVIDIA, true);

				if (gpu_page) {
					struct mem_cgroup *memcg;
					unsigned long flags;
					struct ucm_page_data *pdata = (struct ucm_page_data *)gpu_page->private;

					memcg = mem_cgroup_begin_page_stat(gpu_page);
					gpu_page_idx = gpu_page->index;
					//Remove the page from page cache
					__set_page_locked(gpu_page);
					spin_lock_irqsave(&cpu_vma->vm_file->f_mapping->tree_lock, flags);
					__delete_from_page_cache_gpu(gpu_page, NULL, memcg, GPU_NVIDIA);
					ClearPageonGPU(gpu_page);
					spin_unlock_irqrestore(&cpu_vma->vm_file->f_mapping->tree_lock, flags);
					mem_cgroup_end_page_stat(memcg);
					__clear_page_locked(gpu_page);
					//The virtual address of the page in the shared VM is saved ar page->private
					if (stat_gpu_deleted < 0)
						stat_gpu_deleted = 0;
					stat_gpu_deleted++;
					if (shared_vma->ucm_vm_ops->invalidate_cached_page(shared_vma, pdata->shared_addr ))
						UCM_ERR("Error invalidating page at virt addr 0x%llx on GPU!!!\n", pdata->shared_addr );
				} else
					gpu_missing++;
				radix_tree_tag_clear(&cpu_vma->vm_file->f_mapping->page_tree, page_idx,
                                   		PAGECACHE_TAG_CPU_DIRTY);
			}
		}
		kfree(pages_idx);
	}

out_unlock:
        up_read(&mm->mmap_sem);
out:
	UCM_DBG("done: stat_gpu_deleted=%d, stat_cpu_marked=%d gpu_missing=%d\n", stat_gpu_deleted, stat_cpu_marked, gpu_missing);
        return stat_gpu_deleted;
}
```
+ pagecache_get_gpu_page   

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

# page 迁移

```
static NV_STATUS nv_migrate_vma(struct migrate_vma *args, migrate_vma_state_t *state)
{
    int ret;

#if defined(CONFIG_MIGRATE_VMA_HELPER)
    static const struct migrate_vma_ops uvm_migrate_vma_ops =
    {
        .alloc_and_copy = uvm_migrate_vma_alloc_and_copy_helper,
        .finalize_and_map = uvm_migrate_vma_finalize_and_map_helper,
    };

    ret = migrate_vma(&uvm_migrate_vma_ops, args->vma, args->start, args->end, args->src, args->dst, state);
    if (ret < 0)
        return errno_to_nv_status(ret);
#else // CONFIG_MIGRATE_VMA_HELPER

#if defined(NV_MIGRATE_VMA_FLAGS_PRESENT)
    args->flags = MIGRATE_VMA_SELECT_SYSTEM;
#endif // NV_MIGRATE_VMA_FLAGS_PRESENT

    ret = migrate_vma_setup(args);
    if (ret < 0)
        return errno_to_nv_status(ret);

    uvm_migrate_vma_alloc_and_copy(args, state);
    if (state->status == NV_OK) {
        migrate_vma_pages(args);
        uvm_migrate_vma_finalize_and_map(args, state);
    }

    migrate_vma_finalize(args);
#endif // CONFIG_MIGRATE_VMA_HELPER

    return state->status;
}
```
#  dmirror_device_evict_chunk
```
static void dmirror_device_evict_chunk(struct dmirror_chunk *chunk)
{
	unsigned long start_pfn = chunk->pagemap.range.start >> PAGE_SHIFT;
	unsigned long end_pfn = chunk->pagemap.range.end >> PAGE_SHIFT;
	unsigned long npages = end_pfn - start_pfn + 1;
	unsigned long i;
	unsigned long *src_pfns;
	unsigned long *dst_pfns;

	src_pfns = kcalloc(npages, sizeof(*src_pfns), GFP_KERNEL);
	dst_pfns = kcalloc(npages, sizeof(*dst_pfns), GFP_KERNEL);

	migrate_device_range(src_pfns, start_pfn, npages);
	for (i = 0; i < npages; i++) {
		struct page *dpage, *spage;

		spage = migrate_pfn_to_page(src_pfns[i]);
		if (!spage || !(src_pfns[i] & MIGRATE_PFN_MIGRATE))
			continue;

		if (WARN_ON(!is_device_private_page(spage) &&
			    !is_device_coherent_page(spage)))
			continue;
		spage = BACKING_PAGE(spage);
		dpage = alloc_page(GFP_HIGHUSER_MOVABLE | __GFP_NOFAIL);
		lock_page(dpage);
		copy_highpage(dpage, spage);
		dst_pfns[i] = migrate_pfn(page_to_pfn(dpage));
		if (src_pfns[i] & MIGRATE_PFN_WRITE)
			dst_pfns[i] |= MIGRATE_PFN_WRITE;
	}
	migrate_device_pages(src_pfns, dst_pfns, npages);
	migrate_device_finalize(src_pfns, dst_pfns, npages);
	kfree(src_pfns);
	kfree(dst_pfns);
}
```

# references

[GPU的UM实现](https://zhuanlan.zhihu.com/p/679635240)   

[cuda统一内存优化DeepUM](https://zhuanlan.zhihu.com/p/672931240)  

[CUDA中的UM机制与GDR实现](https://www.ctyun.cn/developer/article/465119451353157)   

[address_space_init_once of gdr](https://github.com/NVIDIA/gdrcopy/blob/fbb6f924e0b6361c382bcb0aaef595f08a2cb61f/src/gdrdrv/gdrdrv.c#L72)   