
#  test-unmap-area1.c 


```
 address = mmap (NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
```
触发内核调用 sgx_get_unmapped_area

```
[16424.302141] [<ffff00000123059c>] sgx_get_unmapped_area+0x2c/0x60 [mmap_example]
[16424.309417] [<ffff000008254a2c>] get_unmapped_area.part.37+0x60/0xd0
[16424.315741] [<ffff000008257f74>] do_mmap+0x12c/0x34c
[16424.320684] [<ffff00000823467c>] vm_mmap_pgoff+0xf0/0x124
[16424.326059] [<ffff000008255a7c>] SyS_mmap_pgoff+0xc0/0x23c
[16424.331519] [<ffff000008089548>] sys_mmap+0x54/0x68
```

+ kernel


```
struct vm_operations_struct mmap_vm_ops = {
	.open = mmap_open,
	.close = mmap_close,
	.fault = mmap_fault,
#if 0
        .page_mkwrite = op_page_mkwrite,
#endif
};
static const struct file_operations mmap_fops = {
	.owner = THIS_MODULE,
	.open = mmapfop_open,
	.release = mmapfop_close,
	.mmap = op_mmap,
    .unlocked_ioctl = device_ioctl,
    .get_unmapped_area = sgx_get_unmapped_area,
        
};
```

#  get_unmapped_area


```
get_unmapped_area(struct file *file, unsigned long addr, unsigned long len,
                unsigned long pgoff, unsigned long flags)
{
        unsigned long (*get_area)(struct file *, unsigned long,
                                  unsigned long, unsigned long, unsigned long);

        unsigned long error = arch_mmap_check(addr, len, flags);
        if (error)
                return error;

        /* Careful about overflows.. */
        if (len > TASK_SIZE)
                return -ENOMEM;

        get_area = current->mm->get_unmapped_area;
        if (file) {
                if (file->f_op->get_unmapped_area)
                        get_area = file->f_op->get_unmapped_area;
        } else if (flags & MAP_SHARED) {
                /*
                 * mmap_region() will call shmem_zero_setup() to create a file,
                 * so use shmem's get_unmapped_area in case it can be huge.
                 * do_mmap_pgoff() will clear pgoff, so match alignment.
                 */
                pgoff = 0;
                get_area = shmem_get_unmapped_area;
        }

        addr = get_area(file, addr, len, pgoff, flags);
        if (IS_ERR_VALUE(addr))
                return addr;

        if (addr > TASK_SIZE - len)
                return -ENOMEM;
        if (offset_in_page(addr))
                return -EINVAL;

        error = security_mmap_addr(addr);
        return error ? error : addr;
}

EXPORT_SYMBOL(get_unmapped_area);
```


# kgsl

[参考源码](https://github.com/F2LX/kernel_xiaomi_sdm660/blob/3d0fc3246b67bff47f3452a20097d1d937c1d4b3/drivers/gpu/msm/kgsl.c#L4294)     
```
static unsigned long
kgsl_get_unmapped_area(struct file *file, unsigned long addr,
			unsigned long len, unsigned long pgoff,
			unsigned long flags)
{
	unsigned long val;
	unsigned long vma_offset = pgoff << PAGE_SHIFT;
	struct kgsl_device_private *dev_priv = file->private_data;
	struct kgsl_process_private *private = dev_priv->process_priv;
	struct kgsl_device *device = dev_priv->device;
	struct kgsl_mem_entry *entry = NULL;

	if (vma_offset == (unsigned long) device->memstore.gpuaddr)
		return get_unmapped_area(NULL, addr, len, pgoff, flags);

	val = get_mmap_entry(private, &entry, pgoff, len);
	if (val)
		return val;

	/* Do not allow CPU mappings for secure buffers */
	if (kgsl_memdesc_is_secured(&entry->memdesc)) {
		val = -EPERM;
		goto put;
	}

	if (!kgsl_memdesc_use_cpu_map(&entry->memdesc)) {
		val = get_unmapped_area(NULL, addr, len, 0, flags);
		if (IS_ERR_VALUE(val))
			KGSL_DRV_ERR_RATELIMIT(device,
				"get_unmapped_area: pid %d addr %lx pgoff %lx len %ld failed error %d\n",
				private->pid, addr, pgoff, len, (int) val);
	} else {
		 val = _get_svm_area(private, entry, addr, len, flags);
		 if (IS_ERR_VALUE(val))
			KGSL_DRV_ERR_RATELIMIT(device,
				"_get_svm_area: pid %d mmap_base %lx addr %lx pgoff %lx len %ld failed error %d\n",
				private->pid, current->mm->mmap_base, addr,
				pgoff, len, (int) val);
	}

put:
	kgsl_mem_entry_put(entry);
	return val;
}
```

```
static unsigned long _get_svm_area(struct kgsl_process_private *private,
		struct kgsl_mem_entry *entry, unsigned long hint,
		unsigned long len, unsigned long flags)
{
	uint64_t start, end;
	int align_shift = kgsl_memdesc_get_align(&entry->memdesc);
	uint64_t align;
	unsigned long result;
	unsigned long addr;

	if (align_shift >= ilog2(SZ_2M))
		align = SZ_2M;
	else if (align_shift >= ilog2(SZ_1M))
		align = SZ_1M;
	else if (align_shift >= ilog2(SZ_64K))
		align = SZ_64K;
	else
		align = SZ_4K;

	/* get the GPU pagetable's SVM range */
	if (kgsl_mmu_svm_range(private->pagetable, &start, &end,
				entry->memdesc.flags))
		return -ERANGE;

	/* now clamp the range based on the CPU's requirements */
	start = max_t(uint64_t, start, mmap_min_addr);
	end = min_t(uint64_t, end, current->mm->mmap_base);
	if (start >= end)
		return -ERANGE;

	if (flags & MAP_FIXED) {
		/* we must use addr 'hint' or fail */
		return _gpu_set_svm_region(private, entry, hint, len);
	} else if (hint != 0) {
		struct vm_area_struct *vma;

		/*
		 * See if the hint is usable, if not we will use
		 * it as the start point for searching.
		 */
		addr = clamp_t(unsigned long, hint & ~(align - 1),
				start, (end - len) & ~(align - 1));

		vma = find_vma(current->mm, addr);

		if (vma == NULL || ((addr + len) <= vma->vm_start)) {
			result = _gpu_set_svm_region(private, entry, addr, len);

			/* On failure drop down to keep searching */
			if (!IS_ERR_VALUE(result))
				return result;
		}
	} else {
		/* no hint, start search at the top and work down */
		addr = end & ~(align - 1);
	}

	/*
	 * Search downwards from the hint first. If that fails we
	 * must try to search above it.
	 */
	result = _search_range(private, entry, start, addr, len, align);
	if (IS_ERR_VALUE(result) && hint != 0)
		result = _search_range(private, entry, addr, end, len, align);

	return result;
}

```

```
static int kgsl_mmap(struct file *file, struct vm_area_struct *vma)
{
	unsigned int ret, cache;
	unsigned long vma_offset = vma->vm_pgoff << PAGE_SHIFT;
	struct kgsl_device_private *dev_priv = file->private_data;
	struct kgsl_process_private *private = dev_priv->process_priv;
	struct kgsl_mem_entry *entry = NULL;
	struct kgsl_device *device = dev_priv->device;

	/* Handle leagacy behavior for memstore */

	if (vma_offset == (unsigned long) device->memstore.gpuaddr)
		return kgsl_mmap_memstore(device, vma);

	/*
	 * The reference count on the entry that we get from
	 * get_mmap_entry() will be held until kgsl_gpumem_vm_close().
	 */
	ret = get_mmap_entry(private, &entry, vma->vm_pgoff,
				vma->vm_end - vma->vm_start);
	if (ret)
		return ret;

	vma->vm_flags |= entry->memdesc.ops->vmflags;

	vma->vm_private_data = entry;

	/* Determine user-side caching policy */

	cache = kgsl_memdesc_get_cachemode(&entry->memdesc);

	switch (cache) {
	case KGSL_CACHEMODE_UNCACHED:
		vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
		break;
	case KGSL_CACHEMODE_WRITETHROUGH:
		vma->vm_page_prot = pgprot_writethroughcache(vma->vm_page_prot);
		if (vma->vm_page_prot ==
			pgprot_writebackcache(vma->vm_page_prot))
			WARN_ONCE(1, "WRITETHROUGH is deprecated for arm64");
		break;
	case KGSL_CACHEMODE_WRITEBACK:
		vma->vm_page_prot = pgprot_writebackcache(vma->vm_page_prot);
		break;
	case KGSL_CACHEMODE_WRITECOMBINE:
	default:
		vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
		break;
	}

	vma->vm_ops = &kgsl_gpumem_vm_ops;

	if (cache == KGSL_CACHEMODE_WRITEBACK
		|| cache == KGSL_CACHEMODE_WRITETHROUGH) {
		int i;
		unsigned long addr = vma->vm_start;
		struct kgsl_memdesc *m = &entry->memdesc;

		for (i = 0; i < m->page_count; i++) {
			struct page *page = m->pages[i];

			vm_insert_page(vma, addr, page);
			addr += PAGE_SIZE;
		}
	}

	vma->vm_file = file;

	entry->memdesc.useraddr = vma->vm_start;

	trace_kgsl_mem_mmap(entry);
	return 0;
}

```
+ vma->vm_ops    
+ vm_insert_page   
+  vma->vm_file = file;   

```
	vma->vm_ops = &kgsl_gpumem_vm_ops;
	vm_insert_page(vma, addr, page);
		 
```
> ## memdesc->pages  分配
```
int
kgsl_sharedmem_page_alloc_user(struct kgsl_memdesc *memdesc,
			uint64_t size)
{
	int ret = 0;
	unsigned int j, page_size, len_alloc;
	unsigned int pcount = 0;
	size_t len;
	unsigned int align;

	static DEFINE_RATELIMIT_STATE(_rs,
					DEFAULT_RATELIMIT_INTERVAL,
					DEFAULT_RATELIMIT_BURST);

	size = PAGE_ALIGN(size);
	if (size == 0 || size > UINT_MAX)
		return -EINVAL;

	align = (memdesc->flags & KGSL_MEMALIGN_MASK) >> KGSL_MEMALIGN_SHIFT;

	page_size = kgsl_get_page_size(size, align);

	/*
	 * The alignment cannot be less than the intended page size - it can be
	 * larger however to accomodate hardware quirks
	 */

	if (align < ilog2(page_size)) {
		kgsl_memdesc_set_align(memdesc, ilog2(page_size));
		align = ilog2(page_size);
	}

	/*
	 * There needs to be enough room in the page array to be able to
	 * service the allocation entirely with PAGE_SIZE sized chunks
	 */

	len_alloc = PAGE_ALIGN(size) >> PAGE_SHIFT;

	memdesc->ops = &kgsl_page_alloc_ops;

	/*
	 * Allocate space to store the list of pages. This is an array of
	 * pointers so we can track 1024 pages per page of allocation.
	 * Keep this array around for non global non secure buffers that
	 * are allocated by kgsl. This helps with improving the vm fault
	 * routine by finding the faulted page in constant time.
	 */

	memdesc->pages = kgsl_malloc(len_alloc * sizeof(struct page *));
```

```
static void *io_uring_validate_mmap_request(struct file *file,
					    loff_t pgoff, size_t sz)
{
	struct io_ring_ctx *ctx = file->private_data;
	loff_t offset = pgoff << PAGE_SHIFT;
	struct page *page;
	void *ptr;

	switch (offset) {
	case IORING_OFF_SQ_RING:
	case IORING_OFF_CQ_RING:
		ptr = ctx->rings;
		break;
	case IORING_OFF_SQES:
		ptr = ctx->sq_sqes;
		break;
	default:
		return ERR_PTR(-EINVAL);
	}

	page = virt_to_head_page(ptr);
	if (sz > page_size(page))
		return ERR_PTR(-EINVAL);

	return ptr;
}

```

#  kgsl_gpumem_vm_fault


```
static int
kgsl_gpumem_vm_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
{
	struct kgsl_mem_entry *entry = vma->vm_private_data;

	if (!entry)
		return VM_FAULT_SIGBUS;
	if (!entry->memdesc.ops || !entry->memdesc.ops->vmfault)
		return VM_FAULT_SIGBUS;

	return entry->memdesc.ops->vmfault(&entry->memdesc, vma, vmf);
}

```

#  io_uring_nommu_get_unmapped_area
```
#ifdef CONFIG_MMU

static int io_uring_mmap(struct file *file, struct vm_area_struct *vma)
{
	size_t sz = vma->vm_end - vma->vm_start;
	unsigned long pfn;
	void *ptr;

	ptr = io_uring_validate_mmap_request(file, vma->vm_pgoff, sz);
	if (IS_ERR(ptr))
		return PTR_ERR(ptr);

	pfn = virt_to_phys(ptr) >> PAGE_SHIFT;
	return remap_pfn_range(vma, vma->vm_start, pfn, sz, vma->vm_page_prot);
}

#else /* !CONFIG_MMU */

static int io_uring_mmap(struct file *file, struct vm_area_struct *vma)
{
	return vma->vm_flags & (VM_SHARED | VM_MAYSHARE) ? 0 : -EINVAL;
}

static unsigned int io_uring_nommu_mmap_capabilities(struct file *file)
{
	return NOMMU_MAP_DIRECT | NOMMU_MAP_READ | NOMMU_MAP_WRITE;
}

static unsigned long io_uring_nommu_get_unmapped_area(struct file *file,
	unsigned long addr, unsigned long len,
	unsigned long pgoff, unsigned long flags)
{
	void *ptr;

	ptr = io_uring_validate_mmap_request(file, pgoff, len);
	if (IS_ERR(ptr))
		return PTR_ERR(ptr);

	return (unsigned long) ptr;
}

#endif /* !CONFIG_MMU */
```

#  references

[GPU的UM实现](https://blog.csdn.net/u010916862/article/details/135907482)   