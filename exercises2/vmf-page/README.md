
# uio

```
static int uio_vma_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
{
	struct uio_device *idev = vma->vm_private_data;
	struct page *page;
	unsigned long offset;
	void *addr;

	int mi = uio_find_mem_index(vma);
	if (mi < 0)
		return VM_FAULT_SIGBUS;

	/*
	 * We need to subtract mi because userspace uses offset = N*PAGE_SIZE
	 * to use mem[N].
	 */
	offset = (vmf->pgoff - mi) << PAGE_SHIFT;

	addr = (void *)(unsigned long)idev->info->mem[mi].addr + offset;
	if (idev->info->mem[mi].memtype == UIO_MEM_LOGICAL)
		page = virt_to_page(addr);
	else
		page = vmalloc_to_page(addr);
	get_page(page);
	vmf->page = page;
	return 0;
}
```

#  mmap-kernel-transfer-data


```
//static int mmap_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
static int mmap_fault(struct vm_fault *vmf)
{
	struct page *page;
	struct mmap_info *info;
        struct vm_area_struct *vma = vmf->vma;
	info = (struct mmap_info *)vma->vm_private_data;
	if (!info->data) {
		printk("No data\n");
		return 0;
	}

	page = virt_to_page(info->data);

	get_page(page);
	vmf->page = page;

	return 0;
}
```