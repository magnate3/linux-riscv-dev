

# get_user_pages_remote 和get_user_pages_fast的区别

```
static int put_pfn(unsigned long pfn, int prot)
{
	if (!is_invalid_reserved_pfn(pfn)) {
		struct page *page = pfn_to_page(pfn);
		if (prot & IOMMU_WRITE)
			SetPageDirty(page);
		put_page(page);
		return 1;
	}
	return 0;
}

static int vaddr_get_pfn(struct mm_struct *mm, unsigned long vaddr,
			 int prot, unsigned long *pfn)
{
	struct page *page[1];
	struct vm_area_struct *vma;
	int ret;

	if (mm == current->mm) {
		ret = get_user_pages_fast(vaddr, 1, !!(prot & IOMMU_WRITE),
					  page);
	} else {
		unsigned int flags = 0;

		if (prot & IOMMU_WRITE)
			flags |= FOLL_WRITE;

		down_read(&mm->mmap_sem);
		ret = get_user_pages_remote(NULL, mm, vaddr, 1, flags, page,
					    NULL, NULL);
		up_read(&mm->mmap_sem);
	}

	if (ret == 1) {
		*pfn = page_to_pfn(page[0]);
		return 0;
	}

	down_read(&mm->mmap_sem);

	vma = find_vma_intersection(mm, vaddr, vaddr + 1);

	if (vma && vma->vm_flags & VM_PFNMAP) {
		*pfn = ((vaddr - vma->vm_start) >> PAGE_SHIFT) + vma->vm_pgoff;
		if (is_invalid_reserved_pfn(*pfn))
			ret = 0;
	}

	up_read(&mm->mmap_sem);
	return ret;
}
```