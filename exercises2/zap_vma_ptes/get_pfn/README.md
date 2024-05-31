

#  get_pfn
```
static unsigned long get_pfn(unsigned long virtpfn, int write)
{
	struct page *page;

	/* gup me one page at this address please! */
	if (get_user_pages_fast(virtpfn << PAGE_SHIFT, 1, write, &page) == 1)
		return page_to_pfn(page);

	/* This value indicates failure. */
	return -1UL;
}
```


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


#  pfn_to_page
```
struct page *vm_normal_page(struct vm_area_struct *vma, unsigned long addr,
				pte_t pte)
{
	unsigned long pfn = pte_pfn(pte);

	if (HAVE_PTE_SPECIAL) {
		if (likely(!pte_special(pte)))
			goto check_pfn;
		if (vma->vm_ops && vma->vm_ops->find_special_page)
			return vma->vm_ops->find_special_page(vma, addr);
		if (vma->vm_flags & (VM_PFNMAP | VM_MIXEDMAP))
			return NULL;
		if (!is_zero_pfn(pfn))
			print_bad_pte(vma, addr, pte, NULL);
		return NULL;
	}

	/* !HAVE_PTE_SPECIAL case follows: */

	if (unlikely(vma->vm_flags & (VM_PFNMAP|VM_MIXEDMAP))) {
		if (vma->vm_flags & VM_MIXEDMAP) {
			if (!pfn_valid(pfn))
				return NULL;
			goto out;
		} else {
			unsigned long off;
			off = (addr - vma->vm_start) >> PAGE_SHIFT;
			if (pfn == vma->vm_pgoff + off)
				return NULL;
			if (!is_cow_mapping(vma->vm_flags))
				return NULL;
		}
	}

	if (is_zero_pfn(pfn))
		return NULL;
check_pfn:
	if (unlikely(pfn > highest_memmap_pfn)) {
		print_bad_pte(vma, addr, pte, NULL);
		return NULL;
	}

	/*
	 * NOTE! We still have PageReserved() pages in the page tables.
	 * eg. VDSO mappings can cause them to exist.
	 */
out:
	return pfn_to_page(pfn);
}
```