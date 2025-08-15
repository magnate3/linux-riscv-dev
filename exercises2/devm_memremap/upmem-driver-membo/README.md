


# struct vm_operations_struct dpu_dax_vm_op
```
static vm_fault_t dpu_dax_fault(
#if LINUX_VERSION_CODE == KERNEL_VERSION(3, 10, 0)
	struct vm_area_struct *vma,
#endif
	struct vm_fault *vmf)
{
	return dpu_dax_huge_fault(vmf, PE_SIZE_PTE);
}

static const struct vm_operations_struct dpu_dax_vm_ops = {
	.huge_fault = dpu_dax_huge_fault,
	.fault = dpu_dax_fault,
};
```


# vmf_insert_pfn_pud
```
#ifdef __x86_64__
static vm_fault_t dpu_dax_pud_huge_fault(struct vm_fault *vmf, void *vaddr)
{
	struct file *filp = vmf->vma->vm_file;
	struct dpu_region *region = filp->private_data;
	phys_addr_t paddr;
	unsigned long pud_addr = (unsigned long)vaddr & PUD_MASK;
	unsigned long pgoff;
	pfn_t pfn;

	pgoff = linear_page_index(vmf->vma, pud_addr);
	paddr = ((phys_addr_t)__pa(region->base) + pgoff * PAGE_SIZE) &
		PUD_MASK;
	pfn = phys_to_pfn_t(paddr, PFN_DEV | PFN_MAP);

#if LINUX_VERSION_CODE == KERNEL_VERSION(3, 10, 0)
	return vmf_insert_pfn_pud(vmf->vma, (unsigned long)vaddr, vmf->pud, pfn,
				  vmf->flags & FAULT_FLAG_WRITE);
#else
	return vmf_insert_pfn_pud(vmf, pfn, vmf->flags & FAULT_FLAG_WRITE);
#endif
}
#endif
```