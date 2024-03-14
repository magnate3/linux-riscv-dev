
# insmod hybridmem_test.ko 

```
insmod hybridmem_test.ko 
mknod /dev/hybridmem c 224 0
```

```
static vm_fault_t vma_fault(struct vm_fault* vmf)
{
    struct vm_area_struct *vma = vmf->vma;
    struct page* page = alloc_page(GFP_HIGHUSER_MOVABLE | __GFP_ZERO);
    unsigned long pfn_start = page_to_pfn(page)<< PAGE_SHIFT + vma->vm_pgoff;
    if(!page)
        return VM_FAULT_SIGBUS;
    vmf->page = page;
    //printk("fault, is_write = %d, vaddr = %lx, paddr = %lx\n", vmf->flags & FAULT_FLAG_WRITE, vmf->address, (size_t)virt_to_phys(page_address(page)));
    printk("fault, is_write = %d, vaddr = %lx, paddr = %lx\n", vmf->flags & FAULT_FLAG_WRITE, vmf->address, pfn_start);
    return 0;
}
```