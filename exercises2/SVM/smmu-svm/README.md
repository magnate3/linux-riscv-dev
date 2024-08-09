

# /dev/devmm_svm    
smmu svm vma  IOPF
```
int devmm_insert_pages_range_to_vma(struct devmm_svm_process *svm_process, unsigned long va,
    u64 page_num, struct page **inpages)
{
    struct vm_area_struct *vma = svm_process->vma;
    phys_addr_t offset;
    int ret;
    u64 i;

    vma->vm_page_prot = devmm_make_pgprot(0);
    for (i = 0; i < page_num; i++) {
        offset = (i << PAGE_SHIFT);
        ret = remap_pfn_range(vma, va + offset, page_to_pfn(inpages[i]), PAGE_SIZE, vma->vm_page_prot);
        if (ret) {
            devmm_drv_err("vm_insert_page() failed,ret=%d. va=0x%lx, i=%llu, page_num=%llu.\n",
                ret, va, i, page_num);
            /* will not return fail ,so free page here */
            devmm_free_pages(1, &inpages[i], svm_process);
        } else {
            devmm_pin_page(inpages[i]);
        }
        devmm_svm_stat_page_inc(PAGE_SIZE);
        devmm_svm_stat_pg_map_inc();
    }

    return 0;
}
```