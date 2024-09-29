// refer to pmd_t *mm_find_pmd(struct mm_struct *mm, unsigned long address)
static struct page * test_walk_page_table(struct mm_struct *mm ,unsigned long addr)
{
    pgd_t *pgd;
    p4d_t *p4d;
    pte_t *ptep, pte;
    pud_t *pud;
    pmd_t *pmd;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0 ;
    unsigned long pfn = 0 ;
    unsigned long mpfn = 0;
    struct page *page = NULL;
    //struct mm_struct *mm = current->mm;

    struct vm_area_struct *vma = find_vma(mm, addr);
    pgd = pgd_offset(mm, addr);
    if (pgd_none(*pgd) || pgd_bad(*pgd))
        goto out;
    //printk(KERN_NOTICE "Valid pgd");

    p4d = p4d_offset(pgd, addr);
    if (!p4d_present(*p4d))
        goto out;
    pud = pud_offset(p4d, addr);
    if (pud_none(*pud) || pud_bad(*pud))
        goto out;
    //printk(KERN_NOTICE "Valid pud");

    pmd = pmd_offset(pud, addr);
    if (pmd_none(*pmd) || pmd_bad(*pmd))
        goto out;
    //printk(KERN_NOTICE "Valid pmd");

    //ptep = pte_offset_kernel(pmd, addr);
    ptep = pte_offset_map(pmd, addr);
    if (!ptep)
    {
        goto out1;
    }
    //printk(KERN_NOTICE "Valid pte");
    pte = *ptep;
    page = pte_page(pte);
    if (page)
    {
            //page_addr = pte_val(pte) & PAGE_MASK;
            page_addr = pte_pfn(pte) << PAGE_SHIFT;
            page_offset = addr & ~PAGE_MASK;
            //paddr = page_addr + page_offset;
            paddr = page_addr | page_offset;
            printk(KERN_INFO " not swap pte,page frame struct is @ %p, and user paddr %lu, virt addr %lu", page, paddr, addr);
        }
    }
 out1:
    pte_unmap(ptep);
 out:
    return page;
}