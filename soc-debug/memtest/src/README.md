
#
 PGDIR_SIZE=40000000 
```
pgtable.h
#define FIXADDR_SIZE     PGDIR_SIZE
```



#  dbg_create_pgd_mapping
条用page_init之前对console sbi进行了初始化可以调用pr_info打印日志
```
setup_arch->page_init-> setup_vm_final->dbg_create_pgd_mapping建立线性映射
```

```
void __init dbg_create_pgd_mapping(pgd_t *pgdp,
                                      uintptr_t va, phys_addr_t pa,
                                      phys_addr_t sz, pgprot_t prot)
{
        pgd_next_t *nextp;
        phys_addr_t next_phys;
        uintptr_t pgd_idx = pgd_index(va);
#if TEST_SET_UP_VM_FINAL_LOG

        pr_info("debug va addr % lx,  pgd_idx %lx and pgdp %lx , sz == PGDIR_SIZE ?  %d \n", va,  pgd_idx, pgdp, sz == PGDIR_SIZE);
        pr_info("debug pgd_val(pgdp[pgd_idx]) :%d  \n", pgd_val(pgdp[pgd_idx]));
#endif
        if (sz == PGDIR_SIZE) {
                if (pgd_val(pgdp[pgd_idx]) == 0)
                        pgdp[pgd_idx] = pfn_pgd(PFN_DOWN(pa), prot);
                return;
        }

        if (pgd_val(pgdp[pgd_idx]) == 0) {
#if TEST_SET_UP_VM_FINAL_LOG
                pr_info("debug call alloc_pmd_fixmap \n");
                next_phys = alloc_pgd_next(va);
                pgdp[pgd_idx] = pfn_pgd(PFN_DOWN(next_phys), PAGE_TABLE);
                pr_info("debug call get_pmd_virt_fixmap\n");
                nextp = get_pgd_next_virt(next_phys);
                memset(nextp, 0, PAGE_SIZE);
#else
                next_phys = alloc_pgd_next(va);
                pgdp[pgd_idx] = pfn_pgd(PFN_DOWN(next_phys), PAGE_TABLE);
                nextp = get_pgd_next_virt(next_phys);
                memset(nextp, 0, PAGE_SIZE);
#endif
        } else {
                next_phys = PFN_PHYS(_pgd_pfn(pgdp[pgd_idx]));
                nextp = get_pgd_next_virt(next_phys);
        }

        dbg_create_pmd_mapping(nextp, va, pa, sz, prot);
}
static void __init dbg_create_pmd_mapping(pmd_t *pmdp,
                                      uintptr_t va, phys_addr_t pa,
                                      phys_addr_t sz, pgprot_t prot)
{
        pte_t *ptep;
        phys_addr_t pte_phys;
        uintptr_t pmd_idx = pmd_index(va);

#if TEST_SET_UP_VM_FINAL_LOG
        pr_info("debug pmd_idx %lx and pmdp %lx \n", pmd_idx, pmdp);
#endif
        if (sz == PMD_SIZE) {
                if (pmd_none(pmdp[pmd_idx]))
                        pmdp[pmd_idx] = pfn_pmd(PFN_DOWN(pa), prot);
                return;
        }

        if (pmd_none(pmdp[pmd_idx])) {
                pte_phys = pt_ops.alloc_pte(va);
                pmdp[pmd_idx] = pfn_pmd(PFN_DOWN(pte_phys), PAGE_TABLE);
                ptep = pt_ops.get_pte_virt(pte_phys);
                memset(ptep, 0, PAGE_SIZE);
        } else {
                pte_phys = PFN_PHYS(_pmd_pfn(pmdp[pmd_idx]));
                ptep = pt_ops.get_pte_virt(pte_phys);
        }

        dbg_create_pte_mapping(ptep, va, pa, sz, prot);
}

static void __init dbg_create_pte_mapping(pte_t *ptep,
				      uintptr_t va, phys_addr_t pa,
				      phys_addr_t sz, pgprot_t prot)
{
	uintptr_t pte_idx = pte_index(va);

	BUG_ON(sz != PAGE_SIZE);

#if TEST_SET_UP_VM_FINAL_LOG
        pr_info("debug pte_idx %lx and ptep %lx \n", pte_idx, ptep);	
#endif
	if (pte_none(ptep[pte_idx]))
		ptep[pte_idx] = pfn_pte(PFN_DOWN(pa), prot);
}
```