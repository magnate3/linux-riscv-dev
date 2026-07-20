#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/export.h>
#include <linux/mm_types.h>
#include <linux/module.h>
#include <linux/kallsyms.h>
#include <asm/tlbflush.h>
static unsigned long vaddr = 0;
static unsigned long cr0, cr3;

static void print_pgtable_macro(void)
{
    //cr0 = read_cr0();
    //cr3 = read_cr3_pa();

    //printk("cr0=0x%lx, cr3=0x%lx\n", cr0, cr3);

    printk("pgdir_SHIFT = %d\n", PGDIR_SHIFT);
    printk("PAGE_OFFSET = 0x%lx\n", PAGE_OFFSET);
    printk("P4D_SHIFT = %d\n", P4D_SHIFT);
    printk("PUD_SHIFT = %d\n", PUD_SHIFT);
    printk("PMD_SHIFT = %d\n", PMD_SHIFT);
    printk("PAGE_SHIFT = %d\n", PAGE_SHIFT);

    printk("PTRS_PER_PGD = %d\n", PTRS_PER_PGD);
    printk("PTRS_PER_P4D = %d\n", PTRS_PER_P4D);
    printk("PTRS_PER_PUD = %d\n", PTRS_PER_PUD);
    printk("PTRS_PER_PMD = %d\n", PTRS_PER_PMD);
    printk("PTRS_PER_PTE = %d\n", PTRS_PER_PTE);
    printk("PAGE_MASK = 0x%lx\n", PAGE_MASK);
}

static unsigned long vaddr2paddr_1(unsigned long vaddr)
{
    pgd_t *pgd;
    p4d_t p4d;
    //p4d_t *p4d;
    p4d_t *p4dp;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0;
    struct mm_struct *mm;
    struct mm_struct *__init_mm;
    
    mm = current->mm;
    pgd = pgd_offset(mm, vaddr);
    printk("pgd_val=0x%lx, pdg_index=0x%lx\n", pgd_val(*pgd), pgd_index(vaddr));
    if (pgd_none(*pgd)) {
        printk("not mapped in pgd\n");
        return -1;
    }
#if 1
    p4dp = p4d_offset(pgd, vaddr);
    p4d = READ_ONCE(*p4dp);
    printk("p4d_val=0x%lx\n", p4d_val(p4d));
    if (p4d_none(*p4d)) {
        printk("not mapped in p4d\n");
        return -1;
    }

    pud = pud_offset(p4dp, vaddr);
#else
    pud = pud_offset(pgd, vaddr);
#endif
    printk("pud_val=0x%lx\n", pud_val(*pud));
    if (pud_none(*pud)) {
        printk("not mapped in pud\n");
        return -1;
    }

    pmd = pmd_offset(pud, vaddr);
    printk("pmd_val=0x%lx, pmd_index=0x%lx\n", pmd_val(*pmd), pmd_index(vaddr));
    if (pmd_none(*pmd)) {
        printk("not mapped in pmd\n");
        return -1;
    }

    pte = pte_offset_kernel(pmd, vaddr);
    printk("pte_val=0x%lx, pte_index=0x%lx\n", pte_val(*pte), pte_index(vaddr));
    if (pte_none(*pte)) {
        printk("not mapped in pte\n");
        return -1;
    }

    page_addr = pte_val(*pte) & PAGE_MASK;
    page_offset = vaddr & ~ PAGE_MASK;
    paddr = page_addr | page_offset;
    printk("page_offset=0x%lx, page_addr=0x%lx\n", page_offset, page_addr);
    printk("vaddr=0x%lx, paddr=0x%lx\n", vaddr, paddr);
    return paddr;
}
static unsigned long vaddr2paddr(unsigned long vaddr)
{
    pgd_t *pgd;
    p4d_t p4d;
    //p4d_t *p4d;
    p4d_t *p4dp;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0;
    struct mm_struct *mm;
    struct mm_struct *__init_mm;
#if 0
    	if (is_ttbr0_addr(vaddr)) {
		/* TTBR0 */
		mm = current->active_mm;
		if (mm == &init_mm) {
			pr_alert("[%016lx] user address but active_mm is swapper\n",
				 vaddr);
			return;
		}
	} else if (is_ttbr1_addr(vaddr)) {
		/* TTBR1 */
		mm = &init_mm;
	} else {
		pr_alert("[%016lx] address between user and kernel address ranges\n",
			 vaddr);
		return;
	}
#endif
    
    mm = current->mm;
    //pgd = pgd_offset(mm, vaddr);
    //pgd = pgd_offset(current->mm, vaddr);
    if (vaddr > PAGE_OFFSET) {
		/* kernel virtual address */
        pr_info("kernel virtual address \n");
        __init_mm = (struct mm_struct *)kallsyms_lookup_name("init_mm");
	pgd = pgd_offset(__init_mm, vaddr);
   } else {
		/* user (process) virtual address */
        pr_info("user virtual address \n");
	pgd = pgd_offset(current->mm, vaddr);
    }
    printk("pgd_val=0x%lx, pdg_index=0x%lx\n", pgd_val(*pgd), pgd_index(vaddr));
    if (pgd_none(*pgd)) {
        printk("not mapped in pgd\n");
        return -1;
    }
#if 1
    p4dp = p4d_offset(pgd, vaddr);
    p4d = READ_ONCE(*p4dp);
    printk("p4d_val=0x%lx \n", p4d_val(p4d));
    //printk("p4d_val=0x%lx, p4d_index=0x%lx\n", p4d_val(*p4d), p4d_index(vaddr));
    if (p4d_none(*p4d)) {
        printk("not mapped in p4d\n");
        return -1;
    }

    pud = pud_offset(p4dp, vaddr);
#else
    pud = pud_offset(pgd, vaddr);
#endif
    printk("pud_val=0x%lx\n", pud_val(*pud));
    //printk("pud_val=0x%lx, pud_index=0x%lx\n", pud_val(*pud), pud_index(vaddr));
    if (pud_none(*pud)) {
        printk("not mapped in pud\n");
        return -1;
    }

    pmd = pmd_offset(pud, vaddr);
    printk("pmd_val=0x%lx, pmd_index=0x%lx\n", pmd_val(*pmd), pmd_index(vaddr));
    if (pmd_none(*pmd)) {
        printk("not mapped in pmd\n");
        return -1;
    }

    pte = pte_offset_kernel(pmd, vaddr);
    printk("pte_val=0x%lx, pte_index=0x%lx\n", pte_val(*pte), pte_index(vaddr));
    if (pte_none(*pte)) {
        printk("not mapped in pte\n");
        return -1;
    }

    page_addr = pte_val(*pte) & PAGE_MASK;
    page_offset = vaddr & ~ PAGE_MASK;
    paddr = page_addr | page_offset;
    printk("page_offset=0x%lx, page_addr=0x%lx\n", page_offset, page_addr);
    printk("vaddr=0x%lx, paddr=0x%lx\n", vaddr, paddr);
    return paddr;
}
#define p4d_pfn(x)	(0)
#define p4d_large(x)	(0)
#define pud_large(x)	(pud_sect(x))
#define pmd_large(x)	(pmd_sect(x))
static  struct vm_area_struct * test_find_vma(unsigned long addr)
{
	struct vm_area_struct *vma;
	struct mm_struct *mm = current->mm;
	down_read(&mm->mmap_sem);
	vma = find_vma(mm, addr);
	if (vma && addr >= vma->vm_start) {
		printk("found vma 0x%lx-0x%lx flag %lx for addr 0x%lx\n",
				vma->vm_start, vma->vm_end, vma->vm_flags, addr);
	} else {
		printk("no vma found for %lx\n", addr);
	}
	up_read(&mm->mmap_sem);
        if (vma->vm_start <= mm->brk && vma->vm_end >= mm->start_brk){
              return vma;
        }
        return NULL;
}
static void printk_pagetable(unsigned long addr)
{
	pgd_t *pgd;
	p4d_t *p4d;
	pud_t *pud = NULL;
	pmd_t *pmd = NULL;
	pte_t *pte = NULL;
	unsigned long phys_addr, offset;
	struct page *page = virt_to_page(addr);
        struct mm_struct *__init_mm;
        size_t pfn;
	printk("  ------------------------------\n");
	printk("  virtual %s addr: %016lx\n", addr > PAGE_OFFSET ? "kernel" :
	       "user", addr);
	printk("  page: %016lx\n", (unsigned long)page);

	if (addr > PAGE_OFFSET) {
		/* kernel virtual address */
                pr_info("kernel virtual address \n");
                __init_mm = (struct mm_struct *)kallsyms_lookup_name("init_mm");
		pgd = pgd_offset(__init_mm, addr);
	} else {
		/* user (process) virtual address */
		pgd = pgd_offset(current->mm, addr);
	}
	printk("  pgd: %016lx (%016lx) ", (unsigned long)pgd,
	       (unsigned long)pgd_val(*pgd));
	//printk_prot(pgd_val(*pgd), PT_LEVEL_PGD);

	p4d = p4d_offset(pgd, addr);
	printk("  p4d: %016lx (%016lx) ", (unsigned long)p4d,
	       (unsigned long)p4d_val(*p4d));
	//printk_prot(p4d_val(*p4d), PT_LEVEL_P4D);
	if (p4d_large(*p4d) || !p4d_present(*p4d)) {
		phys_addr = (unsigned long)p4d_pfn(*p4d) << PAGE_SHIFT;
		offset = addr & ~P4D_MASK;
		goto out;
	}

	pud = pud_offset(p4d, addr);
	printk("  pud: %016lx (%016lx) ", (unsigned long)pud,
	       (unsigned long)pud_val(*pud));
	//printk_prot(pud_val(*pud), PT_LEVEL_PUD);
	if (pud_large(*pud) || !pud_present(*pud)) {
		phys_addr = (unsigned long)pud_pfn(*pud) << PAGE_SHIFT;
		offset = addr & ~PUD_MASK;
		goto out;
	}

	pmd = pmd_offset(pud, addr);
	printk("  pmd: %016lx (%016lx) ", (unsigned long)pmd,
	       (unsigned long)pmd_val(*pmd));
	//printk_prot(pmd_val(*pmd), PT_LEVEL_PMD);
	if (pmd_large(*pmd) || !pmd_present(*pmd)) {
                pr_info("pmd_large(*pmd): %d, pmd_present(*pmd) : %d \n", pmd_large(*pmd), pmd_present(*pmd));
		phys_addr = (unsigned long)pmd_pfn(*pmd) << PAGE_SHIFT;
		offset = addr & ~PMD_MASK;
		goto out;
	}

	pte =  pte_offset_kernel(pmd, addr);
	printk("  pte: %016lx (%016lx) ", (unsigned long)pte,
	       (unsigned long)pte_val(*pte));
	//printk_prot(pte_val(*pte), PT_LEVEL_PTE);
	phys_addr = (unsigned long)pte_pfn(*pte) << PAGE_SHIFT;
	offset = addr & ~PAGE_MASK;

out:
	printk("  p4d_page: %016lx\n", (unsigned long)p4d_page(*p4d));
	if (pud)
		printk("  pud_page: %016lx\n", (unsigned long)pud_page(*pud));
	if (pmd)
		printk("  pmd_page: %016lx\n", (unsigned long)pmd_page(*pmd));

	if (pte)
        {
		
        pte_t newpte;
        printk("  pte_page: %016lx\n", (unsigned long)pte_page(*pte));
	printk("  physical addr: %016lx\n", phys_addr | offset);
	printk("  page addr: %016lx\n", phys_addr);
        pfn = pte_pfn(*pte); //with the old pte
	printk("\n pfn is: %x",pfn); 
        struct vm_area_struct *vma = test_find_vma(addr);
        newpte = pfn_pte(pfn, vma->vm_page_prot);
        printk("\naddress of the newpte is %x" , newpte);
        	//setting these changes
        set_pte(pte, newpte);
        flush_tlb_page(vma,vma->vm_start);
        printk(" new pte_page: %016lx\n", (unsigned long)pte_page(newpte));
        }
	printk("  ------------------------------\n");

}

static int __init v2p_init(void)
{
    printk("****************print page relate macro:\n");
    print_pgtable_macro();
    printk("vaddr to phy addr entry!\n");
    vaddr = __get_free_page(GFP_KERNEL);
    if (vaddr == 0) {
        printk("__get_free_pages fail\n");
        return 0;
    }
    sprintf((char *)vaddr, "hello physical memory");
    printk("__get_free_page, alloc the free page vaddr=0x%lx\n", vaddr);
    printk("************** call vaddr2paddr_1 \n");
    vaddr2paddr_1(vaddr);
    printk("************** call vaddr2paddr \n");
    vaddr2paddr(vaddr);
    printk("************** call printk_pagetable\n");
    printk_pagetable(vaddr);
    return 0;
}

static void __exit v2p_exit(void)
{
    printk("free the alloc page and leave the v2p!\n");
    free_page(vaddr);
}

module_init(v2p_init);
module_exit(v2p_exit);

MODULE_LICENSE("GPL");
