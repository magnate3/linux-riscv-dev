#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/export.h>
#include <linux/mm_types.h>
#include <linux/module.h>

static unsigned long vaddr = 0;
static unsigned long cr0, cr3;

static void print_pgtable_macro(void)
{
    cr0 = read_cr0();
    cr3 = read_cr3_pa();

    printk("cr0=0x%lx, cr3=0x%lx\n", cr0, cr3);

    printk("pgdir_SHIFT = %d\n", PGDIR_SHIFT);
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

static unsigned long vaddr2paddr(unsigned long vaddr)
{
    pgd_t *pgd;
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0;
    pgd = pgd_offset(current->mm, vaddr);
    printk("pgd_val=0x%lx, pdg_index=0x%lx\n", pgd_val(*pgd), pgd_index(vaddr));
    if (pgd_none(*pgd)) {
        printk("not mapped in pgd\n");
        return -1;
    }

    p4d = p4d_offset(pgd, vaddr);
    printk("p4d_val=0x%lx, p4d_index=0x%lx\n", p4d_val(*p4d), p4d_index(vaddr));
    if (p4d_none(*p4d)) {
        printk("not mapped in p4d\n");
        return -1;
    }

    pud = pud_offset(p4d, vaddr);
    printk("pud_val=0x%lx, pud_index=0x%lx\n", pud_val(*pud), pud_index(vaddr));
    //printk("pud_val=0x%lx, pud_index=0x%lx\n", pud_val(*pud));
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

static int __init v2p_init(void)
{
    printk("***************v2p moudle run:\n");
    printk("print page relate macro:\n");
    print_pgtable_macro();
    printk("vaddr to phy addr entry!\n");
    vaddr = __get_free_page(GFP_KERNEL);
    if (vaddr == 0) {
        printk("__get_free_pages fail\n");
        return 0;
    }
    printk("__get_free_page, alloc the free page vaddr=0x%lx\n", vaddr);
    printk("***************before write, vaddr2paddr:\n");
    vaddr2paddr(vaddr);
    sprintf((char *)vaddr, "hello physical memory");
    printk("***************after write, vaddr2paddr :\n");
    vaddr2paddr(vaddr);
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
