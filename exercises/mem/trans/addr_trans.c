#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <asm/pgtable.h>
#include <asm/current.h>

static void show_page_info(void)
{
	pr_info("show page table info:\n");
	pr_info("PAGE_SIZE = 0x%lx (%lu)\n", PAGE_SIZE, PAGE_SIZE);
	pr_info("PAGE_MASK = 0x%lx\n", PAGE_MASK);
	pr_info("PAGE_OFFSET = 0x%lx\n", PAGE_OFFSET);
	//pr_info("PHYS_OFFSET = 0x%lx\n", PHYS_OFFSET);

	pr_info("PGDIR_OFFSET = %u, PGDIR_SIZE = 0x%lx(%lu)\n", PGDIR_SHIFT, PGDIR_SIZE, PGDIR_SIZE);

	pr_info("PUD_SHIFT = %u\n", PUD_SHIFT);
	pr_info("PMD_SHIFT = %u\n", PMD_SHIFT);
	pr_info("PAGE_SHIFT = %u\n", PAGE_SHIFT);

	pr_info("PTRS_PER_PGD = %u\n", PTRS_PER_PGD);
	pr_info("PTRS_PER_PUD = %u\n", PTRS_PER_PUD);
	pr_info("PTRS_PER_PMD = %u\n", PTRS_PER_PMD);
	pr_info("PTRS_PER_PTE = %u\n", PTRS_PER_PTE);
}

static void my_virt_to_phy(unsigned long va)
{
	pgd_t *pgd;
	p4d_t *p4d;
	pud_t *pud;
	pmd_t *pmd;
	pte_t *pte;
	unsigned long pa_offset;
	unsigned long pa_addr;
	unsigned long pa;

	pgd = pgd_offset(current->mm, va);
	pr_info("pad val= 0x%lx\n", pgd_val(*pgd));
	pr_info("pgd index = %lu\n", pgd_index(va));
	if (pgd_none(*pgd)) {
		pr_err("vaddr:0x%lx is not mapped in pgd\n", va);
		return;
	}

	p4d = p4d_offset(pgd, va);
	pr_info("p4d val= 0x%lx\n", p4d_val(*p4d));
	pr_info("p4d index = %lu\n", p4d_index(va));
	if (p4d_none(*p4d)) {
		pr_err("vaddr:0x%lx is not mapped in p4d\n", va);
		return;
	}

	pud = pud_offset(p4d, va);
	pr_info("pud val= 0x%lx\n", pud_val(*pud));
	pr_info("pud index = %lu\n", pud_index(va));
	if (pud_none(*pud)) {
		pr_err("vaddr:0x%lx is not mapped in pud\n", va);
		return;
	}

	pmd = pmd_offset(pud, va);
	pr_info("pmd val= 0x%lx\n", pmd_val(*pmd));
	pr_info("pmd index = %lu\n", pmd_index(va));
	if (pmd_none(*pmd)) {
		pr_err("vaddr:0x%lx is not mapped in pmd\n", va);
		return;
	}

	pte = pte_offset_kernel(pmd, va);
	pr_info("pte val= 0x%lx\n", pte_val(*pte));
	pr_info("pte index = %lu\n", pte_index(va));
	if (pte_none(*pte)) {
		pr_err("vaddr:0x%lx is not mapped in pte\n", va);
		return;
	}

	pa_addr = pte_val(*pte) & PAGE_MASK;
	pa_offset = va & (~(PAGE_MASK));
	pa = pa_addr | pa_offset;
	pr_info("pa_addr = 0x%lx, pa_offset = 0x%lx\n", pa_addr, pa_offset);
	pr_info("va = 0x%lx, pa = 0x%lx\n", va, pa);
}

static int __init my_addr_trans_init(void)
{
	unsigned long test_addr = 0;
	unsigned long v_addr = 0;
	u8 i;

	show_page_info();

	pr_info("[%s] i addr:0x%lx\n", __func__, (unsigned long)&i);

	test_addr = (unsigned long)kmalloc(sizeof(unsigned long), GFP_KERNEL);
	pr_info("[%s] kmalloc: virt addr:0x%lx %pK\n", __func__, test_addr, (void *)test_addr);
	*((unsigned long *)test_addr)  = 1;
	my_virt_to_phy(test_addr);

	v_addr = (unsigned long)vmalloc(100);
	pr_info("[%s] vmalloc: virt addr:%pK\n", __func__, (void *)v_addr);
	my_virt_to_phy(v_addr);

	kfree((void *)test_addr);
	vfree((void *)v_addr);

	return 0;
}

static void __exit my_addr_trans_exit(void)
{
	pr_info("[%s]\n", __func__);
}

module_init(my_addr_trans_init);
module_exit(my_addr_trans_exit);

MODULE_LICENSE("Dual MPL/GPL");
MODULE_AUTHOR("yanli.qian");
MODULE_DESCRIPTION("virt addr to phy addr");
