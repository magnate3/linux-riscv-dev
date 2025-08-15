#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <asm/io.h>
#include <asm/current.h>
#include <asm/pgtable.h>


static int hello_init(void)
{
long addr=0xc0000000;
pgd_t *pgd;
pte_t *ptep, pte;
pud_t *pud;
pmd_t *pmd;
struct page *page = NULL;
struct mm_struct *mm = current->mm;
pgd = pgd_offset(mm, addr);

    printk("%p\n",pgd);

    pud = pud_offset(pgd, addr);

    printk("%p\n",pud);

    pmd = pmd_offset(pud, addr);

    printk("%p\n",pmd);

    ptep = pte_offset_kernel(pmd, addr);
  
    pte = *ptep;
    
    page = pte_page(pte);
   
    if (page)
        printk(KERN_INFO "page frame struct is @ %p", page);

return 0;
}

static void hello_exit(void)
{
long addr=0xc0000000;
pgd_t *pgd;
pte_t *ptep, pte;
pud_t *pud;
pmd_t *pmd;
struct page *page = NULL;
struct mm_struct *mm = current->mm;
    pgd = pgd_offset(mm, addr);

    printk("%p\n",pgd);

    pud = pud_offset(pgd, addr);

    printk("%p\n",pud);

    pmd = pmd_offset(pud, addr);

    printk("%p\n",pmd);

    ptep = pte_offset_kernel(pmd, addr);
        
    pte = *ptep;

    page = pte_page(pte);
    if (page)
        printk(KERN_INFO "page frame struct is @ %p", page);
}
module_init(hello_init);
module_exit(hello_exit);