#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <asm/page.h>
#include <asm/pgtable.h>

int __init wip_init(void)
{
	unsigned long va = 0xffff78ee0000;
	int pid = 8703;
	//struct page p;
	unsigned long long pageFN;
	unsigned long long pa;

	pgd_t *pgd;
	pmd_t *pmd;
	pud_t *pud;
	pte_t *pte;
	
	struct mm_struct *mm;

	int found = 0;

	struct task_struct *task;
        struct pid *vpid = find_vpid(pid);
        if (vpid != NULL) {
		printk("the find_vpid result's count is: %d\n",
		       vpid->count.counter);
		printk("the find_vpid result's level is: %d\n", vpid->level);
	} else {
		printk("failed to find_vpid");
                return 0;
	}
        task=pid_task(vpid,PIDTYPE_PID);
	mm = task->mm;
	pgd  = pgd_offset(mm,va);
	if(!pgd_none(*pgd) && !pgd_bad(*pgd))
	{
		pud = pud_offset(pgd,va);
		if(!pud_none(*pud) && !pud_bad(*pud))
		{
			pmd = pmd_offset(pud,va);
			if(!pmd_none(*pmd) && !pmd_bad(*pmd))
			{
				pte = pte_offset_kernel(pmd,va);
				if(!pte_none(*pte))
				{
					pageFN = pte_pfn(*pte);
					pa = ((pageFN<<12)|(va&0x00000FFF));
					found = 1;
					printk(KERN_ALERT "Physical Address: 0x%08llx\npfn: 0x%04llx\n", pa, pageFN);
				}
			}
		}
	}
	if(pgd_none(*pgd) || pud_none(*pud) || pmd_none(*pmd) || pte_none(*pte))
	{
		unsigned long long swapID = (pte_val(*pte) >> 32);
		found = 1;
		printk(KERN_ALERT "swap ID: 0x%08llx\n", swapID);
	}
	if(found == 0)
	{
		printk(KERN_ALERT "not available\n");
	}
return 0;	
}

void __exit wip_exit(void)
{
	printk(KERN_ALERT "Removed wip_mod");
}

module_init(wip_init);
module_exit(wip_exit);
MODULE_LICENSE("GPL");
