#include <linux/module.h> 
#include <linux/kernel.h> 
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/moduleparam.h>
#include <linux/stat.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/unistd.h>
#include <asm/current.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <asm/pgtable.h>
#include <linux/sched/signal.h>

int totalrss=0, totalva=0,pre=0, mapped=0, notpre=0, pg=0;
pgd_t *PGDir;
pud_t *PUDir;
pmd_t *PMDir;
pte_t *PTEnt ;

static int upid = 0;

module_param(upid, int, S_IRUSR);

int virphy(struct mm_struct *mm, unsigned long ptrval)
{
	unsigned long phyadd;
	PGDir = pgd_offset(mm, ptrval);
	if (pgd_none(*PGDir))
	{
		//printk("PGD not found\n");
		notpre++;
		return 0;
	}
	PUDir = pud_offset(PGDir, ptrval);
	if (pud_none(*PUDir))
	{	
		//printk("PUD not found\n");
		notpre++;
		return 0;
	}
	
	PMDir = pmd_offset(PUDir, ptrval);
	if (pmd_none(*PMDir))
	{
		//printk("PGD not found\n");
		notpre++;
		return 0;
	}

	PTEnt = pte_offset_map(PMDir, ptrval);

	if (PTEnt == NULL)
	{
		notpre++;
		return 0;
	}

	if(!pte_present(*PTEnt))
	{
		mapped++;
		return 0;
	}

	else
	{
		phyadd = pte_val(*PTEnt);
		phyadd = phyadd & PAGE_MASK;
		phyadd = phyadd | (ptrval & ~(PAGE_MASK));
		printk("Virtual Address: %lx - Physical Address: %lx \n", ptrval, phyadd);
		pre++;
		return 1; 
	}

}


int sampleModuleStart(void)
{
	struct mm_struct *mms;
	struct vm_area_struct *vms;
	struct task_struct *task;
	unsigned long vaddr;
	printk(KERN_EMERG "********rss Module start*********\n");
    	for_each_process(task)
    	{
		if(task->pid == upid)
		{
    			break;
		}
    	}
	if(task->mm)
	{
		mms = task ->mm;
		if(mms->mmap)
		{
			vms = mms -> mmap;

			
			while( vms != NULL )
			{
				vaddr = vms->vm_start;
				pre=0;
				notpre=0;
				mapped = 0;
				pg=0;
				while(vaddr < vms->vm_end)
				{
					totalrss = totalrss + virphy(mms,vaddr);
					vaddr = vaddr + PAGE_SIZE;
					totalva++;
					pg++;
				}
				printk("Mapped and present pages: %d\n", pre);
				printk("Mapped but not present pages: %d\n", mapped);
				printk("Not mapped pages: %d\n", notpre);
				printk("RSS for this virtual area is %dK\n", pre*4);
				printk("Total pages in the virtual area: %d\n", pg);
				vms = vms->vm_next;
			}
		}
		else
			printk("mmap not present...\n");

	}
	else
		printk("mm not present...\n");

	printk("Total RSS %dK\n", totalrss*4);
	printk("Total Virtual address size %dK\n", totalva*4);
	
        return 0;
}
void sampleModuleEnd(void)
{
	printk(KERN_EMERG "********Exiting Module*********\n");
}


module_init(sampleModuleStart);
module_exit(sampleModuleEnd);


MODULE_LICENSE("Suhit");
