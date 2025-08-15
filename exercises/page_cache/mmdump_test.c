/* mmdump.c
Ben Luo
*/
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <asm/pgtable.h>
#include <linux/sched/signal.h>


static int pid = -1;
module_param(pid, int, S_IRUGO);
#define p4d_pfn(x)      (0)
#define p4d_large(x)    (0)
#define pud_large(x)    (pud_sect(x))
#define pmd_large(x)    (pmd_sect(x))
static void printbinary(unsigned long x, int nbits)
{
    unsigned char buf[nbits+1];
    unsigned long mask = 1UL << (nbits - 1);
    int i = 0;
    while (mask != 0) {
        buf[i++] = (mask & x ? '1' : '0');
        mask >>= 1;
    }
    buf[i] = '\0';

    printk("%s", buf);
}

static int bad_address(void *p)
{
    unsigned long dummy;

    pr_info("bad_address \n");
    return 0;
    // return probe_kernel_address((unsigned long *)p, dummy);
}

static void dump_pagetable(unsigned long address, pgd_t * pgd)
{
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;
    unsigned long pte_v;

    if (bad_address(pgd))
        goto bad;

    //printk("PGD %lx ", pgd_val(*pgd));

    if (!pgd_present(*pgd))
        goto out;

    pud = pud_offset(pgd, address);
    if (bad_address(pud))
        goto bad;

    printk("PUD %lx ", pud_val(*pud));
    if (!pud_present(*pud) || pud_large(*pud))
        goto out;

    pmd = pmd_offset(pud, address);
    if (bad_address(pmd))
        goto bad;

    printk("PMD %lx ", pmd_val(*pmd));
    if (!pmd_present(*pmd) || pmd_large(*pmd))
        goto out;

    pte = pte_offset_kernel(pmd, address);
    if (bad_address(pte))
        goto bad;

    pte_v = pte_val(*pte);
    printk("PTE %lx|", pte_v&~(PAGE_SIZE-1));
    printbinary(pte_v, 8);
out:
    printk("\n");
    return;
bad:
    printk("BAD\n");
}

void mmdump(struct mm_struct *mm)
{
    struct vm_area_struct * vma = mm->mmap;
    struct file *file ;
    struct address_space *mapping = NULL;
    struct address_space_operations *ops=NULL;
    while(vma) {
        //if(vma->vm_flags)
        if (vma->vm_ops){
                    pr_info("**************** vma->vm_ops: %p \n", vma->vm_ops);
        }
        else {
            pr_info("****************** vma->vm_ops is null \n");
        }
        if (vma->vm_file){
            file = vma->vm_file;
            mapping = file->f_mapping; 
            ops=mapping->a_ops;
            if (ops){
                pr_info("mapping->a_ops: %p \t", ops);
            }
            printk(KERN_INFO "%lx %lx %s\n", vma->vm_start, vma->vm_end, vma->vm_file->f_path.dentry->d_iname);
        }
        else{
            unsigned long addr = vma->vm_start;
            printk(KERN_INFO "%lx %lx [anon], vma->vm_file is null \n", vma->vm_start, vma->vm_end);
            printk(KERN_INFO "---------------start------------------\n");
            for (;addr < vma->vm_end; addr += PAGE_SIZE) 
                dump_pagetable(addr, mm->pgd);
            printk(KERN_INFO "----------------end---------------\n");
        }

        vma = vma->vm_next;
    }
}

int init_module(void)
{
    struct task_struct *task;
    printk(KERN_INFO "*************************** mmdump module load \n");
    for_each_process(task)
    {
        if (pid == -1)
            printk("echo a pid to me");

        if (task->pid == pid)
            mmdump(task->mm);
    }

    return 0;
}

void cleanup_module(void)
{
    printk(KERN_INFO "Cleaning Up.\n");
}
