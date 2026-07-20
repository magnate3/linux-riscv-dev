#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <asm/uaccess.h>
#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/kallsyms.h>
static  struct  class *sample_class;
#define p4d_pfn(x)	(0)
#define p4d_large(x)	(0)
#define pud_large(x)	(pud_sect(x))
#define pmd_large(x)	(pmd_sect(x))
//arm64/mm/hugetlbpage.c
int pmd_huge(pmd_t pmd)
{
        return pmd_val(pmd) && !(pmd_val(pmd) & PMD_TABLE_BIT);
}
int pud_huge(pud_t pud)
{
#ifndef __PAGETABLE_PMD_FOLDED
        return pud_val(pud) && !(pud_val(pud) & PUD_TABLE_BIT);
#else
        return 0;
#endif
}
static void printk_pagetable(unsigned long addr)
{
	pgd_t *pgd;
	p4d_t *p4d;
	pud_t *pud = NULL;
	pmd_t *pmd = NULL;
	pte_t *pte = NULL;
        unsigned long pfn1 = 0;
        unsigned long pfn2 = 0;
	unsigned long phys_addr, offset;
        struct mm_struct *__init_mm;
        struct vm_area_struct * vma;
        char * myaddr;
        struct page * page;
	printk("  ------------------------------\n");
	printk("  virtual %s addr: %016lx\n", addr > PAGE_OFFSET ? "kernel" : "user", addr);

	if (addr > PAGE_OFFSET) {
		/* kernel virtual address */
                pr_info("kernel virtual address \n");
                __init_mm = (struct mm_struct *)kallsyms_lookup_name("init_mm");
		pgd = pgd_offset(__init_mm, addr);
                vma = find_vma(__init_mm, addr);
	} else {
		/* user (process) virtual address */
		pgd = pgd_offset(current->mm, addr);
                vma = find_vma(current->mm, addr);
	}
        if (vma->vm_flags & VM_HUGETLB){
             pr_info("vm flag == VM_HUGETLB is true \n");

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
# if 1
        if (pud_huge(*pud)) {
             pr_info("pud_huge is true \n");
        }
                     
#endif
	printk("  pud: %016lx (%016lx) ", (unsigned long)pud,
	       (unsigned long)pud_val(*pud));
	//printk_prot(pud_val(*pud), PT_LEVEL_PUD);
	if (pud_large(*pud) || !pud_present(*pud)) {
                pr_info("pud_large(*pud): %d, pud_present(*pud) : %d \n", pud_large(*pud), pud_present(*pud));
		phys_addr = (unsigned long)pud_pfn(*pud) << PAGE_SHIFT;
		offset = addr & ~PUD_MASK;
		goto out;
	}

	pmd = pmd_offset(pud, addr);
# if 1
        if (pmd_huge(*pmd)) {
             pr_info("pmd_huge is true \n");
        }
                     
#endif
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
        pfn1 = pte_pfn(*pte);
	printk("  pte: %016lx (%016lx), pfn : %016lx ", (unsigned long)pte,
	       (unsigned long)pte_val(*pte),pfn1);
	//printk_prot(pte_val(*pte), PT_LEVEL_PTE);
	phys_addr = (unsigned long)pte_pfn(*pte) << PAGE_SHIFT;
	offset = addr & ~PAGE_MASK;
        if (vma && vma->vm_flags & VM_PFNMAP) {
		pfn2 = ((addr - vma->vm_start) >> PAGE_SHIFT) + vma->vm_pgoff;
		 
                pr_info("vma addr %lu ,and   pfn1 : %016lx , pfn2 : %016lx \n", addr,pfn1, pfn2);
		//if (is_invalid_reserved_pfn(*pfn))
	}
        if(NULL == pte || 0 == pfn1) {
            goto out;
        }
        page = pfn_to_page(pfn1);
        printk(KERN_INFO "Got page.\n");
        myaddr = kmap(page);
        //printk(KERN_INFO "%s\n", myaddr);
        strcpy(myaddr, "kernel Mohan");
out:
	printk("  p4d_page: %016lx\n", (unsigned long)p4d_page(*p4d));
	if (pud)
		printk("  pud_page: %016lx\n", (unsigned long)pud_page(*pud));
	if (pmd)
		printk("  pmd_page: %016lx\n", (unsigned long)pmd_page(*pmd));
	if (pte)
		printk("  pte_page: %016lx\n", (unsigned long)pte_page(*pte));
	printk("  physical addr: %016lx\n", phys_addr | offset);
	printk("  page addr: %016lx\n", phys_addr);
	printk("  ------------------------------\n");
}

static int sample_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return (0);
}
static int sample_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return (0);
}
static ssize_t sample_read(struct file *file, char __user *buf, size_t count,
                           loff_t *off)
{
        unsigned long arg = (unsigned long)buf;
        struct vm_area_struct *vma = NULL;
        int res;
        printk(KERN_INFO "%s\n", __FUNCTION__);
        down_read(&current->mm->mmap_sem);
        vma = find_vma(current->mm, arg);
        if (!vma)
           return -EIO;
        if(vma)
        {
          res = zap_vma_ptes(vma, vma->vm_start, vma->vm_end - vma->vm_start);
        }
        pr_info("#########  after zap_vma_ptes( %d ) ############# \n", res);
        printk_pagetable(arg);
        up_read(&current->mm->mmap_sem);
        return (0);
}

static ssize_t  sample_write(struct file *file, const char __user *buf, size_t count, loff_t *off)
{
        int     res = 0;
        struct page *pages[1];
        struct  page *page;
        unsigned long pfn = 0;
        unsigned long arg = (unsigned long)buf;
        struct vm_area_struct *vma = NULL;
        //unsigned long pfn;
        printk(KERN_INFO "%s\n", __FUNCTION__);
        down_read(&current->mm->mmap_sem);
        vma = find_vma(current->mm, arg);
        if (!vma)
           return -EIO;

#if 1
    if (follow_pfn(vma, arg, &pfn))
    {
        //vma->vm_flags |= VM_IO | VM_DONTCOPY | VM_DONTEXPAND | VM_NORESERVE |
	//			VM_DONTDUMP | VM_PFNMAP;
        vma->vm_flags |= VM_PFNMAP;
        printk(KERN_INFO "no page for vma addr  %lu \n",arg);
        handle_mm_fault(vma, arg, FAULT_FLAG_WRITE);
      
    }
    printk_pagetable(arg);
#if 0
    res = get_user_pages_remote(current, current->mm,
				arg , 1, 0,  pages, &vma,NULL);
    pr_info("#########  after get_user_pages_remote ############# \n");
    printk_pagetable(arg);

    page = pages[0];
    if (res < 1) {
        printk(KERN_INFO "GUP error: %d\n", res);
        free_page((unsigned long) page);
        return -EFAULT;
    }
#endif
#else
        res = get_user_pages(
                arg ,
                1,
                1,
                &page,
                NULL);
#endif
#if 0
        // no get_user_pages_remote option
        if (res) {
                printk(KERN_INFO "Got mmaped.\n");
                myaddr = kmap(page);
                printk(KERN_INFO "%s\n", myaddr);
                strcpy(myaddr, "Mohan");
                //page_cache_release(page);
                put_page(page);
        }
#endif
        up_read(&current->mm->mmap_sem);
        return (0);
}
static struct   file_operations sample_ops = {
        .owner  = THIS_MODULE,
        .open   = sample_open,
        .release = sample_release,
        .write  = sample_write,
        .read = sample_read
};
static int __init sample_init(void)
{
        int ret;
        ret = register_chrdev(42, "Sample", &sample_ops);
        sample_class = class_create(THIS_MODULE, "Sample");
        device_create(sample_class, NULL, MKDEV(42, 0), NULL, "Sample");
        return (ret);
}
static void __exit sample_exit(void)
{
        device_destroy(sample_class, MKDEV(42, 0));
        class_destroy(sample_class);
        unregister_chrdev(42, "Sample");
}
module_init(sample_init);
module_exit(sample_exit);
MODULE_LICENSE("GPL");
