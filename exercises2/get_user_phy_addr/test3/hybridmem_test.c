#include <linux/fs.h>
#include <linux/sched.h>
#include <linux/module.h>
#include <linux/sched/mm.h>
#include <linux/mm.h>

#define MODULE_MAJOR            224
#define MODULE_NAME             "hybridmem"
#define PAGE_ORDER 2
struct xarray                   arr;
MODULE_LICENSE("Dual BSD/GPL");

struct page* start_page = NULL;
static unsigned long vaddr2paddr(struct mm_struct *mm , unsigned long addr);

static int mmap(struct file* file, struct vm_area_struct* vma)
{
    int ret;
    unsigned long size;
    unsigned long pfn_start, phy_addr;
    void *virt_start;
    struct mm_struct *mm =   file->private_data;
    printk("mmap, start = %lx, end = %lx\n", vma->vm_start, vma->vm_end);
    size = vma->vm_end - vma->vm_start;
    pfn_start = page_to_pfn(start_page) + vma->vm_pgoff;
    virt_start = page_address(start_page) + (vma->vm_pgoff << PAGE_SHIFT);
    phy_addr = (page_to_pfn(start_page)<< PAGE_SHIFT) + vma->vm_pgoff;
    ret = remap_pfn_range(vma, vma->vm_start, pfn_start, size, vma->vm_page_prot);
    printk("vaddr = %lx, paddr = %lx\n", vma->vm_start, phy_addr);
    return 0;
}

static int my_open(struct inode *inode, struct file *file)
{
        struct mm_struct *mm = get_task_mm(current);

        file->private_data = mm;
        return 0;
}
static ssize_t my_read(struct file *filp, /* see include/linux/fs.h   */
		       char __user *buffer, /* buffer to fill with data */
		       size_t length, /* length of the buffer     */
		       loff_t *offset)
{
        struct mm_struct *mm = filp->private_data;
        printk("buffer phy addr %lx, virt addr %lx\n",vaddr2paddr(mm,(long unsigned )buffer),(long unsigned)buffer );
	return 0;
}
static int my_release(struct inode *inode, struct file *file)
{
	return 0;
}
static struct file_operations fops =
{
    .owner = THIS_MODULE,
    .mmap = mmap,
    .open = my_open,
    .read = my_read,
    .release = my_release,
};

static unsigned long vaddr2paddr(struct mm_struct *mm , unsigned long addr)
{
    pgd_t *pgd;
    p4d_t *p4d;
    pte_t *ptep, pte;
    pud_t *pud;
    pmd_t *pmd;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0 ;
    struct page *page = NULL;
    //struct mm_struct *mm = current->mm;

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
        goto out1;
    pte = *ptep;

    page = pte_page(pte);
    if (page)
    {
        //page_addr = pte_val(pte) & PAGE_MASK;
	page_addr = pte_pfn(pte) << PAGE_SHIFT;
        page_offset = addr & ~PAGE_MASK;
        //paddr = page_addr + page_offset;
        paddr = page_addr | page_offset;
        printk(KERN_INFO "page frame struct is @ %p, and user vaddr %lx, paddr %lx", page,addr, paddr);
    }
    else
    {
      goto out1;	
    }

#if 0
    ptep = pte_offset_kernel(pmd, addr);
    if (!ptep)
        goto out;
    pte = *ptep;

    page = pte_page(pte);
    if (page)
    {
        //page_addr = pte_val(pte) & PAGE_MASK;
	page_addr = pte_pfn(pte) << PAGE_SHIFT;
        page_offset = addr & ~PAGE_MASK;
        //paddr = page_addr + page_offset;
        paddr = page_addr | page_offset;
        printk(KERN_INFO "page frame struct is @ %p, and kernel paddr %lx", page, paddr);
    }
#endif
out1:
    pte_unmap(ptep);
 out:
    return paddr;

}
static int init(void)
{
    int ret = register_chrdev(MODULE_MAJOR, MODULE_NAME, &fops);
    if(ret < 0)
    {
        printk("Unable to register device '%s'\n", MODULE_NAME);
        return ret;
    }
    start_page = alloc_page(GFP_HIGHUSER_MOVABLE | __GFP_ZERO);
    printk("*********** moudle '%s' begin :\n", MODULE_NAME);
    return 0;
}

static void cleanup(void)
{
	if(NULL != start_page)
	{
	     __free_page(start_page);
	}
    unregister_chrdev(MODULE_MAJOR, MODULE_NAME);
}

module_init(init);
module_exit(cleanup);
