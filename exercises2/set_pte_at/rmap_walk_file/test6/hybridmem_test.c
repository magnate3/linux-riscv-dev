#include <linux/fs.h>
#include <linux/sched.h>
#include <linux/module.h>
#include <linux/sched/mm.h>
#include <linux/mm.h>

#include <linux/list.h>
#include <linux/slab.h>
#include <linux/pagemap.h>
#include <linux/interval_tree.h> // interval_tree_iter_first
#include <linux/interval_tree_generic.h> // INTERVAL_TREE_DEFINE
#include "common.h"
#define MODULE_MAJOR            224
#define MODULE_NAME             "hybridmem"
struct xarray                   arr;
MODULE_LICENSE("Dual BSD/GPL");
struct my_page
{

        struct page* page;
	struct list_head list;	
	     	
	     	
};
static LIST_HEAD(page_list);
//static unsigned long vaddr2paddr(struct mm_struct *mm , unsigned long addr);
static  struct page * vaddr2paddr(struct mm_struct *mm , unsigned long addr);
static vm_fault_t vma_fault(struct vm_fault* vmf)
{
    struct page* start_page = alloc_page(GFP_HIGHUSER_MOVABLE | __GFP_ZERO);
    if(!start_page)
        return VM_FAULT_SIGBUS;
    struct vm_area_struct *vma = vmf->vma;
    unsigned long phy_addr = (page_to_pfn(start_page)<< PAGE_SHIFT) + vma->vm_pgoff;
    struct my_page * my_page = kmalloc(sizeof(struct my_page),GFP_KERNEL|GFP_ATOMIC);
    vmf->page = start_page;
    printk(KERN_INFO "alloc page frame struct is @ %p", start_page);
    //printk("fault, is_write = %d, vaddr = %lx, paddr = %lx\n", vmf->flags & FAULT_FLAG_WRITE, vmf->address, (size_t)virt_to_phys(page_address(page)));
    printk("fault, is_write = %d, vaddr = %lx, paddr = %lx\n", vmf->flags & FAULT_FLAG_WRITE, vmf->address, phy_addr);
    my_page->page = start_page;
    list_add_tail(&my_page->list, &page_list);
    return 0;
}

static struct vm_operations_struct vma_ops =
{
    .fault = vma_fault,
};

static int mmap(struct file* file, struct vm_area_struct* vma)
{
    printk("mmap, start = %lx, end = %lx\n", vma->vm_start, vma->vm_end);
    vma->vm_ops = &vma_ops;
    return 0;
}

static int my_open(struct inode *inode, struct file *file)
{
        struct mm_struct *mm = get_task_mm(current);

        file->private_data = mm;
        return 0;
}
static inline pgoff_t test_page_to_pgoff(struct page *page)
{
	// index of page indicates the page index of a vma
		        return page_to_index(page);
}
static inline unsigned long
vma_address(struct page *page, struct vm_area_struct *vma)
{
	pgoff_t pgoff;
	unsigned long address;

	VM_BUG_ON_PAGE(PageKsm(page), page);	/* KSM page->index unusable */
	pgoff = test_page_to_pgoff(page);
	//vm_pgoff 这个vma在文件内的offset(页为单位的)
	if (pgoff >= vma->vm_pgoff) {
		address = vma->vm_start +
			((pgoff - vma->vm_pgoff) << PAGE_SHIFT);
		/* Check for address beyond vma (or wrapped through 0?) */
		if (address < vma->vm_start || address >= vma->vm_end)
			address = -EFAULT;
	} else if (PageHead(page) &&
		   pgoff + compound_nr(page) - 1 >= vma->vm_pgoff) {
		/* Test above avoids possibility of wrap to 0 on 32-bit */
		address = vma->vm_start;
	} else {
		address = -EFAULT;
	}
	return address;
}
#if 1
static inline unsigned long vma_start_pgoff(struct vm_area_struct *v)
{
	        return v->vm_pgoff;
}

static inline unsigned long vma_last_pgoff(struct vm_area_struct *v)
{
	        return v->vm_pgoff + vma_pages(v) - 1;
}
INTERVAL_TREE_DEFINE(struct vm_area_struct, shared.rb, unsigned long, shared.rb_subtree_last, vma_start_pgoff, vma_last_pgoff, /* empty */, vma_interval_tree)
#define vma_interval_tree_foreach(vma, root, start, last)               \
       for (vma = vma_interval_tree_iter_first(root, start, last);     \
       vma; vma = vma_interval_tree_iter_next(vma, start, last))
#endif
static int vma_interval_tree_traverse(struct address_space *mapping,struct page *oldpage,  pgoff_t pgoff_start, pgoff_t pgoff_end)
{
   unsigned long address;
   struct vm_area_struct *vma ;
       vma_interval_tree_foreach(vma,&(mapping->i_mmap),pgoff_start, pgoff_end)
       {
	       if(NULL != vma)
	       {
	            address = vma_address(oldpage, vma);
                    printk(KERN_INFO "page frame struct is @ %p, and user vaddr %lx\n", oldpage,address);
	       }
       }
   return 0;
}
#if 0
static ssize_t my_read(struct file *filp, /* see include/linux/fs.h   */
		       char __user *buffer, /* buffer to fill with data */
		       size_t length, /* length of the buffer     */
		       loff_t *offset)
{
   struct mm_struct *mm = filp->private_data;
   int ans = 0;
   struct address_space *mapping;
   struct page *oldpage = vaddr2paddr(mm,(unsigned long) buffer);
   struct address_space *mapping2 = page_mapping(oldpage);
   unsigned long address;
   struct vm_area_struct *vma ;
   pgoff_t pgoff_start, pgoff_end;
   pgoff_start  =  test_page_to_pgoff(oldpage);
   pgoff_end = pgoff_start + thp_nr_pages(oldpage) - 1;
   printk(KERN_INFO "d_name %s, pgoff_start : %lu, pgoff_end: %lu \n, annon: %d", filp->f_path.dentry->d_name.name,  pgoff_start, pgoff_end, PageAnon(oldpage));
   if (!strncmp(filp->f_path.dentry->d_name.name, FILE_NAME,strlen(FILE_NAME))) {
               ans = 1;
   }
   if (ans)
   {
       mapping = filp->f_mapping;
       printk(KERN_INFO "address_space mapping @ %p, mapping2 %p \n", mapping,mapping2);
       vma_interval_tree_foreach(vma,&(mapping->i_mmap),pgoff_start, pgoff_end)
       {
	       if(NULL != vma)
	       {
	            address = vma_address(oldpage, vma);
                    printk(KERN_INFO "page frame struct is @ %p, and user vaddr %lx\n", oldpage,address);
	       }
       }
   }
	return 0;
}
#else
static ssize_t my_read(struct file *filp, /* see include/linux/fs.h   */
		       char __user *buffer, /* buffer to fill with data */
		       size_t length, /* length of the buffer     */
		       loff_t *offset)
{
   struct mm_struct *mm = filp->private_data;
   struct address_space *mapping, *mapping2;
   struct page *oldpage ;
   struct inode *inode;
   unsigned long address;
   struct vm_area_struct *vma, *vma2;
   const char * name;
   pgoff_t pgoff_start, pgoff_end;
   printk(KERN_INFO"********** %s begin to run \n", __func__);
   oldpage = vaddr2paddr(mm,(unsigned long) buffer);
   if(!oldpage)
   {
               printk(KERN_INFO"page not exist \n");
               return 0;
   }
       pgoff_start  =  test_page_to_pgoff(oldpage);
       pgoff_end = pgoff_start + thp_nr_pages(oldpage) - 1;
       mapping2 = page_mapping(oldpage);
    if (PageAnon(oldpage)) {
               printk(KERN_INFO"anon vma %p, page_mapping %p\n", vma->anon_vma, mapping2);
               return 0;
   }
   vma2 =  find_vma(mm, (unsigned long) buffer); 
   if(NULL == vma2 || NULL == vma2->vm_file)
   {
      printk(KERN_INFO"vma2 or vm_file is null \n");
      return 0;
   }
   inode = file_inode(vma2->vm_file);
   mapping = inode->i_mapping;
   name = vma2->vm_file->f_path.dentry->d_iname;
   printk(KERN_INFO "d_name %s, pgoff_start : %lu, pgoff_end: %lu , annon: %d \n", name,  pgoff_start, pgoff_end, PageAnon(oldpage));
   printk(KERN_INFO "address_space mapping @ %p, mapping2 %p , mapping == mapping2 ? %d \n", mapping,mapping2, mapping == mapping2);
   vma_interval_tree_foreach(vma,&(mapping->i_mmap),pgoff_start, pgoff_end)
   {
	       if(NULL != vma)
	       {
	            address = vma_address(oldpage, vma);
                    printk(KERN_INFO "page frame struct is @ %p, and user vaddr %lx\n", oldpage,address);
	       }
    }
	return 0;
}
#endif
static int my_release(struct inode *inode, struct file *file)
{
	//if(NULL != start_page)
	//{
	//     __free_page(start_page);
	//}
	struct my_page *ptr1,*next;
	list_for_each_entry_safe(ptr1,next,&page_list,list){
	    list_del(&ptr1->list);
	    __free_page(ptr1->page);
	    kfree(ptr1);
	}
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

//static unsigned long vaddr2paddr(struct mm_struct *mm , unsigned long addr)
static  struct page * vaddr2paddr(struct mm_struct *mm , unsigned long addr)
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
        printk(KERN_INFO "page ==  present ? %d and page frame struct is @ %p, and user vaddr %lx, paddr %lx \n",pte_present(pte), page,addr, paddr);
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
    return page;

}
static int init(void)
{
    int ret = register_chrdev(MODULE_MAJOR, MODULE_NAME, &fops);
    if(ret < 0)
    {
        printk("Unable to register device '%s'\n", MODULE_NAME);
        return ret;
    }
    printk("*********** moudle '%s' begin :\n", MODULE_NAME);
    return 0;
}

static void cleanup(void)
{
    unregister_chrdev(MODULE_MAJOR, MODULE_NAME);
}

module_init(init);
module_exit(cleanup);
