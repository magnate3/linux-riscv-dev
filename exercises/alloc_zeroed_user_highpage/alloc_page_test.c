#include <linux/init.h>
#include <linux/module.h>
#include <linux/gfp.h>
#include <linux/mm.h>
#include <asm/uaccess.h>
MODULE_LICENSE("DUAL BSD/GPL");
#define MY_PAGE_SIZE 65536
static	int __init alloc_pages_init(void);
static	void __exit alloc_pages_exit(void);

struct page *pages = NULL;
unsigned long  addr = 0;
int	__init alloc_pages_init(void)
{
        
unsigned long  virt;
addr = __get_free_pages(GFP_KERNEL,get_order(4*MY_PAGE_SIZE));
virt = addr;
pr_info("sizeof(struct page): %lu", sizeof(struct page));
pr_info("virt : 0x%lx, phy:0x%lx, page:0x%p, pfn: %ld\n", virt, (unsigned long)virt_to_phys((void *)virt), virt_to_page((void *)virt),page_to_pfn(virt_to_page((void *)virt)));
virt = addr + MY_PAGE_SIZE;
pr_info("virt : 0x%lx, phy:0x%lx, page:0x%p, pfn: %ld\n", virt, (unsigned long)virt_to_phys((void *)virt), virt_to_page((void *)virt),page_to_pfn(virt_to_page((void *)virt)));
virt = addr + 2*MY_PAGE_SIZE;
pr_info("virt : 0x%lx, phy:0x%lx, page:0x%p, pfn: %ld\n", virt, (unsigned long)virt_to_phys((void *)virt), virt_to_page((void *)virt),page_to_pfn(virt_to_page((void *)virt)));
virt = addr + 3*MY_PAGE_SIZE;
pr_info("virt : 0x%lx, phy:0x%lx, page:0x%p, pfn: %ld\n", virt, (unsigned long)virt_to_phys((void *)virt), virt_to_page((void *)virt),page_to_pfn(virt_to_page((void *)virt)));
return 0;
}

void	__exit alloc_pages_exit(void)
{
	if(addr)
	{
		free_pages((unsigned long)addr, get_order(4*MY_PAGE_SIZE));
		printk(KERN_ALERT "free_pages ok!\n");
	}
	printk(KERN_ALERT "exit\n");
}

module_init(alloc_pages_init);
module_exit(alloc_pages_exit);
/***********************************************************************************************
mm/memory.c:93:struct page *mem_map;
arch/arm64/include/asm/memory.h:252:#define page_to_phys(page)  (__pfn_to_phys(page_to_pfn(page)))
arch/arm64/include/asm/pgtable.h:363:#define mk_pmd(page,prot)  pfn_pmd(page_to_pfn(page),prot)
arch/arm64/include/asm/pgtable.h:447:#define mk_pte(page,prot)  pfn_pte(page_to_pfn(page),prot)
include/linux/mm.h:80:#define page_to_virt(x)   __va(PFN_PHYS(page_to_pfn(x)))
include/linux/mm.h:131:#define nth_page(page,n) pfn_to_page(page_to_pfn((page)) + (n))
include/linux/mmzone.h:93:      get_pfnblock_flags_mask(page, page_to_pfn(page),                \
include/linux/pageblock-flags.h:81:     get_pfnblock_flags_mask(page, page_to_pfn(page),                \
include/linux/pageblock-flags.h:85:     set_pfnblock_flags_mask(page, flags, page_to_pfn(page),         \
include/linux/pfn_t.h:72:static inline pfn_t page_to_pfn_t(struct page *page)
include/asm-generic/memory_model.h:34:#define __page_to_pfn(page)       ((unsigned long)((page) - mem_map) + \
include/asm-generic/memory_model.h:44:#define __page_to_pfn(pg)                                         \
include/asm-generic/memory_model.h:55:#define __page_to_pfn(page)       (unsigned long)((page) - vmemmap)
include/asm-generic/memory_model.h:62:#define __page_to_pfn(pg)                                 \
include/asm-generic/memory_model.h:81:#define page_to_pfn __page_to_pfn
*************************************************************************************/
