#include <linux/init.h>
#include <linux/module.h>
#include <linux/gfp.h>
#include <linux/mm.h>
//#include <asm/mem_map.h> //  mem_map

extern struct page *mem_map;
MODULE_LICENSE("DUAL BSD/GPL");

static	int __init alloc_pages_init(void);
static	void __exit alloc_pages_exit(void);

struct page *pages = NULL;
int	__init alloc_pages_init(void)
{
	pages = alloc_pages(GFP_KERNEL, 3);
	if(!pages)
		return -ENOMEM;
	else
	{
		printk(KERN_ALERT "alloc_pages Successfully!\n");
		printk(KERN_ALERT "pages=0x%lx\n", (unsigned long)pages);
		printk(KERN_ALERT "size of 'pages'=%ld\n", sizeof pages);
		printk(KERN_ALERT "pages=0x%lx\n", pages);
		printk(KERN_ALERT "mem_map = 0x%lx\n",(unsigned long)mem_map);
		printk(KERN_ALERT "size of 'mem_map'=%ld\n", sizeof mem_map);
		printk(KERN_ALERT "mem_map = 0x%lx\n",mem_map);
		printk(KERN_ALERT "pages - mem_map = 0x%lx\n",(unsigned long)pages - (unsigned long)mem_map);
		printk(KERN_ALERT "pages - mem_map = 0x%lx\n",(unsigned long)(pages - mem_map));
		//the physical address of the head of the pages
		printk(KERN_ALERT "(pages - mem_map) << 12 = 0x%lx\n",((unsigned long)pages - (unsigned long) mem_map) * 4096);
		printk(KERN_ALERT "(pages - mem_map) << 12 = 0x%lx\n",(unsigned long)(pages - mem_map) * 4096);
		//the kernel logic address of the head of the pages
		printk(KERN_ALERT "page_address(pages) = 0x%lx\n", (unsigned long)page_address(pages));
	}
	return 0;
}

void	__exit alloc_pages_exit(void)
{
	if(pages)
	{
		__free_pages(pages, 3);
		printk(KERN_ALERT "__free_pages ok!\n");
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
