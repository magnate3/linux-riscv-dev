#include <linux/module.h>   
#include <linux/gfp.h> 			 //alloc_pages __GPF_HIGHMEM
#include <linux/mm_types.h>     //struct page

extern void* high_memory;

#include <asm/fixmap.h>   
#include <asm/pgtable.h>   
#include <asm/memory.h>   
#include <linux/highmem.h>   
//#include <asm/highmem.h>   

static int __init kmap_example_init(void)
{
	printk("kmap example init\n");
    unsigned long fixaddr_end,fixaddr_start;
    unsigned long vmalloc_end,vmalloc_start;
    unsigned long lowmem_end,lowmem_start;
    unsigned long pkmap_end,pkmap_start;
    //fixaddr_end = FIXADDR_END;
    fixaddr_start = FIXADDR_START;
    vmalloc_end = VMALLOC_END;
    vmalloc_start = VMALLOC_START;
    lowmem_start = PAGE_OFFSET;
    lowmem_end = (unsigned long)high_memory;
    //pkmap_start = PKMAP_BASE;
    //pkmap_end = PKMAP_BASE + 0x200000;

    printk("***************************************************\n");
    printk("FIXMAP:      %8lx -- %8lx\n", fixaddr_start, fixaddr_end);
    printk("VMALLOC:     %8lx -- %8lx\n", vmalloc_start, vmalloc_end);
    printk("LOWMEM:      %8lx -- %8lx\n", lowmem_start, lowmem_end);
    //printk("PKMAP:       %8lx -- %8lx\n", pkmap_start, pkmap_end);
    printk("***************************************************\n");



    struct page *high_page = NULL;
    unsigned int *KMAP_int = NULL;

    /* Allocate a page from Highmem */
    high_page = alloc_pages(__GFP_HIGHMEM, 0);
    /* Map on pkmap */
    KMAP_int  = kmap(high_page);
    printk("[*]unsigned int *KMAP_int:       Address: %#08x\n",
                               (unsigned int)(unsigned long)KMAP_int);
    if (KMAP_int)
        kunmap(high_page);
    __free_pages(high_page, 0);


	return 0;
}


static void __exit kmap_example_exit(void)
{
	printk("kmap example exit\n");
}

module_init(kmap_example_init);
module_exit(kmap_example_exit);

MODULE_AUTHOR("yeshen");
MODULE_DESCRIPTION("kmap example");
MODULE_LICENSE("GPL");
