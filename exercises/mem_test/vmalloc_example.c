#include <linux/module.h>   
#include <linux/gfp.h>  //alloc_pages __GPF_HIGHMEM
#include <linux/mm_types.h>     //struct page
#include <linux/vmalloc.h>     

#include <linux/highmem.h>

#include <asm/page.h>
#include <asm/sections.h>
extern void* high_memory;

#include <asm/fixmap.h>   
#include <asm/pgtable.h>   
#include <asm/memory.h>   
//#include <asm/highmem.h>   

static int __init vmalloc_example_init(void)
{
	printk("vmalloc example init\n");
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
    //printk("FIXMAP:      %8lx -- %8lx\n", fixaddr_start, fixaddr_end);
    printk("VMALLOC:     %8lx -- %8lx\n", vmalloc_start, vmalloc_end);
    printk("LOWMEM:      %8lx -- %8lx\n", lowmem_start, lowmem_end);
    //printk("PKMAP:       %8lx -- %8lx\n", pkmap_start, pkmap_end);
    printk("***************************************************\n");


    unsigned int *VMALLOC_int = NULL;

    VMALLOC_int = (unsigned int *)vmalloc(sizeof(unsigned int));
    printk("[*]unsigned int *VMALLOC_int:    Address: %#08x\n",
                               (unsigned int)(unsigned long)VMALLOC_int);
    if (VMALLOC_int)
        vfree(VMALLOC_int);

	return 0;
}


static void __exit vmalloc_example_exit(void)
{
	printk("vmalloc example exit\n");
}

module_init(vmalloc_example_init);
module_exit(vmalloc_example_exit);

MODULE_AUTHOR("yeshen");
MODULE_DESCRIPTION("vmalloc example");
MODULE_LICENSE("GPL");
