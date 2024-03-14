#include <linux/module.h>   
#include <linux/gfp.h>  //alloc_pages __GPF_HIGHMEM
#include <linux/mm_types.h>     //struct page
#include <linux/highmem.h>
#include <linux/smp.h>

extern void* high_memory;

#include <asm/fixmap.h>   
#include <asm/pgtable.h>   
#include <asm/memory.h>   
//#include <asm/highmem.h>   
#include <asm/kmap_types.h>
#include <asm-generic/fixmap.h>

//#define LOWAPI
#define HIGHAPI

#ifdef LOWAPI
static int __init lowapi_atomic_example_init(void)
{
    printk("lowapi_atomic map example init\n");
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
    printk("FIXMAP:      %8lx -- \n", fixaddr_start);
    printk("VMALLOC:     %8lx -- %8lx\n", vmalloc_start, vmalloc_end);
    printk("LOWMEM:      %8lx -- %8lx\n", lowmem_start, lowmem_end);
    //printk("PKMAP:       %8lx -- %8lx\n", pkmap_start, pkmap_end);
    printk("***************************************************\n");


    struct page *high_page;
    int idx, type;
    unsigned long vaddr;

    /* Allocate a physical page */
    high_page = alloc_page(__GFP_HIGHMEM);

    /* Obtain current CPU's FIX_KMAP_BEGIN */
    type = kmap_atomic_idx_push();
    idx  = type + KM_TYPE_NR * smp_processor_id();

    /* Obtain fixmap virtual address by index */
    vaddr = __fix_to_virt(FIX_KMAP_BEGIN + idx);
    /* Associate fixmap virtual address with physical address */
    set_fixmap(idx, page_to_phys(high_page));

    printk("[*]unsignd long vaddr:       Address: %#08x\n",
                               (unsigned int)(unsigned long)vaddr);

    /* Remove associate with fixmap */
    clear_fixmap(idx);
	__free_page(high_page);
    
    return 0;
}


static void __exit lowapi_atomic_example_exit(void)
{
    printk("lowapi atomic map example exit\n");
}

module_init(lowapi_atomic_example_init);
module_exit(lowapi_atomic_example_exit);

MODULE_AUTHOR("yeshen");
MODULE_DESCRIPTION("lowapi atomic map example");
MODULE_LICENSE("GPL");

#else

static int __init highapi_atomic_example_init(void)
{
    printk("highapi_atomic map example init\n");
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
    printk("FIXMAP:      %8lx -- \n", fixaddr_start);
    //printk("FIXMAP:      %8lx -- %8lx\n", fixaddr_start, fixaddr_end);
    printk("VMALLOC:     %8lx -- %8lx\n", vmalloc_start, vmalloc_end);
    printk("LOWMEM:      %8lx -- %8lx\n", lowmem_start, lowmem_end);
    //printk("PKMAP:       %8lx -- %8lx\n", pkmap_start, pkmap_end);
    printk("***************************************************\n");


    struct page *high_page  = NULL;
    unsigned int *KMAP_atomic = NULL;

    /* Allocate a page from Highmem */
    high_page = alloc_pages(__GFP_HIGHMEM, 0);
    /* Map on pkmap */
    KMAP_atomic  = kmap_atomic(high_page);
    printk("[*]unsigned int *KMAP_atomic:       Address: %#08x\n",
                               (unsigned int)(unsigned long)KMAP_atomic);
    if (KMAP_atomic)
        kunmap_atomic(KMAP_atomic);	/* 其实做不做都行 */
    __free_pages(high_page, 0);
    
    return 0;
}


static void __exit highapi_atomic_example_exit(void)
{
    printk("highapi atomic map example exit\n");
}

module_init(highapi_atomic_example_init);
module_exit(highapi_atomic_example_exit);

MODULE_AUTHOR("yeshen");
MODULE_DESCRIPTION("highapi atomic map example");
MODULE_LICENSE("GPL");
#endif
