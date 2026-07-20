#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/errno.h>  /* error codes */
#include <linux/sched.h>
#include <linux/mm.h> /* find_vma */
#include <linux/mm_types.h> /* vm_area_struct */
#define PAGE_NUM 1
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Memory");
MODULE_VERSION("110");
MODULE_AUTHOR("Arshad Hussain <arshad.super@gmail.com>");

/*
 * virtual address do not point directly to RAM.
 * They go through a translation (done via MMU)
 * and via page tables to get the actual physical
 * address.
 *
 * logical address are address which are directly
 * mapped to kernel space. logical address are 
 * virtual address. These are first 896MB.
 * 
 * physical address = logical address - PAGE_OFFSET 
 *
 * low memory are memory for which logical address
 * is present. Or in other words mapping are alread
 * done during boot time.
 *
 * high memory are memory which mapping are not
 * present but need to be created on the fly.
 * See other section memory2 for more details
 *
 * [virtual address/logical address]
 * 	|
 * 	\/
 * [Page tables]
 * 	|
 * 	\/
 * [Physical address]
 *
 * page is smallest unit physical memory that is accessed
 * by kernel and is kept in 'struct page'
 *
 */
/*
 *  * This function returns the order of a free page in the buddy system. In
 *   * general, page_zone(page)->lock must be held by the caller to prevent the
 *    * page from being allocated in parallel and returning garbage as the order.
 *     * If a caller does not hold page_zone(page)->lock, it must guarantee that the
 *      * page cannot be allocated or merged in parallel. Alternatively, it must
 *       * handle invalid values gracefully, and use page_order_unsafe() below.
 *        */
static inline unsigned int page_order(struct page *page)
{
        /* PageBuddy() must be checked by the caller */
        return page_private(page);
}

static inline int page_is_buddy(struct page *page, struct page *buddy,
                                                        unsigned int order)
{
        if (page_is_guard(buddy) && page_order(buddy) == order) {
                if (page_zone_id(page) != page_zone_id(buddy))
                        return 0;

                VM_BUG_ON_PAGE(page_count(buddy) != 0, buddy);

                return 1;
        }

        if (PageBuddy(buddy) && page_order(buddy) == order) {
                /*
 *                  * zone check is done late to avoid uselessly
 *                                   * calculating zone/node ids for pages that could
 *                                                    * never merge.
 *                                                                     */
                if (page_zone_id(page) != page_zone_id(buddy))
                        return 0;

                VM_BUG_ON_PAGE(page_count(buddy) != 0, buddy);

                return 1;
        }
        return 0;
}
static int __init code_init(void)
{
	struct page *p,*p1,*p2;
	unsigned long va; /* virtual address returned */
	unsigned long *page1, *page;
        int pageblock_mt, page_mt;
	/*
	 * Allocate single page. (Physical memory)
	 * Similar to alloc_pages(mask,0);
	 */
	p = alloc_pages(GFP_KERNEL, PAGE_NUM);
	if(!p) {
		printk(KERN_INFO "alloc_page Alocation failed\n");
		return -1;
	} else {
		printk(KERN_INFO "alloc_page Alloation done\n");
		page = (unsigned long *)page_address(p);
	}

	/* 
	 * 2^3 = 8 pages allocation will be done here
	 * Allocate set of pages and return struct page
	 */
	p1 = alloc_pages(GFP_KERNEL, PAGE_NUM);
	if(!p1) {
		printk(KERN_INFO "alloc_pages Alocation failed\n");
		return -1;
	} else {
		printk(KERN_INFO "alloc_pages Alloation done\n");
		/*
		 * macro page_address() - return virtual address that 
		 * corresponds to the start of page
		 */
		page1 = (unsigned long *)page_address(p1);

		/*
		 * convert virtual address back to page
		 */
		p2 = virt_to_page(page);
	}
        
	printk(KERN_INFO "page_is_buddy : %d, PageBuddy(p) %d,  PageBuddy(p1) %d \n", page_is_buddy(p,p1,3), PageBuddy(p),  PageBuddy(p1));
        	/* Print information relevant to grouping pages by mobility */
	//pageblock_mt = get_pageblock_migratetype(page);
	//page_mt  = gfpflags_to_migratetype(page_owner->gfp_mask);
        if (p){
		__free_pages(p,PAGE_NUM);
        }
        if (p1){
		__free_pages(p1,PAGE_NUM);
        }
	/*
	 * Return virtual address.
	 * GFP_HIGHMEM, not to be used with this __get_free_pages
	 * As it is not gurantee to be continous.
	 */
	va = __get_free_pages(GFP_KERNEL, 3);
	if (!va) { 
		printk(KERN_INFO "__get_free_pages Alocation failed\n");
		return -1;
	} else {
		printk(KERN_INFO "__get_free_pages Alloation done\n");
		free_pages(va,3);
	}

		
	return 0;
}

static void __exit code_exit(void) {
	printk(KERN_INFO "Goodbye, World!\n");
}

module_init(code_init);
module_exit(code_exit);
