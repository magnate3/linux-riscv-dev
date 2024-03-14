#include <linux/init.h>
#include <linux/module.h>
#include <linux/pagemap.h>

struct page *pages = NULL;

static int __init page_cache_get_init(void)
{
	int ret = 0;
	printk(KERN_INFO "%s\n", __func__);
#if 0
static inline struct page *
alloc_pages(gfp_t gfp_mask, unsigned int order)
#endif
	pages = alloc_pages(GFP_KERNEL, 0);
	if (NULL == pages) {
		ret = -ENOMEM;
		printk(KERN_INFO "alloc_pages error \n");
		goto finish;
	}
#if 1
#define page_cache_get(page)		get_page(page)
//static inline void get_page(struct page *page)
#endif
	printk(KERN_INFO "befor page_cache_get pages->_count :%lu, %d\n", pages->counters, page_count(pages));
	page_cache_get(pages);
	printk(KERN_INFO "after page_cache_get pages->_count :%lu, %d\n", pages->counters, page_count(pages));
#if 1
#define page_cache_release(page)	put_page(page)
//void put_page(struct page *page)
#endif
	page_cache_release(pages);
	printk(KERN_INFO "after page_cache_release pages->_count :%lu, %d\n", pages->counters, page_count(pages));
finish:
	return ret;
}

static void __exit page_cache_get_exit(void)
{
	printk(KERN_INFO "%s\n", __func__);
#if 0
void __free_pages(struct page *page, unsigned int order)
#endif
	if (NULL != pages) {
		__free_pages(pages, 0);
		pages = NULL;
	}
}

module_init(page_cache_get_init);
module_exit(page_cache_get_exit);

MODULE_LICENSE("GPL");
