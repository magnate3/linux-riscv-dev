#include <linux/module.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/init.h>
#undef CONFIG_MEMCG
#include <linux/slub_def.h>
static char *kbuf;
static int size = 21*PAGE_SIZE;
static struct kmem_cache *my_cache;
module_param(size, int, 0644);

static int __init my_init(void)
{
        struct page *mem1_page;
        struct page *page_ptr;
        struct kmem_cache *k_cache_ptr;
	/* create a memory cache */
	if (size > KMALLOC_MAX_SIZE) {
		pr_err
		    (" size=%d is too large; you can't have more than %lu!\n",
		     size, KMALLOC_MAX_SIZE);
		return -1;
	}

	my_cache = kmem_cache_create("mycache", size, 0,
				     SLAB_HWCACHE_ALIGN, NULL);
	if (!my_cache) {
		pr_err("kmem_cache_create failed\n");
		return -ENOMEM;
	}
	pr_info("create mycache correctly\n");

	/* allocate a memory cache object */
	kbuf = kmem_cache_alloc(my_cache, GFP_ATOMIC);
	if (!kbuf) {
		pr_err(" failed to create a cache object\n");
		(void)kmem_cache_destroy(my_cache);
		return -1;
	}
	pr_info(" successfully created a object, kbuf_addr=0x%p\n", kbuf);
        mem1_page = virt_to_head_page(kbuf);
        page_ptr = virt_to_page(kbuf);
	pr_info("page_prt : %p,  mem1_page: %p\n", page_ptr, mem1_page);
        if (PageSlab(mem1_page)) {
			k_cache_ptr = mem1_page->slab_cache; 
			pr_info("[+][kmem_cache]name : %s, size : %x\n", k_cache_ptr->name, k_cache_ptr->size);
	} 
	return 0;
}

static void __exit my_exit(void)
{
	/* destroy a memory cache object */
	kmem_cache_free(my_cache, kbuf);
	pr_info("destroyed a cache object\n");

	/* destroy the memory cache */
	kmem_cache_destroy(my_cache);
	pr_info("destroyed mycache\n");
}

module_init(my_init);
module_exit(my_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Ben ShuShu");
