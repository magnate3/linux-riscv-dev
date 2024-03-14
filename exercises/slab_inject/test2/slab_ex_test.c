#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/slab.h>
#undef CONFIG_MEMCG
#include <linux/slub_def.h>

static int __init show_init(void)
{
	struct page *mem1_page, *mem2_page;
	struct kmem_cache *k_cache_ptr;
	u32 *mem_ptr1, *mem_ptr2;

	mem_ptr1 = kmalloc(124, GFP_KERNEL);
	mem_ptr2 = kmalloc(48, GFP_KERNEL);
	memset(mem_ptr1, 0x78, 124);
	memset(mem_ptr2, 0x56, 48);

	printk("[+] mem_ptr1: %p, mem_ptr2: %p\n", mem_ptr1, mem_ptr2);

	mem1_page = virt_to_head_page(mem_ptr1);
	printk("[+] page: %p \n", mem1_page);
	if (PageSlab(mem1_page)) {
		k_cache_ptr = mem1_page->slab_cache;
		printk("[+][kmem_cache] name: %s, size: %d\n",
				k_cache_ptr->name, k_cache_ptr->size);
	}

	mem2_page = virt_to_head_page(mem_ptr2);
	printk("[+] page: %p \n", mem2_page);
	if (PageSlab(mem2_page)) {
		k_cache_ptr = mem2_page->slab_cache;
		printk("[+][kmem_cache] name: %s, size: %d\n",
				k_cache_ptr->name, k_cache_ptr->size);
	}

	kfree(mem_ptr1);
	kfree(mem_ptr2);

	return 0;
}

static void __exit show_exit(void)
{
}

module_init(show_init);
module_exit(show_exit);
MODULE_LICENSE("GPL");