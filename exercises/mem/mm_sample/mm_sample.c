#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/gfp.h>

static int __init my_mem_init(void)
{
	enum zone_type type;
	int migrate;
	//unsigned long size_8MB = 8 * 1024 * 1024;
	unsigned long size_4MB = 4 * 1024 * 1024;
	unsigned char *p;

	pr_info("[%s] start!\n", __func__);

	pr_info("[%s] PAGE_SIZE = %lu MAX_ORDER = %d\n", __func__, PAGE_SIZE, MAX_ORDER);

	type = gfp_zone(GFP_KERNEL);
	migrate = gfpflags_to_migratetype(GFP_KERNEL);
	pr_info("GFP_KERNEL: zone type is %d, migrate type is %d\n", type, migrate);

	type = gfp_zone(GFP_DMA);
	migrate = gfpflags_to_migratetype(GFP_DMA);
	pr_info("GFP_KERNEL: zone type is %d, migrate type is %d\n", type, migrate);
/*
	p = kmalloc(size_8MB, GFP_KERNEL);
	if (p == NULL) {
		pr_err("[%s] out of memory, 8MB\n", __func__);
	} else {
		pr_err("[%s] kmalloc 8MB successfully\n", __func__);
		kfree(p);
	}
*/
	p = kmalloc(size_4MB, GFP_KERNEL);
	if (p == NULL) {
		pr_err("[%s] out of memory, 4MB\n", __func__);
	} else {
		pr_err("[%s] kmalloc 4MB successfully\n", __func__);
		kfree(p);
	}
	return 0;
}

static void __exit my_mem_exit(void)
{
	pr_info("[%s] exit!\n", __func__);
}

module_init(my_mem_init);
module_exit(my_mem_exit);

MODULE_LICENSE("Dual MPL/GPL");
MODULE_AUTHOR("yanli.qian");
MODULE_DESCRIPTION("memory test code");
