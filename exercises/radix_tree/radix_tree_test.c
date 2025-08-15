#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/jiffies.h>
#include <linux/vmalloc.h>
#include <linux/delay.h>
#include <linux/list.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/blkdev.h>

static struct radix_tree_root root;

static int __init radix_tree_test_init(void)
{	
	int key;
	void *pointer;
	void **slot_pointer;
	int first_index;
	int max_items;	
	int ret_number;
	void **results;
	void ***slot_results;	

	printk(KERN_ALERT "Now start to init a radix tree...\n");
	INIT_RADIX_TREE(&root, GFP_KERNEL);
	printk(KERN_ALERT "This radix tree has been inited succeed!\n");

	printk(KERN_ALERT "Now start to insert 5 nodes into this radix tree, hope it works~~~\n");
	for(key=0; key<5; key++)
	{
		pointer = (void *) __get_free_page(GFP_KERNEL);
		if(!pointer)
		{
			printk(KERN_EMERG "get free page failed!\n");
			return -1;
		}
		memcpy(pointer, "I love you, my cat: HXY.", sizeof("I love you, my cat: HXY."));
		if(radix_tree_insert(&root, key, pointer))
		{
			printk(KERN_EMERG "radix tree insert failed!\n");
			return -2;
		}
	}
	printk(KERN_ALERT "radix tree insert succeed!\n");

	printk(KERN_ALERT "now try to look up, which means: key=>value.\n");
	for(key=0; key<5; key++)
	{
		pointer = radix_tree_lookup(&root, key);
		printk(KERN_ALERT "This radix tree[%d]'s value is: %s\n", key, (char*)pointer);
		printk(KERN_ALERT "look up succeed!\n");
		
		slot_pointer = radix_tree_lookup_slot(&root, key);
		printk(KERN_ALERT "This radix tree[%d]'s value is: %s\n", key, (char*) *slot_pointer);
		printk(KERN_ALERT "look up slot succeed!\n");	
	}
	
	printk(KERN_ALERT "Now try to gang lookup.\n");
	first_index = 2;
	max_items = 2;
	results = kmalloc(1024, GFP_KERNEL);
	ret_number = radix_tree_gang_lookup(&root, results, first_index, max_items);
	printk(KERN_ALERT "the returned number is %d.\n", ret_number);
	for(key=first_index; key<first_index+max_items; key++)
	{	
		printk(KERN_ALERT "This radix tree[%d]'s value is: %s\n", key, (char*) *results);
	}
	
	printk(KERN_ALERT "Now try to gang lookup slot.\n");
	first_index = 1;
	max_items = 3;
	slot_results = kmalloc(1024, GFP_KERNEL);
	ret_number = radix_tree_gang_lookup_slot(&root, slot_results, NULL, first_index, max_items);
	printk(KERN_ALERT "the returned slot number is %d.\n", ret_number);
	for(key=first_index; key<first_index+max_items; key++)
	{
		printk(KERN_ALERT "This radix tree[%d]'s value is: %s\n", key, (char*) **slot_results);
	}

	printk(KERN_ALERT "now try to set radix_tree[2]'s tag is dirty.\n");
	radix_tree_tag_set(&root, 2, PAGECACHE_TAG_DIRTY);
	printk(KERN_ALERT "set over, now try to test it using radix_tree_tag_get(), should return 1.\n");
	printk(KERN_ALERT "radix_tree[2]'s tag is set to dirty, because the return of radix_tree_tag_get() is %d.\n", radix_tree_tag_get(&root, 2, PAGECACHE_TAG_DIRTY));
	printk(KERN_ALERT "now try to clear this tag.\n");
	radix_tree_tag_clear(&root, 2, PAGECACHE_TAG_DIRTY);
	printk(KERN_ALERT "clear over, now radix_tree[2]'s tag is clear, because the return of radix_tree_get() is %d.\n", radix_tree_tag_get(&root, 2, PAGECACHE_TAG_DIRTY));	
	return 0;
}

static void __exit radix_tree_test_exit(void)
{	
	int key;
	void *pointer;	
	printk(KERN_ALERT "now start to delete this radix tree and exit! Stay focus, boy!\n");
	for(key=1; key<5; key++)
	{
		pointer = radix_tree_lookup(&root, key);
		radix_tree_delete(&root, key);
//		free_page(pointer);
	}
	printk(KERN_ALERT "delete this radix tree succeed!\n");
	printk(KERN_ALERT "radix_tree_test_exit succeed!\n");
}

module_init(radix_tree_test_init);
module_exit(radix_tree_test_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Zhao Shulin");
MODULE_DESCRIPTION("This is a simple test file for radix_tree.");