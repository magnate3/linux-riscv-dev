#include <linux/init.h>
#include <linux/module.h>
#include <linux/namei.h>
#include <linux/mount.h>
#include <linux/fs.h>
#include <linux/mm_types.h>

void show_page( struct radix_tree_root *rtr, int num)
{
	struct page *pg = NULL;
	pg = (struct page *)radix_tree_lookup(rtr, num);
        if(pg)
        {
	printk(KERN_ALERT "****** page num index: %d\t", num);
	printk(KERN_ALERT "flags: %ld\t", pg->flags);
	printk(KERN_ALERT "index: %ld\t", pg->index);
	printk(KERN_ALERT "_mapcount: %d\n", pg->_mapcount.counter);
        }
}
static int __init path_test_init(void)
{
	struct path p;
	struct dentry *d;
	struct inode *ino;
	struct address_space *as;
	struct radix_tree_root *rtr;
        int index = 0;

	int err = kern_path("/root/programming/kernel/fs_test/kern_path_test.c", LOOKUP_FOLLOW, &p);
	if (err) {
		printk(KERN_ALERT "kern_path failed\n");
		return err;
	}

	printk(KERN_ALERT "mnt_root: %s\n", p.mnt->mnt_root->d_iname);

	d = p.dentry;
	printk(KERN_ALERT "name: %s\n", d->d_name.name);
	printk(KERN_ALERT "d_iname: %s\n", d->d_iname);

	ino = d->d_inode;
	printk(KERN_ALERT "i_ino: %ld\n", ino->i_ino);

	as = ino->i_mapping;
	printk(KERN_ALERT "host: %ld\n", as->host->i_ino);
	printk(KERN_ALERT "nrpages: %ld\n", as->nrpages);

	rtr = &as->page_tree;
	//printk(KERN_ALERT "page_tree height: %d\n", rtr->height);
        while(index  < as->nrpages)
        {
             show_page(rtr, index++);
        } 
	return 0;
}

static void __exit path_test_exit(void)
{
}

module_init(path_test_init);
module_exit(path_test_exit);
