#include <linux/init.h>
#include <linux/module.h>
#include <linux/namei.h>
#include <linux/mount.h>
#include <linux/fs.h>
#include <linux/mm_types.h>
#include <linux/mm.h>

struct page * show_page(struct radix_tree_root *rtr, int num)
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
        return pg;
}
//struct file *file = vma->vm_file
#define TEST_FILE_NAME "/root/programming/kernel/fs_test/test.txt"
static int __init path_test_init(void)
{
	struct path p;
	struct dentry *d;
	struct inode *ino;
	struct address_space *as;
	struct radix_tree_root *rtr;
	struct page *page = NULL;
        int index = 0;
        int err;
        struct file * filp =filp_open(TEST_FILE_NAME,O_RDWR,0644);
        char * addr = NULL;
	if (!filp) {
		printk(KERN_ALERT "filp open failed\n");
		return err;
	}
	err = kern_path(TEST_FILE_NAME, LOOKUP_FOLLOW, &p);
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
             page = show_page(rtr, index++);
             if(page)
             {
                  err = as->a_ops->readpage(filp, page);
                  addr = page_address(page);
	          printk(KERN_ALERT "content : %s\n", addr);
             }
        } 
        if(filp)
        {
             filp_close(filp,NULL);
        }
	return 0;
}

static void __exit path_test_exit(void)
{
}

module_init(path_test_init);
module_exit(path_test_exit);
