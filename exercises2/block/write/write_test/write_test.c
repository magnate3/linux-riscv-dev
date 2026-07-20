#include <linux/init.h>
#include <linux/module.h>
#include <linux/namei.h>
#include <linux/mount.h>
#include <linux/fs.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/uio.h>
#include <linux/writeback.h>
#include <asm/cacheflush.h>

#define TEST_STRING "test file write"
static ssize_t new_sync_write_test(struct iov_iter *iter, char __user *buf, size_t len, loff_t *ppos)
{
       struct iovec iov = { .iov_base = buf, .iov_len = len };
       ssize_t ret=0;
       iov_iter_init(iter, WRITE, &iov, 1, len);
       return ret;
}
size_t iov_iter_copy_from_user_atomic_test(struct page *page, struct iov_iter *i, unsigned long offset, size_t bytes)
{
#if 0
     char *kaddr = kmap_atomic(page),  *p = kaddr + offset;
     memcpy(p, i->iov->iov_base, i->iov->iov_len);
     kunmap_atomic(kaddr);
#else
     char * addr = page_address(page);
     memcpy(addr, TEST_STRING, strlen(TEST_STRING));
#endif
     return bytes;
}
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
#define TEST_FILE_NAME "/work/kernel_learn/fs_test/test.txt"
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
        loff_t pos = 0;
        unsigned int flags = 0;
        unsigned long bytes = 0;
        void *fsdata;
        unsigned long offset; 
	struct iov_iter iter;
	size_t copied;
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

	//rtr = &as->page_tree;
	//printk(KERN_ALERT "page_tree height: %d\n", rtr->height);
#if 0
        while(index  < as->nrpages)
        {
             page = show_page(rtr, index++);
             if(page)
             {
                  addr = page_address(page);
                  strncpy(addr,TEST_STRING,strlen(TEST_STRING));
                  bytes = strlen(TEST_STRING);
                  err = as->a_ops->write_begin(filp, as, pos, bytes, flags,
                                               &page, &fsdata);
               if (unlikely(err< 0))
                       break; 
	          //printk(KERN_ALERT "content : %s\n", addr);
             }
        } 
#else
	       printk(KERN_ALERT "*********** test file write begin\n");
               bytes = strlen(TEST_STRING);
	       new_sync_write_test(&iter,TEST_STRING, bytes,&pos);
               offset = (pos & (PAGE_SIZE - 1));
               //bytes = min_t(unsigned long, PAGE_SIZE - offset, bytes);
               bytes = min_t(unsigned long, PAGE_SIZE - offset, iov_iter_count(&iter));
	       if (unlikely(iov_iter_fault_in_readable(&iter, bytes))) {
	               printk(KERN_ALERT "iov iter fault err \n");
	               goto err1;
	       }
	       printk(KERN_ALERT "after iov iter fault nrpages: %ld\n", as->nrpages);
               err = as->a_ops->write_begin(filp, as, pos, bytes, flags, &page, &fsdata);
	       if (unlikely(err < 0)){
	                     goto err1;
	       }
	       printk(KERN_ALERT "after write begin nrpages: %ld\n", as->nrpages);
               if(page)
	       {
	       	    addr = page_address(page);
	       	    printk(KERN_ALERT "old page content : %s\n", addr);
	       }
	       if (mapping_writably_mapped(as))
	       {
	               flush_dcache_page(page);
	       }
	       //copied = iov_iter_copy_from_user_atomic(page, &iter, offset, bytes);
	       copied = iov_iter_copy_from_user_atomic_test(page, &iter, offset, bytes);
	       printk(KERN_ALERT "copied bytes : %ld\n", copied);
	       flush_dcache_page(page);
               if(page)
	       {
	       	    addr = page_address(page);
	       	    printk(KERN_ALERT "new page content : %s\n", addr);
	       }
	       err = as->a_ops->write_end(filp, as, pos, bytes, copied, page, fsdata);
	       if (unlikely(err< 0)){
	                     goto err1;
	       }
	       printk(KERN_ALERT "write bytes : %lu\n", bytes);
	       balance_dirty_pages_ratelimited(as);
#endif
err1:
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
MODULE_LICENSE("GPL v2");
