/*
 * SO2 Lab - Filesystem drivers
 * Exercise #1 (no-dev filesystem)
 */

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/pagemap.h>
#include <linux/uio.h>
#include <linux/mpage.h>
#include <linux/backing-dev.h>


MODULE_DESCRIPTION("Simple no-dev filesystem");
MODULE_AUTHOR("SO2");
MODULE_LICENSE("GPL");

#define MYFS_BLOCKSIZE		4096
#define MYFS_BLOCKSIZE_BITS	12
#define MYFS_MAGIC		0xbeefcafe
#define LOG_LEVEL		KERN_ALERT

/* declarations of functions that are part of operation structures */

static int myfs_mknod(struct inode *dir,
		struct dentry *dentry, umode_t mode, dev_t dev);
static int myfs_create(struct inode *dir, struct dentry *dentry,
		umode_t mode, bool excl);
static int myfs_mkdir(struct inode *dir, struct dentry *dentry, umode_t mode);

static int test_generic_file_mmap(struct file * file, struct vm_area_struct * vma);
static  int test_simple_readpage(struct file *file, struct page *page);
static ssize_t new_sync_read(struct file *filp, char __user *buf, size_t len, loff_t *ppos);
int test_generic_file_fsync(struct file *file, loff_t start, loff_t end,
                       int datasync);
static int test_null_writepage(struct page *page, struct writeback_control *wbc);
static int test_null_writepages(struct address_space *mapping, struct writeback_control *wbc);
/* TODO 2/4: define super_operations structure */
static const struct super_operations myfs_ops = {
	.statfs		= simple_statfs,
	.drop_inode	= generic_drop_inode,
};

static const struct inode_operations myfs_dir_inode_operations = {
	/* TODO 5/8: Fill dir inode operations structure. */
	.create         = myfs_create,
	.lookup         = simple_lookup,
	.link           = simple_link,
	.unlink         = simple_unlink,
	.mkdir          = myfs_mkdir,
	.rmdir          = simple_rmdir,
	.mknod          = myfs_mknod,
	.rename         = simple_rename,
};


static const struct file_operations myfs_file_operations = {
	/* TODO 6/4: Fill file operations structure. */
        .read		= new_sync_read,
	.read_iter      = generic_file_read_iter,
	.write_iter     = generic_file_write_iter,
	.mmap           = test_generic_file_mmap,
	//.mmap           = generic_file_mmap,
	.llseek         = generic_file_llseek,
        //.fsync          = generic_file_fsync,
        .fsync          = test_generic_file_fsync,
};

static const struct inode_operations myfs_file_inode_operations = {
	/* TODO 6/1: Fill file inode operations structure. */
	.getattr        = simple_getattr,
};

static const struct address_space_operations myfs_aops = {
	/* TODO 6/3: Fill address space operations structure. */
	.readpage       = test_simple_readpage,
	//.readpage       = simple_readpage,
        .writepages     = test_null_writepages,
	.writepage      = test_null_writepage,
	.write_begin    = simple_write_begin,
	.write_end      = simple_write_end,
};
static int test_null_writepages(struct address_space *mapping, struct writeback_control *wbc)
{

        dump_stack();
        return mpage_writepages(mapping, wbc, NULL);
        //return mpage_writepages(mapping, wbc, ext2_get_block);
}
static int test_null_writepage(struct page *page, struct writeback_control *wbc)
{

        //struct inode *inode = page->mapping->host;
        struct address_space *mapping = page->mapping;
        //dump_stack();
        return mpage_writepages(mapping, wbc, NULL);
        //return mpage_writepages(mapping, wbc, ext2_get_block);
}
int test_generic_file_fsync(struct file *file, loff_t start, loff_t end,
                       int datasync)
{
#if 0
        struct inode *inode = file->f_mapping->host;
        int err;

        err = __generic_file_fsync(file, start, end, datasync);
        if (err)
                return err;
        return blkdev_issue_flush(inode->i_sb->s_bdev, GFP_KERNEL, NULL);
#else
        int err;
        struct inode *inode = file_inode(file);
        //struct address_space *mapping = file->f_mapping;
#if 0
        // will cause coredump
        struct backing_dev_info *bdi = inode_to_bdi(inode);
        bdi->capabilities &= (!BDI_CAP_NO_WRITEBACK);
#endif
        pr_info("%s \n",__func__);
        inode_lock(inode); 
        
        err = filemap_write_and_wait_range(file->f_mapping, start, end);
        err = invalidate_inode_pages2_range(file->f_mapping, start >> PAGE_SHIFT, end >> PAGE_SHIFT);
        inode_unlock(inode); 
        return err;
#endif
}
static ssize_t new_sync_read(struct file *filp, char __user *buf, size_t len, loff_t *ppos)
{
        struct iovec iov = { .iov_base = buf, .iov_len = len };
        struct kiocb kiocb;
        struct iov_iter iter;
        ssize_t ret;

        //pr_info("%s \n",__func__);
        init_sync_kiocb(&kiocb, filp);
        kiocb.ki_pos = *ppos;
        iov_iter_init(&iter, READ, &iov, 1, len);

        ret = call_read_iter(filp, &kiocb, &iter);
        BUG_ON(ret == -EIOCBQUEUED);
        *ppos = kiocb.ki_pos;
        return ret;
}

static  int test_simple_readpage(struct file *file, struct page *page)
{
    //dump_stack();
    //pr_info("%s \n",__func__);
    return simple_readpage(file,page);
}
int test_generic_file_mmap(struct file * file, struct vm_area_struct * vma)
{
    //dump_stack();
    //struct address_space *mapping = file->f_mapping;
    int ret = 0;
    ret = generic_file_mmap(file,vma);
    if(vma->vm_ops)
    {
         pr_info(" vma->vm_ops : %p \n", vma->vm_ops); 
    }
    return ret;
}

struct inode *myfs_get_inode(struct super_block *sb, const struct inode *dir,
		int mode)
{
	struct inode *inode = new_inode(sb);

	if (!inode)
		return NULL;

	/* TODO 3/3: fill inode structure
	 *     - mode
	 *     - uid
	 *     - gid
	 *     - atime,ctime,mtime
	 *     - ino
	 */
	inode_init_owner(inode, dir, mode);
	inode->i_atime = inode->i_mtime = inode->i_ctime = current_time(inode);
	inode->i_ino = 1;

	/* TODO 5/1: Init i_ino using get_next_ino */
	inode->i_ino = get_next_ino();

	/* TODO 6/1: Initialize address space operations. */
	inode->i_mapping->a_ops = &myfs_aops;

	if (S_ISDIR(mode)) {
		/* TODO 3/2: set inode operations for dir inodes. */
		inode->i_op = &simple_dir_inode_operations;
		inode->i_fop = &simple_dir_operations;

		/* TODO 5/1: use myfs_dir_inode_operations for inode
		 * operations (i_op).
		 */
		inode->i_op = &myfs_dir_inode_operations;

		/* TODO 3/1: directory inodes start off with i_nlink == 2 (for "." entry).
		 * Directory link count should be incremented (use inc_nlink).
		 */
		inc_nlink(inode);
	}

	/* TODO 6/4: Set file inode and file operations for regular files
	 * (use the S_ISREG macro).
	 */
	if (S_ISREG(mode)) {
		inode->i_op = &myfs_file_inode_operations;
		inode->i_fop = &myfs_file_operations;
	}

	return inode;
}

/* TODO 5/33: Implement myfs_mknod, myfs_create, myfs_mkdir. */
static int myfs_mknod(struct inode *dir,
		struct dentry *dentry, umode_t mode, dev_t dev)
{
	struct inode *inode = myfs_get_inode(dir->i_sb, dir, mode);

	if (inode == NULL)
		return -ENOSPC;

	d_instantiate(dentry, inode);
	dget(dentry);
	dir->i_mtime = dir->i_ctime = current_time(inode);

	return 0;
}

static int myfs_create(struct inode *dir, struct dentry *dentry,
		umode_t mode, bool excl)
{
	return myfs_mknod(dir, dentry, mode | S_IFREG, 0);
}

static int myfs_mkdir(struct inode *dir, struct dentry *dentry, umode_t mode)
{
	int ret;

	ret = myfs_mknod(dir, dentry, mode | S_IFDIR, 0);
	if (ret != 0)
		return ret;

	inc_nlink(dir);

	return 0;
}

static int myfs_fill_super(struct super_block *sb, void *data, int silent)
{
	struct inode *root_inode;
	struct dentry *root_dentry;

	/* TODO 2/5: fill super_block
	 *   - blocksize, blocksize_bits
	 *   - magic
	 *   - super operations
	 *   - maxbytes
	 */
	sb->s_maxbytes = MAX_LFS_FILESIZE;
	sb->s_blocksize = MYFS_BLOCKSIZE;
	sb->s_blocksize_bits = MYFS_BLOCKSIZE_BITS;
	sb->s_magic = MYFS_MAGIC;
	sb->s_op = &myfs_ops;

	/* mode = directory & access rights (755) */
	root_inode = myfs_get_inode(sb, NULL,
			S_IFDIR | S_IRWXU | S_IRGRP |
			S_IXGRP | S_IROTH | S_IXOTH);

	printk(LOG_LEVEL "root inode has %d link(s)\n", root_inode->i_nlink);

	if (!root_inode)
		return -ENOMEM;

	root_dentry = d_make_root(root_inode);
	if (!root_dentry)
		goto out_no_root;
	sb->s_root = root_dentry;

	return 0;

out_no_root:
	iput(root_inode);
	return -ENOMEM;
}

static struct dentry *myfs_mount(struct file_system_type *fs_type,
		int flags, const char *dev_name, void *data)
{
	/* TODO 1/1: call superblock mount function */
	return mount_nodev(fs_type, flags, data, myfs_fill_super);
}

/* TODO 1/6: define file_system_type structure */
static struct file_system_type myfs_fs_type = {
	.owner		= THIS_MODULE,
	.name		= "myfs",
	.mount		= myfs_mount,
	.kill_sb	= kill_litter_super,
};

static int __init myfs_init(void)
{
	int err;

	/* TODO 1/1: register */
	err = register_filesystem(&myfs_fs_type);
	if (err) {
		printk(LOG_LEVEL "register_filesystem failed\n");
		return err;
	}

	return 0;
}

static void __exit myfs_exit(void)
{
	/* TODO 1/1: unregister */
	unregister_filesystem(&myfs_fs_type);
}

module_init(myfs_init);
module_exit(myfs_exit);
