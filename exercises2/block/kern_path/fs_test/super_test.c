#include <linux/init.h>
#include <linux/module.h>
#include <linux/namei.h>
#include <linux/mount.h>
#include <linux/fs.h>
//#include "ext4.h"

static int __init super_init(void)
{
	struct path p;
	struct dentry *d;
	struct inode *ino;
	struct super_block *sb;
	struct ext4_sb_info *esi;

	int err = kern_path("/bin/bash", LOOKUP_FOLLOW, &p);
	if (err) {
		pr_err("kern_path failed\n");
		return err;
	}

	d = p.dentry;

	ino = d->d_inode;
	pr_info("i_ino: %ld\n", ino->i_ino);

	sb = ino->i_sb;
	pr_info("s_blocksize: %ld\n", sb->s_blocksize);
	pr_info("s_dev: %d\n", sb->s_dev);
	pr_info("s_type: %s\n", sb->s_type->name);
	pr_info("s_type: %lx\n", sb->s_magic);

	esi = sb->s_fs_info;
	//pr_info("s_decs_size: %ld\n", esi->s_decs_size);

	return 0;
}

static void __exit super_exit(void)
{
}

module_init(super_init);
module_exit(super_exit);
MODULE_LICENSE("GPL");