#include <linux/init.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/fs_struct.h>
#include <linux/path.h>
#include <linux/list.h>
#include <linux/spinlock.h>

static int list_children(struct dentry *parent)
{
	struct dentry *child;

	spin_lock(&parent->d_lock);
	list_for_each_entry(child, &parent->d_subdirs, d_child) {
		spin_lock(&child->d_lock);
		pr_info("d_name: %s:%s", child->d_name.name,
				d_unhashed(child) ? "unhashed\n" : "hashed\n");
		spin_unlock(&child->d_lock);
	}
	spin_unlock(&parent->d_lock);

	return 0;
}

static int __init dcachetest_init(void)
{
	struct dentry *dentry = current->fs->root.dentry;

	list_children(dentry);
	return 0;
}

static void __exit dcachetest_exit(void)
{
}

module_init(dcachetest_init);
module_exit(dcachetest_exit);