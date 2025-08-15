/*
	
	kglow - Kernel-level glow for Counter-Strike: Global Offensive.
	Copyright (C) 2017, Emma N. Skye. <emma@skyenet.org>

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with kglow. If not, see <http://www.gnu.org/licenses/>.

*/

#include <linux/highmem.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/sched/signal.h>

static uint size;
static ulong offset;

module_param_named(patch_size, size, uint, 0600);
MODULE_PARM_DESC(patch_size, "Amount of bytes to replace with NOP instruction.");

module_param_named(patch_offset, offset, ulong, 0600);
MODULE_PARM_DESC(patch_offset, "Target patching address relative to 'client_client.so'.");
static char *test_name = "test2";
module_param(test_name, charp, 0000);
MODULE_PARM_DESC(test_name, "Test name");
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Emma N. Skye");
MODULE_DESCRIPTION("A glow hack implemented in kernel space.");

static int kglow_proc_show(struct seq_file* m, void* v) {
	struct task_struct* task;

	for_each_process(task) {
		if (strcmp(task->comm, test_name) == 0) {
			struct vm_area_struct* current_mmap;
			
			for (current_mmap = task->mm->mmap; (current_mmap = current_mmap->vm_next) != NULL;) {
				if (current_mmap->vm_file == NULL)
					continue;

				if (current_mmap->vm_file->f_path.dentry->d_iname) {
                                     pr_info("vm_file->f_path.dentry->d_iname:  %s \n", current_mmap->vm_file->f_path.dentry->d_iname);
                                 }

			}
		}
	}

	return 0;
}

static int kglow_proc_open(struct inode* i, struct file* f) {
	return single_open(f, kglow_proc_show, 0);
}

static const struct file_operations kglow_proc_fops = {
	.owner = THIS_MODULE,
	.open = kglow_proc_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = single_release,
};

static int __init kglow_proc_init(void) {
	proc_create("kglow", 0, 0, &kglow_proc_fops);
	return 0;
}

static void __exit kglow_proc_exit(void) {
	remove_proc_entry("kglow", NULL);
}

module_init(kglow_proc_init);
module_exit(kglow_proc_exit);
