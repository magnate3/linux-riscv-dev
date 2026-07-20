#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/seq_file.h>
#include <linux/sched/mm.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/slab.h>

#ifndef find_task_by_pid
#define find_task_by_pid(nr)	pid_task(find_vpid(nr), PIDTYPE_PID)
#endif

#define BUF_SIZE	1024

static int pid = 1;
module_param(pid, int, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);

static int is_stack(struct vm_area_struct *vma)
{
	return vma->vm_start <= vma->vm_mm->start_stack &&
		vma->vm_end >= vma->vm_mm->start_stack;
}

static void show_map(pid_t pid)
{
	struct task_struct *task = NULL;
	struct mm_struct *mm = NULL;
	struct vm_area_struct *vma = NULL;

	int flags = 0;
	struct file *file = NULL;
	dev_t dev = 0;
	char *fullname;
	char *buf = kmalloc(BUF_SIZE, GFP_KERNEL);
	struct inode *inode;
	int mm_count = 0;

	if ((task = find_task_by_pid(pid)) == NULL) {
		printk(KERN_ERR "find_task_by_pid error \n");
		return;
	}

	if ((mm = get_task_mm(task)) == NULL) {
		printk(KERN_ERR "get_task_mm error \n");
		return;
	}

	down_read(&mm->mmap_sem);
	vma = mm->mmap;
	up_read(&mm->mmap_sem);

	if (vma == NULL) {
		printk(KERN_ERR "get vma error \n");
		return;
	}

	printk("Address                     Mode Offset       dev_t   inode       Mapping/Type\n");
	
	down_read(&mm->mmap_sem);
	for (mm_count = mm->map_count; mm_count > 0; mm_count--) {
		file = vma->vm_file;
		flags = vma->vm_flags;

		// virtual address scope
		printk(KERN_CONT "%08lx - %08lx ", vma->vm_start, vma->vm_end);

		// virtual address privilege
		printk(KERN_CONT "vm_flags: %c%c%c%c ",
				flags & VM_READ ? 'r' : '-',
				flags & VM_WRITE ? 'w' : '-',
				flags & VM_EXEC ? 'x' : '-',
				flags & VM_MAYSHARE ? 's' : 'p');
		
		// virtual address offset
		printk(KERN_CONT "virtual address offset: %012lx ", vma->vm_pgoff << PAGE_SHIFT);

		if (file) {
			inode = file->f_inode;
			dev = inode->i_sb->s_dev;
			printk(KERN_CONT "<MAJOR:MINOR>%02x:%02x   ", MAJOR(dev), MINOR(dev));
			printk(KERN_CONT "<inode no>%08lu   ", inode->i_ino);
			memset(buf, 0, BUF_SIZE);

			fullname = d_path(&file->f_path, buf, BUF_SIZE);
			printk(KERN_CONT "file path: %s", fullname);
		} else {
			if (!vma->vm_mm) {
				printk(KERN_CONT "                    [ vdso ]");
			}

			if (mm) {
				if (vma->vm_start <= mm->brk && vma->vm_end >= mm->start_brk)
					printk(KERN_CONT "                    [ heap ]");
				else if (is_stack(vma))
					printk(KERN_CONT "                    [ stack ]");
			}
			
		}
		printk(KERN_CONT "\n");
		vma = vma->vm_next;
	}
	up_read(&mm->mmap_sem);

	kfree(buf);
}

#define PRINTK_DEC(str, val) \
	printk(str, (val) << (PAGE_SHIFT - 10))

static int __init seq_init(void)
{
	struct task_struct *task = NULL;
	struct mm_struct *mm;
	struct vm_area_struct *vm;

	unsigned long text, lib, swap, anon, file, shmem;
	unsigned long hiwater_vm, total_vm, hiwater_rss, total_rss;

	if ((task = find_task_by_pid(pid)) == NULL) {
		printk(KERN_ERR "find_task_by_pid error \n");
		return -1;
	}

	printk("View Process [%s]\n", task->comm);

	if ((mm = get_task_mm(task)) == NULL) {
		printk(KERN_ERR "get_task_mm error \n");
		return -1;
	}
	
	down_read(&mm->mmap_sem);
	vm = mm->mmap;

	anon = get_mm_counter(mm, MM_ANONPAGES);
	file = get_mm_counter(mm, MM_FILEPAGES);
	shmem = get_mm_counter(mm, MM_SHMEMPAGES);

	hiwater_vm = total_vm = mm->total_vm;
	if (hiwater_vm < mm->hiwater_vm)
		hiwater_vm = mm->hiwater_vm;
	hiwater_rss = total_rss = anon + file + shmem;
	if (hiwater_rss < mm->hiwater_rss)
		hiwater_rss = mm->hiwater_rss;

	text = PAGE_ALIGN(mm->end_code) - (mm->start_code & PAGE_MASK);
	text = min(text, mm->exec_vm << PAGE_SHIFT);
	lib = (mm->exec_vm << PAGE_SHIFT) - text;
	swap = get_mm_counter(mm, MM_SWAPENTS);
#if 0
	PRINTK_DEC("VmPeak : \t%8lu kB\n", hiwater_vm);
	PRINTK_DEC("VmSize : \t%8lu kB\n", total_vm);
	PRINTK_DEC("VmLock : \t%8lu kB\n", mm->locked_vm);
	PRINTK_DEC("VmPin  : \t%8llu kB\n", atomic64_read(&mm->pinned_vm));
	PRINTK_DEC("VmHWM  : \t%8lu kB\n", hiwater_rss);
	PRINTK_DEC("VmRSS  : \t%8lu kB\n", total_rss);
	PRINTK_DEC("RssAnon: \t%8lu kB\n", anon);
	PRINTK_DEC("RssFile: \t%8lu kB\n", file);
	PRINTK_DEC("RssShm : \t%8lu kB\n", shmem);
	PRINTK_DEC("VmData : \t%8lu kB\n", mm->data_vm);
	PRINTK_DEC("VmStck : \t%8lu kB\n", mm->stack_vm);
	printk("VmExec : \t%8lu kB\n", text >> 10);
	printk("VmLib  : \t%8lu kB\n", lib >> 10);
	printk("VmPTE  : \t%8lu kB\n", mm_pgtables_bytes(mm) >> 10);
	PRINTK_DEC("VmSwap : \t%8lu kB\n", swap);
#endif
	up_read(&mm->mmap_sem);
	show_map(pid);

	return 0;
}

static void __exit seq_exit(void)
{}

module_init(seq_init);
module_exit(seq_exit);
MODULE_LICENSE("GPL");
