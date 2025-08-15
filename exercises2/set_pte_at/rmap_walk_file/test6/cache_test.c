#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <asm/uaccess.h>
#include <linux/fdtable.h>
#include <linux/pagemap.h>

#define PFX "mycache: "
#define PROC_NAME "hogehoge"
#ifndef find_task_by_pid
#define find_task_by_pid(nr)    pid_task(find_vpid(nr), PIDTYPE_PID)
#endif

//static size_t proc_write( struct file *filp, const char __user *buff,
//                                   unsigned long len, void *data );


ssize_t write_handler(struct file * filp, const char __user *buff, size_t len, loff_t *offp);
static struct proc_dir_entry *mycache_proc;
static const struct proc_ops  mycache_ops = {
		                          .proc_write = write_handler,
		                            };
static int proc_init_module(void)
{
#if 0
   dirp = (struct proc_dir_entry *)
   create_proc_entry(PROC_NAME, 0666, (struct proc_dir_entry *) 0);
   if (dirp == 0)
       return(-EINVAL);
   dirp->write_proc = (write_proc_t *) proc_write;
 #else
    mycache_proc = proc_create(PROC_NAME, 0666, NULL, &mycache_ops);
    if (mycache_proc == NULL) {
               printk(KERN_ERR PFX "cannot create /proc/mycache\n");
                return -ENOMEM;
    }
 #endif
   return 0;
}

static void proc_cleanup_module(void)
{
   remove_proc_entry(PROC_NAME, (struct proc_dir_entry *) 0);
}
#if 1
#define vma_interval_tree_foreach(vma, root, start, last)               \
       for (vma = vma_interval_tree_iter_first(root, start, last);     \
       vma; vma = vma_interval_tree_iter_next(vma, start, last))
#endif
static int do_page_cache(struct task_struct *process)
{
   int i, ans = 0;
   struct  files_struct *files;
   struct address_space *mapping;
   struct radix_tree_node *node;
   struct page *page;

   files = process->files;
// ファイルaのファイルディスクリプターの取得
   for (i = 0; i < files->fdt->max_fds; i++) {
       if (files->fdt->fd[i]) {
           if (!strcmp(files->fdt->fd[i]->f_path.dentry->d_name.name, "a")) {
               ans = 1;
               break;
           }
       }
   }
   if (ans) {
//       mapping = files->fdt->fd[i]->f_mapping;
//       node = mapping->page_tree.rnode;
//       page = node;
//       printk("%s\n", kmap(page));
//	   kunmap(page);
   }
   return 0;
}

//static size_t proc_write( struct file *filp, const char __user *buff,
//                                   unsigned long len, void *data )
ssize_t
write_handler(struct file * filp, const char __user *buff,
		                              size_t len, loff_t *offp)
{
   char    _buff[64];

   if (copy_from_user(_buff, buff, len )) {
       return -EFAULT;
   }
   int mypid = simple_strtol(_buff, NULL, 0);

   struct task_struct *process;
   process = find_task_by_pid(mypid);
   if (process) { 
       do_page_cache(process);
   }
   else {
       mypid = -1;
       printk("pid error:%d\n", mypid);
       return -EFAULT;
   }
   return len;
}

module_init(proc_init_module);
module_exit(proc_cleanup_module);
MODULE_LICENSE("GPL");
