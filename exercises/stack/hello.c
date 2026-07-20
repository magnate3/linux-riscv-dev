#include <linux/init.h>
#include <linux/module.h>
#include <linux/sched.h>

//#define task_stack_page(task)    ((void *)(task)->stack)
//#define task_pt_regs(p) \
//    ((struct pt_regs *)(THREAD_SIZE + task_stack_page(p)) - 1)

static void print_task_info(struct task_struct *task)
{
    printk(KERN_NOTICE "%10s %5d task_struct (%p) / stack(%p~%p) / thread_info(%p)",
        task->comm, 
        task->pid,
        task,
        task->stack,
        ((unsigned long *)task->stack) + THREAD_SIZE,
        task_thread_info(task));
  
     printk(KERN_NOTICE "thread_info(%p)", &(task->thread_info));
     struct pt_regs *regs = ((struct pt_regs *)(THREAD_SIZE + task->stack) - 1);
    printk(KERN_NOTICE "pc(%lu),  sp(%lu, fp(%lu)",regs->user_regs.pc, regs->user_regs.sp, regs->user_regs.pstate);
    // struct thread_struct  struct cpu_context 
    printk(KERN_NOTICE "saved pc(%lu), saved sp(%lu,saved fp(%lu)",task->thread.cpu_context.pc, task->thread.cpu_context.sp, task->thread.cpu_context.fp);
      
}

static int __init task_init(void)
{
    struct task_struct *task = current;

    printk(KERN_INFO "task module init\n");

    print_task_info(task);
    do {
        task = task->parent;
        print_task_info(task);
    } while (task->pid != 0);

    return 0;
}
module_init(task_init);

static void __exit task_exit(void)
{
    printk(KERN_INFO "task module exit\n ");
}
module_exit(task_exit);
