#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/version.h>

#include <linux/kprobes.h>



// preempt_count_display --------------------------------------------



static void preempt_count_display(void)
{
    unsigned preempt_cnt = preempt_count();
    // if  (preempt_cnt)
    pr_err("preempt_count: 0x%08x\n", preempt_cnt);
    pr_err("test_preempt_need_resched: %d\n", test_preempt_need_resched());
}

static void current_display(void)
{
    struct thread_info* thread_p = current_thread_info();
    struct task_struct* task_p = current;

    pr_err("thread_p: %px, task_p: %px\n", thread_p, task_p);
    
    
    // comm should use get_task_comm
    pr_err("cpu_id: %d, task tid/pid: %d, pid/tgid: %d, comm: %s\n", 
            smp_processor_id(), task_p->pid, task_p->tgid, task_p->comm);

    pr_err("thread flags: %lx\n", thread_p->flags);
   // pr_err("thread status: %lx\n", thread_p->status);

    pr_err("task flags: %lx\n", task_p->flags);
    pr_err("task state: %lx\n", task_p->state);


}

// kprobe -----------------------------------------------------------

#define MAX_SYMBOL_LEN	64
static char symbol[MAX_SYMBOL_LEN] = "_do_fork";
module_param_string(symbol, symbol, sizeof(symbol), 0644);



/* For each probe you need to allocate a kprobe structure */
static struct kprobe kp = {
    .symbol_name	= symbol,
};

/* for any context, information need to know
 * cpu, tid, pid, command name
 * preempt_count: thread context, irq context, hard_irq, soft_irq, preempt_disable, 
 * need_resched
 * lock
 */


static unsigned long cnt = 0;

/* kprobe pre_handler: called just before the probed instruction is executed */
static int handler_pre(struct kprobe *p, struct pt_regs *regs)
{
    if  (++cnt % 10 == 0)
    {
        pr_err("%s cnt: %ld --------------------------------------------------------------\n", 
                p->symbol_name, cnt);
        // pr_err("cnt: %ld\n", cnt);
        preempt_count_display();
        
        dump_stack();
    }
    

    return 0;
}

/*
 * fault_handler: this is called if an exception is generated for any
 * instruction within the pre- or post-handler, or when Kprobes
 * single-steps the probed instruction.
 */
static int handler_fault(struct kprobe *p, struct pt_regs *regs, int trapnr)
{
    pr_info("fault_handler: p->addr = 0x%p, trap #%dn", p->addr, trapnr);
    /* Return 0 because we don't handle the fault. */
    return 0;
}

// preempt_count_display -----------------------------------------------------


spinlock_t example_lock;
static void preempt_count_test(void)
{
    pr_err("module_init\n");
    preempt_count_display();



    spin_lock(&example_lock);

    pr_err("spin_lock\n");
    preempt_count_display();

    spin_unlock(&example_lock);


    local_bh_disable();

    pr_err("local_bh_disable\n");
    preempt_count_display();

    local_bh_enable();

    local_bh_disable();
    local_bh_disable();

    pr_err("local_bh_disable * 2\n");
    preempt_count_display();

    local_bh_enable();
    local_bh_enable();

    local_bh_disable();
    local_bh_disable();
    local_bh_disable();

    pr_err("local_bh_disable * 3\n");
    preempt_count_display();

    local_bh_enable();
    local_bh_enable();
    local_bh_enable();

    spin_lock_bh(&example_lock);

    pr_err("spin_lock_bh\n");
    preempt_count_display();

    spin_unlock_bh(&example_lock);


    preempt_disable();

    pr_err("preempt_disable\n");
    preempt_count_display();

    preempt_enable();


}


static void dump_stack_test(void)
{
    local_irq_disable();

    dump_stack();

    local_irq_enable();
}

// module init ---------------------------------------------------------------




static int __init preempt_count_display_init(void)
{
    // int ret;
    // kp.pre_handler = handler_pre;
    // // kp.post_handler = handler_post;
    // kp.post_handler = NULL;
    // kp.fault_handler = handler_fault;

    // ret = register_kprobe(&kp);
    // if (ret < 0) {
    //     pr_err("register_kprobe failed, returned %d\n", ret);
    //     return ret;
    // }
    // pr_info("Planted kprobe at %p\n", kp.addr);

    // current_display();

     preempt_count_display();
     preempt_count_test();
    
     //dump_stack_test();

    return 0;
}

static void __exit preempt_count_display_exit(void)
{
    // unregister_kprobe(&kp);

    // preempt_count_display();


    // pr_info("kprobe at %p unregistered\n", kp.addr);
    pr_err("----------------- (preempt_count_display -----------------------------\n");
}

module_init(preempt_count_display_init)
module_exit(preempt_count_display_exit)
MODULE_LICENSE("GPL");
