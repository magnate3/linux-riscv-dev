#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/version.h>
#include <linux/sched.h>
#include <linux/binfmts.h>
#include <linux/profile.h>
#include "ktask_hook.h"
#include "ktask.h"


/*
 * Note: be carefull the return value,the value -ENOEXEC will suppose kernel to continue
 * the value -EACCES will suppose kernel to stop process executing.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,8,0)
static int ktask_load_binary(struct linux_binprm *bprm)
#else
static int ktask_load_binary(struct linux_binprm *bprm, struct pt_regs* regs)
#endif
{
    int rc = -ENOEXEC;
	ktask_exec_notify(bprm,current);
    return rc;
}

static struct linux_binfmt ktask_binfmt = {
    .module         = THIS_MODULE,
    .load_binary    = ktask_load_binary,
};

void ktask_exec_hook_init(void) {
    printk("init ktask exec hook\n");
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2,6,30)
    insert_binfmt(&ktask_binfmt);
#else
    register_binfmt(&ktask_binfmt);
#endif
}

void ktask_exec_hook_uninit(void) {
    unregister_binfmt(&ktask_binfmt);
    printk("exit ktask exec hook\n");
}
