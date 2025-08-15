#ifndef KTASK_H
#define KTASK_H

#include <linux/types.h>
#include <linux/module.h>
#include <linux/version.h>
#include <linux/sched.h>
#include <linux/binfmts.h>
#include <linux/profile.h>


#if LINUX_VERSION_CODE > KERNEL_VERSION(2,6,23)
    #define PID(ts) task_tgid_vnr(ts)
#else
    #define PID(ts) ((ts)->tgid)
#endif

 /*
 * get struct task_struct by pid and pid_type
 * Caller must call ktask_put_struct(struct task_struct*) to release task_struct reference
 */
struct task_struct* ktask_get_struct(pid_t pid);
#define ktask_put_struct put_task_struct
pid_t ktask_gettid(struct task_struct* tsk);

void ktask_exec_notify(struct linux_binprm *bprm,struct task_struct* tsk);

#endif
