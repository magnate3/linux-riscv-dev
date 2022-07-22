#include <linux/types.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/pid.h>
#include <linux/version.h>
#include <linux/vermagic.h>
#include "ktask_cache.h"
#include "ktask_memcache.h"
#include "ktask_cmdline.h"
#include "ktask_hook.h"
#include "ktask.h"
#include "kpath.h"

struct task_struct* ktask_get_struct(pid_t pid)
{
    struct task_struct* tsk = NULL;

    #if LINUX_VERSION_CODE > KERNEL_VERSION(2,6,26)
        struct pid* spid = NULL;
        spid = find_get_pid(pid);
        if(spid) {
            rcu_read_lock();
            tsk = pid_task(spid,PIDTYPE_PID);
            if(tsk) { get_task_struct(tsk); }
            rcu_read_unlock();
            put_pid(spid);
        }
    #else
        rcu_read_lock();
    	tsk = find_task_by_pid(pid);
    	if(tsk) { get_task_struct(tsk); }
    	rcu_read_unlock();
    #endif

    return tsk;
}

//get thread-id of task
pid_t ktask_gettid(struct task_struct* tsk)
{
#if LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,23)
    pid_t pid = tsk->pid;
#else
    pid_t pid = task_pid_nr(tsk);
#endif

    return pid;
}

static char* ktask_get_pathname(struct linux_binprm *bprm,unsigned* plen)
{
    struct file* filp = NULL;
    char* pathname = ERR_PTR(-EINVAL);

    filp = bprm->file;
    if(filp) {
        pathname = kfilp_pathname(filp,plen);
    }

    return pathname;
}

void ktask_exec_notify(struct linux_binprm *bprm,struct task_struct* tsk)
{
    unsigned pathlen = 0;
    unsigned cmdlen = 0;
    pid_t pid = PID(tsk);
    struct kmem_cache* cachep = NULL;
    char* cmdline = ERR_PTR(-EINVAL);
    char* pathname = ERR_PTR(-EINVAL);

    cachep = ktask_get_cache(MAX_CACHE_SIZE);
    if(!cachep) { goto out; }

    pathname = ktask_get_pathname(bprm,&pathlen);
    if(IS_ERR(pathname)) { goto out; }

    cmdline = ktask_get_cmdline(cachep,bprm,&cmdlen);
    if(IS_ERR(cmdline)) { goto out; }

    printk("program: %s execute,pid: %d,cmdline: %s\n",
        pathname,(int)pid,cmdline);

out:
    if(!IS_ERR(pathname)) { kput_pathname(pathname); }
    if(!IS_ERR(cmdline)) { ktask_mem_cache_free(cachep,cmdline); }
}


#define DEVICE_NAME     "ktaskcmdline"

static int __init ktask_cmline_init(void)
{
    int rc = 0;
    printk("-----Start ktask_cmdline module,"
        "kernel-version: %s\n",UTS_RELEASE);
    rc = ktask_cache_init();
    if(rc) { return rc; }

    ktask_exec_hook_init();
    return rc;
}

static void __exit ktask_cmdline_exit(void)
{
    ktask_exec_hook_uninit();
    ktask_cache_uninit();
    printk("-----Exit ktask cmdline module-----\n");
}

module_init(ktask_cmline_init);
module_exit(ktask_cmdline_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("qudreams");
MODULE_DESCRIPTION(DEVICE_NAME);
MODULE_VERSION(DEVICE_VERSION);
