/*
 *ktask_cmdline.h: 2019-02-21 created by qudreams
 *
 *get task cmdline
 */

#ifndef KTASK_CMDLINE_H
#define KTASK_CMDLINE_H

struct task_struct;
struct kmem_cache;
struct linux_binprm;
char* ktask_get_cmdline(struct kmem_cache* cachep,
        struct linux_binprm *bprm,unsigned* plen);

#endif
