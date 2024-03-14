//
// Created by zhangchao12 on 2020/12/8.
//

#ifndef SHAREMEM_CONFIG_H
#define SHAREMEM_CONFIG_H

#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

#define NGX_OK 0
#define NGX_ERROR -1
#define NGX_EINTR EINTR
#define ngx_memzero(buf, n)       (void) memset(buf, 0, n)

#ifndef NGX_HAVE_ATOMIC_OPS
#define NGX_HAVE_ATOMIC_OPS 1
#endif

#ifndef NGX_HAVE_SCHED_YIELD
#define NGX_HAVE_SCHED_YIELD  1
#endif

#ifndef NGX_HAVE_MAP_ANON
#define NGX_HAVE_MAP_ANON 1
#endif

#ifndef NGX_HAVE_MAP_DEVZERO
#define NGX_HAVE_MAP_DEVZERO 1
#endif

#ifndef NGX_HAVE_SYSVSHM
#define NGX_HAVE_SYSVSHM 1
#endif

#ifndef NGX_HAVE_SC_NPROCESSORS_ONLN
#define NGX_HAVE_SC_NPROCESSORS_ONLN 1
#endif

#ifndef ngx_inline
#define ngx_inline      inline
#endif

#define ngx_atomic_cmp_set(lock, old, set)                                    \
    __sync_bool_compare_and_swap(lock, old, set)

#define ngx_atomic_fetch_add(value, add)                                      \
    __sync_fetch_and_add(value, add)

#define ngx_memory_barrier()        __sync_synchronize()


#define ngx_trylock(lock)  (*(lock) == 0 && ngx_atomic_cmp_set(lock, 0, 1))
#define ngx_unlock(lock)    *(lock) = 0

#define ngx_align(d, a)     (((d) + (a - 1)) & ~(a - 1))
#define ngx_align_ptr(p, a)                                                   \
    (u_char *) (((uintptr_t) (p) + ((uintptr_t) a - 1)) & ~((uintptr_t) a - 1))


#if NGX_HAVE_SC_NPROCESSORS_ONLN
#define init_ncpu() \
    do{\
        if(ngx_ncpu == 0){ngx_ncpu = sysconf(_SC_NPROCESSORS_ONLN);}\
        if(ngx_ncpu < 1){ngx_ncpu = 1;}\
    }while(0)
#else
#define init_ncpu() do {ngx_ncpu = 1;}while(0)
#endif

#define init_pid() ngx_pid = getpid()


#if (NGX_HAVE_SCHED_YIELD)
#include <sched.h>
#define ngx_sched_yield()  sched_yield()
#else
#define ngx_sched_yield()  usleep(1)
#endif

#if ( __i386__ || __i386 || __amd64__ || __amd64 )
#define ngx_cpu_pause()             __asm__ ("pause")
#else
#define ngx_cpu_pause()
#endif

typedef intptr_t    ngx_int_t;
typedef uintptr_t   ngx_uint_t;
typedef pid_t       ngx_pid_t;

static ngx_int_t    ngx_ncpu;
static ngx_pid_t    ngx_pid;

#endif //SHAREMEM_CONFIG_H
