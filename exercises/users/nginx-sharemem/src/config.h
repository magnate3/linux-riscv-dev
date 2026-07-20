//
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
#include "ngx_config.h"
#include "ngx_atomic.h"

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

typedef pid_t       ngx_pid_t;

static ngx_int_t    ngx_ncpu;
static ngx_pid_t    ngx_pid;

#endif //SHAREMEM_CONFIG_H
