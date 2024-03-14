/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_CORE_H_INCLUDED_
#define _NGX_CORE_H_INCLUDED_


#define NGX_HAVE_POSIX_MEMALIGN  1


typedef struct ngx_pool_s            ngx_pool_t;


#define  NGX_OK          0
#define  NGX_ERROR      -1
#define  NGX_AGAIN      -2
#define  NGX_BUSY       -3
#define  NGX_DONE       -4
#define  NGX_DECLINED   -5
#define  NGX_ABORT      -6



#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <error.h>
#include <sys/stat.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#ifdef USE_32_SYS
typedef intptr_t        ngx_int_t;
typedef uintptr_t       ngx_uint_t;
#else 
typedef long int  ngx_int_t;
typedef unsigned long int  ngx_uint_t;
#endif
#define NGX_ALIGNMENT   sizeof(unsigned long)    /* platform word */
#define ngx_align(d, a)     (((d) + (a - 1)) & ~(a - 1))
#define ngx_align_ptr(p, a)                                                   \
    (u_char *) (((uintptr_t) (p) + ((uintptr_t) a - 1)) & ~((uintptr_t) a - 1))

#define ngx_memzero(buf, n)       (void) memset(buf, 0, n)

#include "mem_alloc.h"
#include "mem_pool_palloc.h"


#endif /* _NGX_CORE_H_INCLUDED_ */
