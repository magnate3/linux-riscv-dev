
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_SHMTX_H_INCLUDED_
#define _NGX_SHMTX_H_INCLUDED_

#include "config.h"
#include <semaphore.h>
/* GCC 4.1 builtin atomic operations */


typedef long                        ngx_atomic_int_t;
typedef unsigned long               ngx_atomic_uint_t;

#if (NGX_PTR_SIZE == 8)
#define NGX_ATOMIC_T_LEN            (sizeof("-9223372036854775808") - 1)
#else
#define NGX_ATOMIC_T_LEN            (sizeof("-2147483648") - 1)
#endif

typedef volatile ngx_atomic_uint_t  ngx_atomic_t;


typedef struct {
    ngx_atomic_t   lock;
    ngx_atomic_t   wait;
} ngx_shmtx_sh_t;


typedef struct {
    ngx_atomic_t  *lock;
    ngx_atomic_t  *wait;
    ngx_uint_t     semaphore;
    sem_t          sem;
    ngx_uint_t     spin;
} ngx_shmtx_t;


ngx_int_t ngx_shmtx_create(ngx_shmtx_t *mtx, ngx_shmtx_sh_t *addr,
    u_char *name);
void ngx_shmtx_destroy(ngx_shmtx_t *mtx);
ngx_uint_t ngx_shmtx_trylock(ngx_shmtx_t *mtx);
void ngx_shmtx_lock(ngx_shmtx_t *mtx);
void ngx_shmtx_unlock(ngx_shmtx_t *mtx);
void ngx_spinlock(ngx_atomic_t *lock, ngx_atomic_int_t value, ngx_uint_t spin);
ngx_uint_t ngx_shmtx_force_unlock(ngx_shmtx_t *mtx, ngx_pid_t pid);


#endif /* _NGX_SHMTX_H_INCLUDED_ */