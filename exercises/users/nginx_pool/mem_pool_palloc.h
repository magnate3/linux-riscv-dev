/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_PALLOC_H_INCLUDED_
#define _NGX_PALLOC_H_INCLUDED_


#include "mem_core.h"


/*
 * NGX_MAX_ALLOC_FROM_POOL should be (ngx_pagesize - 1), i.e. 4095 on x86.
 * On Windows NT it decreases a number of locked pages in a kernel.
 */
#define NGX_MAX_ALLOC_FROM_POOL  (ngx_pagesize - 1)

#define NGX_DEFAULT_POOL_SIZE    (16 * 1024)

#define NGX_POOL_ALIGNMENT       16
#define NGX_MIN_POOL_SIZE                                                     \
    ngx_align((sizeof(ngx_pool_t) + 2 * sizeof(ngx_pool_large_t)),            \
              NGX_POOL_ALIGNMENT)




typedef struct ngx_pool_large_s  ngx_pool_large_t;



struct ngx_pool_large_s {
    ngx_pool_large_t     *next;     // 指向下一块大内存块的指针
    void                 *alloc;    // 大内存块的起始地址
};
 
typedef struct {
    u_char               *last;     // 保存当前数据块中内存分配指针的当前位置。每次Nginx程序从内存池中申请内存时，
                                    //从该指针保存的位置开始划分出请求的内存大小，并更新该指针到新的位置。
    u_char               *end;      // 保存内存块的结束位置
    ngx_pool_t           *next;     // 内存池由多块内存块组成，指向下一个数据块的位置。
    ngx_uint_t            failed;   // 当前数据块内存不足引起分配失败的次数
} ngx_pool_data_t;
 
struct ngx_pool_s {
    ngx_pool_data_t       d;        // 内存池当前的数据区指针的结构体
    size_t                max;      // 当前数据块最大可分配的内存大小（Bytes）
    ngx_pool_t           *current;  // 当前正在使用的数据块的指针
    ngx_pool_large_t     *large;    // pool 中指向大数据块的指针（大数据快是指 size > max 的数据块）

};



void *ngx_alloc(size_t size);
void *ngx_calloc(size_t size);

ngx_pool_t *ngx_create_pool(size_t size);
void ngx_destroy_pool(ngx_pool_t *pool);
void ngx_reset_pool(ngx_pool_t *pool);

void *ngx_palloc(ngx_pool_t *pool, size_t size);
void *ngx_pnalloc(ngx_pool_t *pool, size_t size);
void *ngx_pcalloc(ngx_pool_t *pool, size_t size);
void *ngx_pmemalign(ngx_pool_t *pool, size_t size, size_t alignment);
ngx_int_t ngx_pfree(ngx_pool_t *pool, void *p);



#endif /* _NGX_PALLOC_H_INCLUDED_ */
