
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_SLAB_H_INCLUDED_
#define _NGX_SLAB_H_INCLUDED_


#include <ngx_config.h>
#include <ngx_core.h>


typedef struct ngx_slab_page_s  ngx_slab_page_t;
//图形化理解参考:http://blog.csdn.net/u013009575/article/details/17743261
struct ngx_slab_page_s { //初始化赋值在ngx_slab_init
    //多种情况，多个用途
    //当需要分配新的页的时候，分配N个页ngx_slab_page_s结构中第一个页的slab表示这次一共分配了多少个页 //标记这是连续分配多个page，并且我不是首page，例如一次分配3个page,分配的page为[1-3]，则page[1].slab=3  page[2].slab=page[3].slab=NGX_SLAB_PAGE_BUSY记录
    //如果OBJ<128一个页中存放的是多个obj(例如128个32字节obj),则slab记录里面的obj的大小，见ngx_slab_alloc_locked
    //如果obj移位大小为ngx_slab_exact_shift，也就是obj128字节，page->slab = 1;page->slab存储obj的bitmap,例如这里为1，表示说第一个obj分配出去了   见ngx_slab_alloc_locked
    //如果obj移位大小为ngx_slab_exact_shift，也就是obj>128字节，page->slab = ((uintptr_t) 1 << NGX_SLAB_MAP_SHIFT) | shift;//大于128，也就是至少256?4K最多也就16个256，因此只需要slab的高16位表示obj位图即可
    //当分配某些大小的obj的时候(一个缓存页存放多个obj)，slab表示被分配的缓存的占用情况(是否空闲)，以bit位来表示
    //如果
    uintptr_t         slab; //ngx_slab_init中初始赋值为共享内存中剩余页的个数
    //在ngx_slab_init中初始化的9个ngx_slab_page_s通过next连接在一起
    //如果页中的ojb<128 = 128 或者>128 ,则next直接指向对应的页slots[slot].next = page; 同时pages_m[]指向page->next = &slots[slot];
    ngx_slab_page_t  *next; //在分配较小obj的时候，next指向slab page在pool->pages的位置    //下一个page页
    //由于指针是4的倍数,那么后两位一定为0,此时我们可以利用指针的后两位做标记,充分利用空间. 用低两位记录NGX_SLAB_PAGE等标记
    //如果页中的obj<128,标记该页中存储的是小与128的obj page->prev = (uintptr_t) &slots[slot] | NGX_SLAB_SMALL
    //obj=128 page->prev = (uintptr_t) &slots[slot] | NGX_SLAB_EXACT;
    uintptr_t         prev;//上一个page页
};
/*
共享内存的其实地址开始处数据:ngx_slab_pool_t + 9 * sizeof(ngx_slab_page_t)(slots_m[]) + pages * sizeof(ngx_slab_page_t)(pages_m[]) +pages*ngx_pagesize(这是实际的数据部分，
每个ngx_pagesize都由前面的一个ngx_slab_page_t进行管理，并且每个ngx_pagesize最前端第一个obj存放的是一个或者多个int类型bitmap，用于管理每块分配出去的内存)

m_slot[0]:链接page页面,并且page页面划分的slot块大小为2^3
m_slot[1]:链接page页面,并且page页面划分的slot块大小为2^4
m_slot[2]:链接page页面,并且page页面划分的slot块大小为2^5
……………….
m_slot[8]:链接page页面,并且page页面划分的slot块大小为2k(2048)

m_page数组:数组中每个元素对应一个page页.
m_page[0]对应page[0]页面
m_page[1]对应page[1]页面
m_page[2]对应page[2]页面
………………………….
m_page[k]对应page[k]页面
另外可能有的m_page[]没有相应页面与他相对应.

*/
//图形化理解参考:http://blog.csdn.net/u013009575/article/details/17743261
typedef struct { //初始化赋值在ngx_slab_init
    ngx_shmtx_sh_t    lock; //mutex的锁

    size_t            min_size; //内存缓存obj最小的大小，一般是1个byte   //最小分配的空间是8byte 见ngx_slab_init
    //slab pool以shift来比较和计算所需分配的obj大小、每个缓存页能够容纳obj个数以及所分配的页在缓存空间的位置
    size_t            min_shift; //ngx_init_zone_pool中默认为3
/*
共享内存的其实地址开始处数据:ngx_slab_pool_t + 9 * sizeof(ngx_slab_page_t)(slots_m[]) + pages * sizeof(ngx_slab_page_t)(pages_m[]) +pages*ngx_pagesize(这是实际的数据部分，
每个ngx_pagesize都由前面的一个ngx_slab_page_t进行管理，并且每个ngx_pagesize最前端第一个obj存放的是一个或者多个int类型bitmap，用于管理每块分配出去的内存)
*/
    //指向ngx_slab_pool_t + 9 * sizeof(ngx_slab_page_t) + pages * sizeof(ngx_slab_page_t) +pages*ngx_pagesize(这是实际的数据部分)中的pages * sizeof(ngx_slab_page_t)开头处
    ngx_slab_page_t  *pages; //slab page空间的开头   初始指向pages * sizeof(ngx_slab_page_t)首地址
    ngx_slab_page_t  *last; // 也就是指向实际的数据页pages*ngx_pagesize，指向最后一个pages页
    //管理free的页面   是一个链表头,用于连接空闲页面.
    ngx_slab_page_t   free; //初始化赋值在ngx_slab_init  free->next指向pages * sizeof(ngx_slab_page_t)  下次从free.next是下次分配页时候的入口开始分配页空间

    u_char           *start; //实际缓存obj的空间的开头   这个是对地址空间进行ngx_pagesize对齐后的起始地址，见ngx_slab_init
    u_char           *end;

    ngx_shmtx_t       mutex; //ngx_init_zone_pool->ngx_shmtx_create->sem_init进行初始化

    u_char           *log_ctx;//pool->log_ctx = &pool->zero;
    u_char            zero;

    unsigned          log_nomem:1; //ngx_slab_init中默认为1

    //ngx_http_file_cache_init中cache->shpool->data = cache->sh;
    void             *data; //指向ngx_http_file_cache_t->sh
    void             *addr; //指向ngx_slab_pool_t的开头    //指向共享内存ngx_shm_zone_t中的addr+size尾部地址
} ngx_slab_pool_t;

//图形化理解参考:http://blog.csdn.net/u013009575/article/details/17743261
void ngx_slab_init(ngx_slab_pool_t *pool);
void *ngx_slab_alloc(ngx_slab_pool_t *pool, size_t size);
void *ngx_slab_alloc_locked(ngx_slab_pool_t *pool, size_t size);
void *ngx_slab_calloc(ngx_slab_pool_t *pool, size_t size);
void *ngx_slab_calloc_locked(ngx_slab_pool_t *pool, size_t size);
void ngx_slab_free(ngx_slab_pool_t *pool, void *p);
void ngx_slab_free_locked(ngx_slab_pool_t *pool, void *p);


#endif /* _NGX_SLAB_H_INCLUDED_ */
