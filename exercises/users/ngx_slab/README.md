# Nginx_---

## log
```
void ngx_log_error(ngx_uint_t level, ngx_log_t *log,...){
   printf("errno==%d\n",errno);

}
```


 
## mutex

```
ngx_int_t
ngx_shm_alloc(ngx_shm_t *shm)
{
    int  id;

    id = shmget(IPC_PRIVATE, shm->size, (SHM_R|SHM_W|IPC_CREAT));

    if (id == -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,"shmget(%uz) failed", shm->size);
        return NGX_ERROR;
    }

    //ngx_log_debug1(NGX_LOG_DEBUG_CORE, shm->log, 0, "shmget id: %d", id);

    shm->addr = shmat(id, NULL, 0);

    if (shm->addr == (void *) -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno, "shmat() failed");
    }

    if (shmctl(id, IPC_RMID, NULL) == -1) {
        ngx_log_error(NGX_LOG_ALERT, shm->log, ngx_errno,
                      "shmctl(IPC_RMID) failed");
    }

    return (shm->addr == (void *) -1) ? NGX_ERROR : NGX_OK;
}
```
IPC_PRIVATE，得到的消息队列的key值都是0，而且再次调用又会创建一个新的消息队列，因此无法实现通信。只能在创建子进程之前先创建一个消息队列，让子进程继承，来实现父子进程之间或者兄弟进程之间的通信。    

## ngx_shmtx_lock

```
void *
ngx_slab_alloc(ngx_slab_pool_t *pool, size_t size)
{
    void  *p;

    ngx_shmtx_lock(&pool->mutex);

    p = ngx_slab_alloc_locked(pool, size);

    ngx_shmtx_unlock(&pool->mutex);

    return p;
}
```


##  ngx_init_zone_pool
```
static ngx_int_t
ngx_init_zone_pool(ngx_cycle_t *cycle, ngx_shm_zone_t *zn)
{
    u_char           *file;
    ngx_slab_pool_t  *sp;

    sp = (ngx_slab_pool_t *) zn->shm.addr;

    if (zn->shm.exists) {

        if (sp == sp->addr) {
            return NGX_OK;
        }

#if (NGX_WIN32)

        /* remap at the required address */

        if (ngx_shm_remap(&zn->shm, sp->addr) != NGX_OK) {
            return NGX_ERROR;
        }

        sp = (ngx_slab_pool_t *) zn->shm.addr;

        if (sp == sp->addr) {
            return NGX_OK;
        }

#endif

        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "shared zone \"%V\" has no equal addresses: %p vs %p",
                      &zn->shm.name, sp->addr, sp);
        return NGX_ERROR;
    }

    sp->end = zn->shm.addr + zn->shm.size;
    sp->min_shift = 3;
    sp->addr = zn->shm.addr;

#if (NGX_HAVE_ATOMIC_OPS)

    file = NULL;

#else

    file = ngx_pnalloc(cycle->pool,
                       cycle->lock_file.len + zn->shm.name.len + 1);
    if (file == NULL) {
        return NGX_ERROR;
    }

    (void) ngx_sprintf(file, "%V%V%Z", &cycle->lock_file, &zn->shm.name);

#endif

    if (ngx_shmtx_create(&sp->mutex, &sp->lock, file) != NGX_OK) {
        return NGX_ERROR;
    }

    ngx_slab_init(sp);

    return NGX_OK;
}
```
#  ngx_slab_free(): pointer to wrong chunk
+ 没有初始化 NGX_PTR_SIZE 、ngx_pagesize 和 ngx_pagesize_shift

```
   // 因为地址对齐，p是slot的起始地址，因此p的地址一定是(size-1)的倍数，例如size = 8,那么p的地址一定是8的倍数
        if ((uintptr_t) p & (size - 1)) 
        {
            goto wrong_chunk;
        }
```
## NGX_PTR_SIZE

```
#if (NGX_PTR_SIZE == 4)
#define NGX_INT_T_LEN   NGX_INT32_LEN
#define NGX_MAX_INT_T_VALUE  2147483647

#else
#define NGX_INT_T_LEN   NGX_INT64_LEN
#define NGX_MAX_INT_T_VALUE  9223372036854775807
#endif
```

```
#if (NGX_PTR_SIZE == 4)

#define NGX_SLAB_PAGE_FREE   0
#define NGX_SLAB_PAGE_BUSY   0xffffffff
#define NGX_SLAB_PAGE_START  0x80000000

#define NGX_SLAB_SHIFT_MASK  0x0000000f
#define NGX_SLAB_MAP_MASK    0xffff0000
#define NGX_SLAB_MAP_SHIFT   16

#define NGX_SLAB_BUSY        0xffffffff

#else /* (NGX_PTR_SIZE == 8) */

#define NGX_SLAB_PAGE_FREE   0
#define NGX_SLAB_PAGE_BUSY   0xffffffffffffffff
#define NGX_SLAB_PAGE_START  0x8000000000000000

#define NGX_SLAB_SHIFT_MASK  0x000000000000000f
#define NGX_SLAB_MAP_MASK    0xffffffff00000000
#define NGX_SLAB_MAP_SHIFT   32

#define NGX_SLAB_BUSY        0xffffffffffffffff

#endif
```

## 编译 和运行
 
+ 定义NGX_PTR_SIZE（4字节还是8字节）   
+  定义uintptr_t（4字节还是8字节）   
+  定义NGX_ALIGNMENT（4字节还是8字节）、ngx_uint_t、ngx_int_t      
+   初始化ngx_pagesize 和 ngx_pagesize_shift   
+  ngx_slab_max_size 、ngx_slab_exact_size、 ngx_slab_exact_shift
+  ngx_pid = ngx_getpid() 用于trylock
```C
#define ngx_align_ptr(p, a)                                                   \
    (u_char *) (((uintptr_t) (p) + ((uintptr_t) a - 1)) & ~((uintptr_t) a - 1))
```

```shell
 gcc ngx_alloc.c  ngx_palloc.c  ngx_shmem.c  ngx_shmtx.c  ngx_slab.c  ngx_slab_main.c  -o   test -I ../head -DNGX_HAVE_ATOMIC_OPS -DNGX_PTR_SIZE=8
````

```
void
ngx_slab_sizes_init(void)
{
    ngx_uint_t  n;

    ngx_slab_max_size = ngx_pagesize / 2;
    ngx_slab_exact_size = ngx_pagesize / (8 * sizeof(uintptr_t));
    for (n = ngx_slab_exact_size; n >>= 1; ngx_slab_exact_shift++) {
        /* void */
    }
}
```

```

void
ngx_slab_sizes_init(void)
{
    ngx_uint_t  n;

    ngx_slab_max_size = ngx_pagesize / 2;
    ngx_slab_exact_size = ngx_pagesize / (8 * sizeof(uintptr_t));
    for (n = ngx_slab_exact_size; n >>= 1; ngx_slab_exact_shift++) {
        /* void */
    }
}
```


## slab 大小

slab allocator 代码中用了相当多的位操作，很大一部分操作和 slab allocator 的 分级相关。从 2^3 bytes开始，到 pagesize/2 bytes 为止，提供 `2^3, 2^4, 2^5, ..., 2^(ngx_pagesize_shift - 1)` 等 `ngx_pagesize_shift - 3 `个内存片段大小等级。

## slab page

```C
struct ngx_slab_page_s {
    uintptr_t         slab;
    ngx_slab_page_t  *next;
    uintptr_t         prev;
};
```

本数据结构对应于Nginx slab内存管理的页概念，页在slab管理设计中是很核心的概念。ngx_slab_page_t中各字段根据不同内存页类型有不同的含义，下面我们就分别介绍一下：
1) 小块内存，小于ngx_slab_exact_size  
2) 中等内存， 等于ngx_slab_exact_size  
3) 大块内存，大于ngx_slab_exact_size而小于等于ngx_slab_max_size  
4) 超大内存，大于ngx_slab_max_size  

 