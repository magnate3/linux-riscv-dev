#include "ngx_mem_pool_x64.h"

void* ngx_create_pool(size_t size)
{
    struct ngx_pool_s  *p;

    // 源码中的 ngx_memalign 实际上是跨屏台调用不同函数，但最终都是直接 malloc
    p = (struct ngx_pool_s *)malloc(size);
    if (p == nullptr) 
    {
        return nullptr;
    }

    p->d.last = (u_char *) p + sizeof(struct ngx_pool_s);
    p->d.end = (u_char *) p + size;
    p->d.next = nullptr;
    p->d.failed = 0;

    size = size - sizeof(struct ngx_pool_s);
    p->max = (size < NGX_MAX_ALLOC_FROM_POOL) ? size : NGX_MAX_ALLOC_FROM_POOL;

    p->current = p;
    p->large = nullptr;
    p->cleanup = nullptr;

   // this->pool_ = p;

    return p;

}

void* ngx_palloc(size_t size)
{

    if (size <= g_pool->max) 
    {
        return ngx_palloc_small(size,1);
    }

    return ngx_palloc_large(size);
}

void* ngx_pnalloc(size_t size)
{

    if (size <= g_pool->max) 
    {
        // 不进行内存对齐
        return ngx_palloc_small(size,0);
    }

    return ngx_palloc_large(size);
}

void* ngx_pcalloc(size_t size)
{
    void *p;

    p = ngx_palloc(size);
    if (p) 
    {
        ngx_memzero(p, size);
    }

    return p;
}

void* ngx_palloc_small(size_t size, ngx_uint_t align)
{
    u_char      *m;
    struct ngx_pool_s  *p;

    p = g_pool->current;

    do {
        m = p->d.last;

        if (align) 
        {
            m = ngx_align_ptr(m, NGX_ALIGNMENT);
        }

        if ((size_t) (p->d.end - m) >= size) {
            p->d.last = m + size;

            return m;
        }

        p = p->d.next;

    } while (p);

    return ngx_palloc_block(size);

}

void* ngx_palloc_block(size_t size)
{
    u_char      *m;
    size_t       psize;
    struct ngx_pool_s  *p, *new_pool;

    psize = (size_t) (g_pool->d.end - (u_char *)g_pool);

    m = (u_char *)malloc(psize);
    if (m == nullptr) 
    {
        return nullptr;
    }

    new_pool = (struct ngx_pool_s *) m;

    new_pool->d.end = m + psize;
    new_pool->d.next = nullptr;
    new_pool->d.failed = 0;

    m += sizeof(struct ngx_pool_data_t);
    m = ngx_align_ptr(m, NGX_ALIGNMENT);
    new_pool->d.last = m + size;

    for (p = g_pool->current; p->d.next; p = p->d.next) 
    {
        if (p->d.failed++ > 4) 
        {
            g_pool->current = p->d.next;
        }
    }

    p->d.next = new_pool;

    return m;
}


void* ngx_palloc_large(size_t size)
{
    void             *p;
    ngx_uint_t         n;
    struct ngx_pool_large_s  *large;

    p = (struct ngx_pool_large_s *)malloc(size);
    if (p == nullptr) 
    {
        return nullptr;
    }

    n = 0;

    for (large = g_pool->large; large; large = large->next) 
    {
        if (large->alloc == nullptr) {
            large->alloc = p;
            return p;
        }

        if (n++ > 3) {
            break;
        }
    }

    large = (struct ngx_pool_large_s *)ngx_palloc_small(sizeof(struct ngx_pool_large_s), 1);
    if (large == nullptr) 
    {
        free(p);
        return nullptr;
    }

    large->alloc = p;
    large->next = g_pool->large;
    g_pool->large = large;

    return p;

}

void ngx_pfree(void *p)
{
    struct ngx_pool_large_s  *l;

    for (l = g_pool->large; l; l = l->next) 
    {
        if (p == l->alloc) 
        {
            free(l->alloc);
            l->alloc = nullptr;

            return;
        }
    }
}



void ngx_rest_pool()
{
    struct ngx_pool_s        *p;
    struct ngx_pool_large_s  *l;

    for (l = g_pool->large; l; l = l->next) 
    {
        if (l->alloc) 
        {
            free(l->alloc);
        }
    }

    // 处理第一个块内存池
    p = g_pool;
    p->d.last = (u_char *)p + sizeof(struct ngx_pool_s);
    p->d.failed = 0;


    for (p = p->d.next; p; p = p->d.next) 
    {
        p->d.last = (u_char *) p + sizeof(struct ngx_pool_s);
        p->d.failed = 0;
    }

    g_pool->current = g_pool;
    g_pool->large = nullptr;
}


void ngx_destroy_pool()
{
    struct ngx_pool_s          *p, *n;
    struct ngx_pool_large_s    *l;
    struct ngx_pool_cleanup_s  *c;

    for (c = g_pool->cleanup; c; c = c->next) 
    {
        if (c->handler) 
        {
            c->handler(c->data);
        }
    }


    for (l = g_pool->large; l; l = l->next) 
    {
        if (l->alloc) 
        {
            free(l->alloc);
        }
    }

    for (p = g_pool, n = g_pool->d.next; /* void */; p = n, n = n->d.next) 
    {
        free(p);

        if (n == nullptr) 
        {
            break;
        }
    }
}


struct ngx_pool_cleanup_s *ngx_pool_cleanup_add(size_t size)
{
    struct ngx_pool_cleanup_s  *c;

    c = (struct ngx_pool_cleanup_s *)ngx_palloc(sizeof(struct ngx_pool_cleanup_s));
    if (c == nullptr) 
    {
        return nullptr;
    }

    if (size) 
    {
        c->data = ngx_palloc(size);
        if (c->data == nullptr) 
        {
            return nullptr;
        }

    } 
    else 
    {
        c->data = nullptr;
    }

    c->handler = nullptr;
    c->next = g_pool->cleanup;

    g_pool->cleanup = c;

    return c;
}
