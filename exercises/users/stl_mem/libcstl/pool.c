/* $Id: pool.c,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "pool.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

/* pheap - singular allocation of memory */
struct pheap
{
    void *block;
    int size, used;
};

/* pfree - a linked list node which stores an
 *    allocation chunk, plus a callback */
struct pfree
{
    pool_cleaner f;
    void *arg;
    struct pheap *heap;
    struct pfree *next;
};

/* pool - base node for a pool. Maintains a linked list
 *    of pool entries (pfree) */
typedef struct pool_struct
{
    int size;
    struct pfree *cleanup;
    struct pheap *heap;
} pool_;

/* make an empty pool */
static pool s_pool_new_empty()
{
    pool p;
    while((p = (pool) malloc(sizeof(pool_))) == NULL) sleep(1);
    p->cleanup = NULL;
    p->heap = NULL;
    p->size = 0;

    return p;
}

/* free a heap */
static void ss_pool_heap_free(void *arg)
{
    struct pheap *h = (struct pheap *)arg;

    free(h->block);
    free(h);
}

/* mem should always be freed last */
static void s_pool_cleanup_append(pool p, struct pfree *pf)
{
    struct pfree *cur;

    if(NULL == p->cleanup) {
        p->cleanup = pf;
        return;
    }

    /* fast forward to end of list */
    for(cur = p->cleanup; cur->next != NULL; cur = cur->next);
    cur->next = pf;
}

/* create a cleanup tracker */
static struct pfree *s_pool_free(pool p, pool_cleaner f, void *arg)
{
    struct pfree *ret;

    /* make the storage for the tracker */
    while((ret = (struct pfree *) malloc(sizeof(struct pfree))) == NULL)
        sleep(1);
    ret->f = f;
    ret->arg = arg;
    ret->next = NULL;

    return ret;
}

/* create a heap and make sure it get's cleaned up */
static struct pheap *s_pool_heap(pool p, int size)
{
    struct pheap *ret;
    struct pfree *clean;

    /* make the return heap */
    while((ret = (struct pheap *) malloc(sizeof(struct pheap))) == NULL)
        sleep(1);
    while((ret->block = malloc(size)) == NULL) sleep(1);
    ret->size = size;
    p->size += size;
    ret->used = 0;

    /* append to the cleanup list */
    clean = s_pool_free(p, ss_pool_heap_free, (void *)ret);
    clean->heap = ret; /* for future use in finding used mem for pstrdup */
    s_pool_cleanup_append(p, clean);

    return ret;
}

pool pool_new(int size)
{
    pool p;
    p = s_pool_new_empty();
    if (0 == size)
        size = getpagesize();
    p->heap = s_pool_heap(p,size);
    return p;
}

void *pmalloc(pool p, int size)
{
    void *block;

    if(NULL == p) {
        fprintf(stderr,"Memory Leak! [pmalloc received NULL pool, "
                "unable to track allocation, exiting]\n");
        abort();
    }

    /* if there is no heap for this pool or it's a big request, just raw,
     * I like how we clean this :)
     */
    if(NULL == p->heap || size > (p->heap->size / 2)) {
        while((block = malloc(size)) == NULL) sleep(1);
        p->size += size;
        s_pool_cleanup_append(p, s_pool_free(p, free, block));
        return block;
    }

    /* we have to preserve boundaries, long story :) */
    /* TODO: here, I changed to another method to improve performance
     * jfwan@vip.sina.com 07/04/2003
     */
    if (size >= 4)
        p->heap->used = ALIGN(p->heap->used);
    /*
    if(size >= 4)
        while(p->heap->used&7) p->heap->used++;
     */

    /* if we don't fit in the old heap, replace it */
    if(size > (p->heap->size - p->heap->used))
        p->heap = s_pool_heap(p, p->heap->size);

    /* the current heap has room */
    block = (char *)p->heap->block + p->heap->used;
    p->heap->used += size;
    return block;
}

void *pmalloc_x(pool p, int size, char c)
{
   void* result = pmalloc(p, size);
   if (result != NULL)
           memset(result, c, size);
   return result;
}  

/* easy safety utility (for creating blank mem for structs, etc) */
void *pmalloco(pool p, int size)
{
    void *block = pmalloc(p, size);
    memset(block, 0, size);
    return block;
}  

/* XXX efficient: move this to const char * and then loop throug
 * the existing heaps to see if src is within a block in this pool
 */
char *pstrdup(pool p, const char *src)
{
    char *ret;

    if(NULL == src)
        return NULL;

    ret = (char*) pmalloc(p,strlen(src) + 1);
    strcpy(ret,src);

    return ret;
}

/* when move above, this one would actually return a new block */
char *pstrdupx(pool p, const char *src)
{
    return pstrdup(p, src);
}

int pool_size(pool p)
{
    if(NULL == p) return 0;

    return p->size;
}

void pool_free(pool p)
{
    struct pfree *cur, *stub;

    if(NULL == p) return;

    cur = p->cleanup;
    while(cur != NULL) {
        (*cur->f)(cur->arg);
        stub = cur->next;
        free(cur);
        cur = stub;
    }

    free(p);
}

/* public cleanup utils, insert in a way that they are run FIFO, before mem frees */
void pool_cleanup(pool p, pool_cleaner f, void *arg)
{
    struct pfree *clean;

    clean = s_pool_free(p, f, arg);
    clean->next = p->cleanup;
    p->cleanup = clean;
}
