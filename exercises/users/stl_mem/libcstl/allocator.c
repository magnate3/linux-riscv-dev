/* $Id: allocator.c,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#include <stdlib.h>
#include <string.h>
#include "allocator.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */
union obj
{
    union obj *free_list_link;
    char client_data[1];
};
typedef union obj obj;

static obj * VOLATILE s_free_list[NR_FREELIST] =
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static char    *s_start_free = 0;
static char    *s_end_free   = 0;
static size_t   s_heap_size  = 0;

static size_t s_round_up(size_t sz);
static size_t s_freelist_index(size_t sz);
static void * s_refill(size_t sz);
static char * s_chunk_alloc(size_t sz, size_t *nobj);

void * allocate(size_t sz)
{
    obj * VOLATILE * my_free_list;
    obj * result;

    if (sz > NR_MAX_BYTES)
        return malloc(sz);

    my_free_list = s_free_list + s_freelist_index(sz);
    result = *my_free_list;
    if (0 == result) { /* No corresponding freelist, need refill */
        void *r = s_refill(s_round_up(sz));
        return r;
    }

    *my_free_list = result->free_list_link;
    return (void*) result;
}

void deallocate(void *p, size_t sz)
{
    obj * VOLATILE * my_free_list;

    /* XXX: sgi stl doesn't permit p to be null pointer,
     * but I do.
     */
    if (!p) return;
    if (sz > (size_t) NR_MAX_BYTES)
        return free(p);

    my_free_list = s_free_list + s_freelist_index(sz);
    ((obj*) p)->free_list_link = *my_free_list;
    *my_free_list = (obj*) p;
}

void * reallocate(void *p, size_t old_sz, size_t new_sz)
{
    void *result;

    /* XXX: sgi stl is not permitted p to be null pointer,
     * but I do.
     */
    if (!p) return allocate(new_sz);

    if (old_sz > (size_t) NR_MAX_BYTES && new_sz > (size_t) NR_MAX_BYTES)
        return realloc(p, new_sz);
    if (s_round_up(old_sz) == s_round_up(new_sz))
        return p;

    result = allocate(new_sz);
    memcpy(result, p, old_sz);
    deallocate(p, old_sz);
    return result;
}

static size_t s_round_up(size_t sz)
{
    return ALIGN(sz);
}

static size_t s_freelist_index(size_t sz)
{
    return (size_t)((sz + NR_ALIGN_BYTES - 1) >> NR_ALIGN_BASE);
}

static void * s_refill(size_t sz)
{
    size_t nobj = 20;
    obj * VOLATILE *my_free_list;
    obj *result, *curr_obj, *next_obj;
    size_t i;

    char *chunk = s_chunk_alloc(sz, &nobj);
    if (1 == nobj) return chunk;

    my_free_list = s_free_list + s_freelist_index(sz);
    result = (obj*) chunk;
    *my_free_list = next_obj = (obj*) (chunk + sz);
    for (i = 1; ; ++i) { /* begin from 1, 0 returned to user */
        curr_obj = next_obj;
        next_obj = (obj*) ((char*) curr_obj + sz);
        if (nobj - 1 == i) {
            curr_obj->free_list_link = 0;
            break;
        } else {
            curr_obj->free_list_link = next_obj;
        }
    }

    return result;
}

static char * s_chunk_alloc(size_t sz, size_t *nobj)
{
    char *result;
    size_t total_bytes = sz * (*nobj);
    size_t left_bytes = s_end_free - s_start_free;

    if (left_bytes >= total_bytes) {
        result = s_start_free;
        s_start_free += total_bytes;
        return result;
    } else if (left_bytes >= sz) {
        *nobj = left_bytes / sz;
        result = s_start_free;
        s_start_free += sz * (*nobj);
        return result;
    } else {
        /* XXX: why to right shift 4, I'm puzzled, just get it from sgi alloc,
         * I think it must be an experience value.
         */
        size_t bytes_to_get = (total_bytes << 1) + s_round_up(s_heap_size >> 4);
        if (left_bytes > 0) {
            obj * VOLATILE *my_free_list = s_free_list + s_freelist_index(left_bytes);
            ((obj*) s_start_free)->free_list_link = *my_free_list;
            *my_free_list = (obj*) s_start_free;
        }

        s_start_free = (char*) malloc(bytes_to_get);
        if(0 == s_start_free) { /* no memory to malloc */
            size_t i;
            obj * VOLATILE *my_free_list;
            obj *p;

            for (i = sz; i < (size_t) NR_MAX_BYTES; i += (size_t) NR_ALIGN_BYTES) {
                my_free_list = s_free_list + s_freelist_index(i);
                p = *my_free_list;
                if(0 != p) {
                    *my_free_list = p->free_list_link;
                    s_start_free  = (char*) p;
                    s_end_free    = s_start_free + i;
                    return s_chunk_alloc(sz, nobj);
                }
            }
            s_end_free = 0;
            s_start_free = (char*) malloc(bytes_to_get);
        }

        s_end_free   = s_start_free + bytes_to_get;
        s_heap_size += bytes_to_get;
        return s_chunk_alloc(sz, nobj);
    }
}
