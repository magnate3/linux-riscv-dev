/* $Id: vector.c,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#include <stdio.h>
#include <string.h>
#include "vector.h"
#include "allocator.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

typedef struct vector_struct
{
    void    *data;
    size_t   bytes;
    size_t   size;
    size_t   used;
} vector_;

static void   s_expand_capacity(vector v);

vector vector_new_aux(size_t size)
{
    vector v;

    v = (vector) allocate(sizeof(vector_));
    v->data = NULL;
    v->bytes = size;
    v->size = v->used = 0;
    return v;
}

void vector_delete_aux(vector *pv)
{
    deallocate((*pv)->data, (*pv)->size);
    deallocate(*pv, sizeof(vector_));
    *pv = NULL;
}

void vector_push_back_aux(vector v, const void *x)
{
    if (v->used >= v->size) s_expand_capacity(v);

    memcpy((char*) v->data + v->used * v->bytes, (char*) &x, v->bytes);
    ++v->used;
}

void vector_pop_back_aux(vector v, void *x)
{
    /* XXX: If empty, should give a warning! */
    if (v->used > 0) {
        --v->used;
        memcpy((char*) x, (char*) v->data + v->used * v->bytes, v->bytes);
    } else {
        warning("Warning: pop back from empty vector");
    }
}

void vector_insert_aux(vector v, void *position, const void *x)
{
    if (v->used >= v->size) s_expand_capacity(v);
    memmove((char*) position + v->bytes, (char*) position,
            DISTANCE(position, vector_end(v)));
    memcpy(position, x, v->bytes);
    ++v->used;
}

void * vector_begin(const vector v)
{
    return v->data;
}

void * vector_end(const vector v)
{
    return (char*) v->data + v->used * v->bytes;
}

void * vector_rbegin(const vector v)
{
    return (char*) v->data + (v->used - 1) * v->bytes;
}

void * vector_rend(const vector v)
{
    return (char*) v->data - v->bytes;
}

size_t vector_max_size(const vector v)
{
    return (size_t)-1 / v->bytes;
}

size_t vector_capacity(const vector v)
{
    return v->size;
}

size_t vector_size(const vector v)
{
    return v->used;
}

boolean vector_empty(const vector v)
{
    return (boolean) (v->used == 0);
}

void vector_resize(vector v, size_t sz)
{
    void *tmp;

    if (sz > v->size) {
        tmp = reallocate(v->data, v->size * v->bytes, sz * v->bytes);
        memmove(tmp, v->data, v->used * v->bytes);
        v->data = tmp;
        v->size = sz;
    }
}

void vector_reserve(vector v, size_t sz)
{
    vector_resize(v, sz);
}

void vector_erase(vector v, void *position)
{
    if (v->data <= position && position < vector_end(v)) {
        memmove((char*) position, (char*) position + v->bytes,
                DISTANCE((char*) position + v->bytes, vector_end(v)));
        /* XXX: should end to be zero */
        memset(vector_end(v), 0, v->bytes);
        --v->used;
    }
}

void vector_clear(vector v)
{
    /* XXX: should all the entries to be set zero */
    memset(v->data, 0, v->used * v->bytes);
    v->used = 0;
}

void vector_swap(vector v1, vector v2)
{
    void *tmp = allocate(sizeof(vector_));
    memcpy(tmp, v1, sizeof(vector_));
    memcpy(v1, v2, sizeof(vector_));
    memcpy(v2, tmp, sizeof(vector_));
}

static void s_expand_capacity(vector v)
{
    int len = v->size != 0 ? 2 * v->size : 1; 
    v->data = reallocate(v->data, v->size * v->bytes, len * v->bytes);
    v->size = len;
}
