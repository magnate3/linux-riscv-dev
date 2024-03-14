/* $Id: vector.h,v 1.4 2003/09/15 12:47:46 jfwan Exp $ */
#ifndef WAN_VECTOR_H
#define WAN_VECTOR_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#include <stddef.h>
#include "general.h"
#include "pool.h"

EXTERN_TYPE(vector);

#define    vector_new(type) vector_new_aux(sizeof(type))
#define    vector_delete(v) vector_delete_aux(&(v))
#define    vector_push_back(v,val) vector_push_back_aux(v, (void*)val)
#define    vector_pop_back(v,val)  vector_pop_back_aux(v, &(val))
#define    vector_insert(v,position,x) vector_insert_aux(v, position, &(x))

void *     vector_begin(vector v);
void *     vector_end(vector v);
void *     vector_rbegin(vector v);
void *     vector_rend(vector v);

size_t     vector_max_size(const vector v);
size_t     vector_capacity(const vector v);
size_t     vector_size(const vector v);
boolean    vector_empty(const vector v);
void       vector_resize(vector v, size_t sz);
void       vector_reserve(vector v, size_t sz);

void       vector_erase(vector v, void *position);
void       vector_clear(vector v);
void       vector_swap(vector v1, vector v2);

#define VECTOR_INTERNAL
#include "bits/_vector.h"
#undef VECTOR_INTERNAL

#endif // WAN_VECTOR_H
