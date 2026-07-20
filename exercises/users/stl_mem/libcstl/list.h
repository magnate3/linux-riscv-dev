/* $Id: list.h,v 1.4 2003/09/15 12:47:46 jfwan Exp $ */
#ifndef WAN_LIST_H
#define WAN_LIST_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#include <stddef.h>
#include "general.h"

typedef struct list_node_struct
{
    struct list_node_struct *prev;
    struct list_node_struct *next;
    void                    *data;
} list_node_;

EXTERN_TYPE(list_node);
EXTERN_TYPE(list);
CONST_TYPE(list, const_list);

#define   list_new(type)       list_new_aux(sizeof(type))
#define   list_delete(l)       list_delete_aux(&(l))

list_node list_begin(list l);
list_node list_end(list l);
list_node list_rbegin(list l);
list_node list_rend(list l);

boolean   list_empty(const list l);
size_t    list_size(const list l);
size_t    list_max_size(const_list l);
void      list_resize(list l, size_t sz);

#define   list_front(l,x)      list_front_aux(l, (void*) &(x))
#define   list_back(l,x)       list_back_aux(l, (void*) &(x))

#define   list_push_front(l,x) list_push_front_aux(l, (void*) (x))
void      list_pop_front(list l);
#define   list_push_back(l,x)  list_push_back_aux(l, (void*) (x))
void      list_pop_back(list l);

#define   list_insert(l,position,x)     list_insert_aux(l, position, (void*) (x))
#define   list_insert_n(l,position,n,x) list_insert_n_aux(l, position, n, (void*) (x))

list_node list_erase(list l, list_node node);
list_node list_erase_range(list l, list_node b, list_node e);
void      list_swap(list x, list y);
void      list_clear(list l);
void      list_splice(list l, list_node position, list_node b, list_node e);
#define   list_remove(l,x)     list_remove_aux(l, (void*) (x))
/* FIXME: stl standard has a method remove_if(), I haven't support it till now. */

void      list_unique(list l);
/* FIXME: stl standard has a method with predicate unique(), but I have no. */
void      list_merge(list dest, list src);
/* FIXME: stl standard has a method with comp merge(), but I have no. */
void      list_sort(list l);
/* FIXME: stl standard has a method with comp sort(), but I have no. */
void      list_reverse(list l);

comp_fun  list_set_comparer(comp_fun func);

#define LIST_INTERNAL_
#include "bits/_list.h"
#undef LIST_INTERNAL_

#endif // WAN_LIST_H
