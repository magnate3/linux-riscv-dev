/* $Id: _list.h,v 1.3 2003/09/15 12:47:50 jfwan Exp $ */
#ifndef WAN_LIST_INTERNAL_H
#define WAN_LIST_INTERNAL_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#ifndef LIST_INTERNAL_
#error Error: Never include this file directly
#endif

list      list_new_aux(size_t sz);
void      list_delete_aux(list *l);
void      list_push_front_aux(list l, void *x);
void      list_push_back_aux(list l, void *x);
void      list_front_aux(list l, void *x);
void      list_back_aux(list l, void *x);
list_node list_insert_aux(list l, list_node position, void *x);
void      list_insert_n_aux(list l, list_node position, size_t n, void *x);
void      list_remove_aux(list l, void *x);

#endif // WAN_LIST_INTERNAL_H
