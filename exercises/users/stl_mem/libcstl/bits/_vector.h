/* $Id: _vector.h,v 1.3 2003/09/15 12:47:50 jfwan Exp $ */
#ifndef WAN_VECOTR_INTERNAL_H
#define WAN_VECOTR_INTERNAL_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#ifndef VECTOR_INTERNAL
#error Error: vector_internal.h should not be included directly.
#endif

vector vector_new_aux(size_t size);
void   vector_delete_aux(vector *pv);
void   vector_push_back_aux(vector v, const void *val);
void   vector_pop_back_aux(vector v, void *val);
void   vector_insert_aux(vector v, void *position, const void *x);


#endif // WAN_VECOTR_INTERNAL_H
