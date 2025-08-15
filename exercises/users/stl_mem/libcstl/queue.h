/* $Id: queue.h,v 1.4 2003/09/15 12:47:46 jfwan Exp $ */
#ifndef WAN_QUEUE_H
#define WAN_QUEUE_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#include "general.h"

EXTERN_TYPE(queue);

#define   queue_new(type)   queue_new_aux(sizeof(type))
#define   queue_delete(q)   queue_delete_aux(&(q))

boolean   queue_empty(queue q);
size_t    queue_size(queue q);

#define   queue_front(q,x)  queue_front_aux(q, &(x))
#define   queue_back(q,x)   queue_back_aux(q, &(x))

#define   queue_push(q,x)   queue_push_aux(q, &(x))
void      queue_pop(queue q);

#define QUEUE_INTERNAL_
#include "bits/_queue.h"
#undef QUEUE_INTERNAL_

#endif // WAN_QUEUE_H
