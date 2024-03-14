/* $Id: _queue.h,v 1.3 2003/09/15 12:47:50 jfwan Exp $ */
#ifndef WAN_QUEUE_INTERNAL_H
#define WAN_QUEUE_INTERNAL_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#ifndef QUEUE_INTERNAL_
#error Error: Never include queue_internal.h directly.
#endif

queue    queue_new_aux(size_t sz);
void     queue_delete_aux(queue *q);

void     queue_front_aux(queue q, void *x);
void     queue_back_aux(queue q, void *x);

void     queue_push_aux(queue q, void *x);

#endif // QUEUE_INTERNAL_H
