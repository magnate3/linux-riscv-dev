/* $Id: queue.c,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#include "queue.h"
#include "list.h"
#include "allocator.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

typedef struct queue_struct
{
    list list_;
} queue_;

queue queue_new_aux(size_t sz)
{
    queue q = (queue) allocate(sizeof(queue_));
    q->list_ = (list) allocate(sz);
    return q;
}

void queue_delete_aux(queue *q)
{
    list_delete((*q)->list_);
    deallocate(*q, sizeof(queue_));
}

boolean queue_empty(queue q)
{
    return list_empty(q->list_);
}

size_t queue_size(queue q)
{
    return list_size(q->list_);
}

void queue_front_aux(queue q, void *x)
{
    list_front_aux(q->list_, x);
}

void queue_back_aux(queue q, void *x)
{
    list_back_aux(q->list_, x);
}

void queue_push_aux(queue q, void *x)
{
    list_push_back_aux(q->list_, x);
}

void queue_pop(queue q)
{
    list_pop_back(q->list_);
}
