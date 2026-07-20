/* $Id: stack.c,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#include "stack.h"
#include "list.h"
#include "allocator.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

typedef struct stack_struct
{
    list list_;
} stack_;

stack stack_new_aux(size_t sz)
{
    stack s = (stack) allocate(sizeof(stack_));
    s->list_ = (list) list_new_aux(sz);
    return s;
}

void stack_delete_aux(stack *s)
{
    list_delete((*s)->list_);
    deallocate(*s, sizeof(stack_));
    *s = NULL;
}

boolean stack_empty(stack s)
{
    return list_empty(s->list_);
}

void stack_top_aux(stack s, void *x)
{
    list_back_aux(s->list_, x);
}

void stack_push_aux(stack s, void *x)
{
    list_push_back_aux(s->list_, x);
}

void stack_pop(stack s)
{
    list_pop_back(s->list_);
}
