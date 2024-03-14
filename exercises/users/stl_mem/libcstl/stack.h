/* $Id: stack.h,v 1.4 2003/09/15 12:47:46 jfwan Exp $ */
#ifndef WAN_STACK_H
#define WAN_STACK_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#include "general.h"

EXTERN_TYPE(stack);

#define   stack_new(type) stack_new_aux(sizeof(type))
#define   stack_delete(s) stack_delete_aux(&(s))

boolean   stack_empty(stack s);
#define   stack_top(s,x)  stack_top_aux(s, &(x))
#define   stack_push(s,x) stack_push_aux(s, &(x))
void      stack_pop(stack s);

#define STACK_INTERNAL_
#include "bits/_stack.h"
#undef STACK_INTERNAL_

#endif // WAN_STACK_H
