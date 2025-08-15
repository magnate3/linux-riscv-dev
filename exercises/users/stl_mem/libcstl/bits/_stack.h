/* $Id: _stack.h,v 1.3 2003/09/15 12:47:50 jfwan Exp $ */
#ifndef WAN_STACK_INTERNAL_H
#define WAN_STACK_INTERNAL_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#ifndef STACK_INTERNAL_
#error Error: Never include stack_internal.h directly.
#endif

stack    stack_new_aux(size_t sz);
void     stack_delete_aux(stack *s);

void     stack_top_aux(stack s, void *x);
void     stack_push_aux(stack s, void *x);

#endif // WAN_STACK_INTERNAL_H
