/* $Id: allocator.h,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#ifndef WAN_ALLOCATOR_H
#define WAN_ALLOCATOR_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#include <stddef.h>
#include "general.h"

enum { NR_MAX_BYTES = 128 };
enum { NR_FREELIST = NR_MAX_BYTES >> NR_ALIGN_BASE };

void * allocate(size_t sz);
void   deallocate(void *p, size_t sz);
void * reallocate(void *p, size_t old_sz, size_t new_sz);

#endif // WAN_ALLOCATOR_H
