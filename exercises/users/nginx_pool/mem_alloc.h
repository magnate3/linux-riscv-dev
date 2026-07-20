/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_ALLOC_H_INCLUDED_
#define _NGX_ALLOC_H_INCLUDED_



#include "mem_core.h"


void *ngx_alloc(size_t size);
void *ngx_calloc(size_t size);

#define ngx_free         free


/*
 * Linux has memalign() or posix_memalign()
 * Solaris has memalign()
 * FreeBSD 7.0 has posix_memalign(), besides, early version's malloc()
 * aligns allocations bigger than page size at the page boundary
 */

/*#if (NGX_HAVE_POSIX_MEMALIGN || NGX_HAVE_MEMALIGN)

void *ngx_memalign(size_t alignment, size_t size);

#else
*/
#define ngx_memalign(alignment, size)  ngx_alloc(size)
/*
#endif
*/

extern ngx_uint_t  ngx_pagesize;
extern ngx_uint_t  ngx_pagesize_shift;
extern ngx_uint_t  ngx_cacheline_size;


#endif /* _NGX_ALLOC_H_INCLUDED_ */
