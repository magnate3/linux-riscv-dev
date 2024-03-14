/* $Id: pool.h,v 1.4 2003/09/15 12:48:17 jfwan Exp $ */
#ifndef WAN_POOL_H
#define WAN_POOL_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#include "general.h"

EXTERN_TYPE(pool);

/* pool_cleaner - callback type which is associated
 * with a pool entry; invoked when the pool entry is free'd
 */
typedef void (*pool_cleaner)(void *arg);

/* creates a new memory pool with an initial heap size */
pool pool_new(int size);
/* wrapper around malloc, takes from the pool, cleaned up automatically */
void *pmalloc(pool p, int size);
/* Wrapper around pmalloc which prefils buffer with c */
void *pmalloc_x(pool p, int size, char c);
/* YAPW for zeroing the block */
void *pmalloco(pool p, int size);
/* wrapper around strdup, gains mem from pool */
char *pstrdup(pool p, const char *src);
/* temp stub */
char *pstrdupx(pool p, const char *src);
/* calls f(arg) before the pool is freed during cleanup */
void pool_cleanup(pool p, pool_cleaner f, void *arg);
/* calls the cleanup functions, frees all the data on the pool, and deletes the pool itself */
void pool_free(pool p);
/* returns total bytes allocated in this pool */
int pool_size(pool p);

#endif /* WAN_POOL_H */
