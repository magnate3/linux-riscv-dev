/*
 *ktask_cache.h: 2019-02-21 created by qudreams
 *
 *create kmem_cache for kernel task cmdline
 */

#ifndef KTASK_CACHE_H
#define KTASK_CACHE_H

#define MAX_CACHE_SIZE 4096
struct kmem_cache;

struct kmem_cache*
ktask_get_cache(unsigned size);

int ktask_cache_init(void);
void ktask_cache_uninit(void);

#endif
