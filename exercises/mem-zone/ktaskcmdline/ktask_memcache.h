#ifndef KTASK_MEM_CACHE_H
#define KTASK_MEM_CACHE_H

#include <linux/slab.h>

struct kmem_cache *ktask_mem_cache_create(const char* name,size_t size, size_t align);
#define ktask_mem_cache_destroy kmem_cache_destroy
#define ktask_mem_cache_zalloc kmem_cache_zalloc
#define ktask_mem_cache_free   kmem_cache_free
#endif
