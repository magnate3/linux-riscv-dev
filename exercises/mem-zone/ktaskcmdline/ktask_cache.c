#include <linux/slab.h>
#include <linux/string.h>
#include <linux/stddef.h>
#include "ktask_cache.h"
#include "ktask_memcache.h"

#define MIN_CACHE_SIZE 128

typedef struct {
    const char* name;
    size_t size;
    struct kmem_cache* cachep;
} ktask_cache_t;


static ktask_cache_t cache_array[] = {
        {"ktask_cache128",MIN_CACHE_SIZE,NULL},
        {"ktask_cache256",MIN_CACHE_SIZE * 2,NULL},
        {"ktask_cache512",MIN_CACHE_SIZE * 4,NULL},
        {"ktask_cache1024",MIN_CACHE_SIZE * 8,NULL},
        {"ktask_cache2048",MIN_CACHE_SIZE * 16,NULL},
        {"ktask_cache4096",MAX_CACHE_SIZE,NULL}
    };


int ktask_cache_init(void)
{
    int rc = 0;
    size_t i = 0;
    size_t size = 0;
    struct kmem_cache* cachep = NULL;

    size = ARRAY_SIZE(cache_array);
    for(i = 0;i < size;i++) {
        cachep = ktask_mem_cache_create(cache_array[i].name,
                                cache_array[i].size,0);
        if(!cachep) { rc = -ENOMEM; break; }
        cache_array[i].cachep = cachep;
    }

    if(rc) { ktask_cache_uninit(); }
    return rc;
}

void ktask_cache_uninit(void)
{
    size_t i = 0;
    size_t size = 0;
    struct kmem_cache* cachep = NULL;

    size = ARRAY_SIZE(cache_array);
    for(i = 0;i < size;i++) {
        cachep = cache_array[i].cachep;
        if(!cachep) { continue; }

        ktask_mem_cache_destroy(cachep);
        cache_array[i].cachep = NULL;
    }
}

static int calc_cache_idx(unsigned size)
{
    int i = -1;
    int max = 0;
    size_t mid = 0;
    size_t arr_size = 0;

    arr_size = ARRAY_SIZE(cache_array);

    mid = arr_size / 2;
    i = mid;
    if((size > cache_array[mid - 1].size) &&
        (size <= cache_array[mid].size))
    {
        return i;
    }

    i = 0;
    max = mid;
    if(cache_array[mid].size < size) {
        i = mid + 1;
        max = arr_size;
    }

    while(i < max) {
        if(cache_array[i].size >= size) { break; }
        i++;
    }
    return i;
}

struct kmem_cache*
ktask_get_cache(unsigned size)
{
    int idx = 0;
    struct kmem_cache* cachep = NULL;

    if(size > MAX_CACHE_SIZE) {
        return cachep;
    }

    idx = calc_cache_idx(size);
    cachep = cache_array[idx].cachep;

    return cachep;
}
