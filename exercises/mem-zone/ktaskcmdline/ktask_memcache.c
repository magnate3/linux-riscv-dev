#include <linux/types.h>
#include <linux/version.h>
#include "ktask_memcache.h"


struct kmem_cache *ktask_mem_cache_create(const char* name,size_t size, size_t align)
{
    struct kmem_cache* pcache = NULL;

    //the flags must be 0,or else will occur system-dump on Neokylin 6.5
    //but I don't know why
    #if LINUX_VERSION_CODE > KERNEL_VERSION(2,6,23)
        pcache = kmem_cache_create(name,size,align,0,
                                    NULL);
    #else
        pcache = kmem_cache_create(name,size,align,0,
                            NULL,NULL);
    #endif

    return pcache;
}
