/*
 * SLAB injection module.
 *
 * Author: Maxim Menshchikov <MaximMenshchikov@gmail.com>
 */
#include <linux/module.h>
#include <linux/kernel.h>

#include <linux/init.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/errno.h>
#include <asm/page.h>

#ifdef CONFIG_SLAB
#include <linux/slab_def.h>
#endif

#ifdef CONFIG_SLUB
#include <linux/slub_def.h>
#endif

static char *slab = NULL;
module_param(slab, charp, 0000);
MODULE_PARM_DESC(slab, "Target SLAB cache name");

unsigned int active_objs = 0;
module_param(active_objs, uint, 0);
MODULE_PARM_DESC(active_objs, "Number of active objects in SLAB");

unsigned int num_objs = 0;
module_param(num_objs, uint, 0);
MODULE_PARM_DESC(num_objs, "Total Number of objects in SLAB");

/**
 * find_cache - find cache in the list of caches by name.
 *
 * @param slab_caches: SLAB cache list head.
 * @param name:        Cache name.
 * @return struct kmem_cache * pointer if cache is found, NULL otherwise.
 */
static struct kmem_cache *
find_cache(struct list_head *slab_caches,
           const char *name)
{
    struct kmem_cache *c = NULL;

    list_for_each_entry(c, slab_caches, list)
    {
        pr_info("kmem_cache name:%s \n", c->name);
        if (strcmp(c->name, name) == 0)
            return c;
    }

    return NULL;
}

/**
 * Free objects allocated by prefill_cache.
 *
 * @param cache: Cache to free objects from.
 * @param objs: Pointer to array of pointers ending with NULL.
 */
static void
free_prefilled_objects(struct kmem_cache *cache, void **objs)
{
    void **p = objs;

    while (*p != NULL)
    {
        kmem_cache_free(cache, *p);
        p++;
    }
    kfree(objs);
}

/**
 * Prefill the cache to compensate the difference between total and active
 * objects.
 *
 * @param cache: Cache to prefill.
 * @return Pointer to array of pointers ending with NULL.
 */
static void **
prefill_cache(struct kmem_cache *cache)
{
    void **objs;
    int    i;

    objs = kzalloc(sizeof(void *) * (num_objs - active_objs + 1),
                   GFP_KERNEL);

    for (i = 0; i < (num_objs - active_objs); ++i)
    {
        objs[i] = kmem_cache_alloc(cache, GFP_KERNEL);
        if (objs[i] == NULL)
        {
            free_prefilled_objects(cache, objs);
            kfree(objs);
            return NULL;
        }
       else 
       {
 
            if (0 == i)
            {
                  pr_info("kmem_cache name: %s, objs[i]->name: %s \n", cache->name,strcmp(cache->name, "kmem_cache") == 0  ? ((struct kmem_cache*)objs[i])->name : "other");
            }
       }
    }

    return objs;
}

/**
 * Inject single page to cache.
 *
 * @param cache: Target cache.
 * @return Status code.
 */
static int
inject_cache_page(struct kmem_cache *cache)
{
    int     err;
    int     i;
    int     n;
    void  **objs;
    int     boundary;

    /*
     * Calculate the number of objects and the 'boundary', i.e. the number of
     * objects to free up.
     */
    n = PAGE_SIZE / cache->size;
    boundary = n - 1;

    objs = kzalloc(sizeof(void *) * n, GFP_KERNEL);
    if (objs == NULL)
        return -ENOMEM;

    for (i = 0; i < n; ++i)
    {
        objs[i] = kmem_cache_alloc(cache, GFP_KERNEL);
        if (objs[i] == NULL)
        {
            printk(KERN_ERR "Failed to allocate object for SLAB injection");
            /* Change boundary to 'all' objects */
            boundary = n;
            err = -ENOMEM;
            goto clear_objects;
        }
    }

    err = 0;

clear_objects:
    for (i = 0; i < boundary; ++i)
    {
        if (objs[i] != NULL)
            kmem_cache_free(cache, objs[i]);
    }

    kfree(objs);
    return err;
}

static int __init
slab_inject_init(void)
{
    struct kmem_cache *test_cache;
    struct kmem_cache *c;
    int                err = -EBUSY;

    if (slab == NULL || strlen(slab) == 0)
    {
        printk(KERN_ERR "No SLAB to fill\n");
        return -EINVAL;
    }

    if (active_objs > num_objs)
    {
        printk(KERN_ERR "Number of objects is invalid\n");
        return -EINVAL;
    }

    test_cache = kmem_cache_create("test_cache",
                                  100, 0, SLAB_PANIC, NULL);
    if (test_cache == NULL)
    {
        printk(KERN_ERR "Couldn't allocate test SLAB\n");
        return -ENOMEM;
    }

    c = find_cache(&test_cache->list, slab);
    if (c != NULL)
    {
        void **prefilled_objs;

        printk(KERN_INFO "Cache: %s\n", c->name);
        printk(KERN_INFO "Object size: %u\n", c->object_size);
        printk(KERN_INFO "Aligned size: %u\n", c->size);
        printk(KERN_INFO "Can fit %lu objects to page (page size=%lu)\n",
               PAGE_SIZE / c->size, PAGE_SIZE);

        if ((PAGE_SIZE / c->size) < 2)
        {
            printk(KERN_INFO "SLAB inject cannot be used for objects "
                             "bigger than half of page");
            err = -EINVAL;
            goto cleanup;
        }

        prefilled_objs = prefill_cache(c);
        if (prefilled_objs == NULL)
        {
            err = -ENOMEM;
            goto cleanup;
        }

        err = inject_cache_page(c);
        if (err != 0)
        {
            printk(KERN_ERR "Couldn't inject a page to cache: %d", err);
        }
        else
        {
            err = 0;
        }

        free_prefilled_objects(c, prefilled_objs);
        kmem_cache_shrink(c);
    }
    else
    {
        printk(KERN_INFO "SLAB '%s' not found\n", slab);
        err = -ENOENT;
    }

cleanup:
    kmem_cache_destroy(test_cache);
    return err;
}

static void __exit slab_inject_exit(void)
{
}

module_init(slab_inject_init);
module_exit(slab_inject_exit);

MODULE_AUTHOR("Maxim Menshchikov <MaximMenshchikov@gmail.com>");
MODULE_LICENSE("MIT");
