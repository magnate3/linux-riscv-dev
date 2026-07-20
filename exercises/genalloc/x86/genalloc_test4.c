#include <linux/module.h>
#include <linux/slab.h>
#include <linux/genalloc.h>
#include <linux/kernel.h>
#include <linux/gfp.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/bitmap.h>
#include <linux/rculist.h>
static struct gen_pool *sram_pool;
struct page * pages = NULL;
static int bitmap_clear_ll(unsigned long *map, int start, int nr);
static int bitmap_set_ll(unsigned long *map, int start, int nr);
int gen_pool_chunk_test(void)
{
        struct gen_pool_chunk *chunk;
        int nbits = 2*1024*1024;
        //int nbits = 64*1024*1024;
        //int nbits = size >> pool->min_alloc_order;
        int nbytes = sizeof(struct gen_pool_chunk) +
                                BITS_TO_LONGS(nbits) * sizeof(long);
         pr_info(" %s test begin >>> \n", __func__);
        pr_info("nbits  %d and BITS_TO_LONGS(nbits) %ld \n", nbits, BITS_TO_LONGS(nbits));
        pr_info("nbytes %d and sizeof(struct gen_pool_chunk) %ld,sizeof(long) %ld, BITS_TO_LONGS* sizeof(long) %ld \n", nbytes, sizeof(struct gen_pool_chunk),sizeof(long), BITS_TO_LONGS(nbits) * sizeof(long));
        chunk = kzalloc_node(nbytes, GFP_KERNEL, -1);
        if (unlikely(chunk == NULL))
                return -ENOMEM;

#if 1
        //chunk->bits = (long unsigned int *)(chunk + 1);
        if (chunk->bits == (char *)chunk + sizeof(struct gen_pool_chunk))
        {
              pr_info("chunk->bits equals chunk + sizeof(struct gen_pool_chunk) \n");
        }
        *((char*)chunk + sizeof(struct gen_pool_chunk)) = 0xff;
        *(chunk->bits)  = 0xff;
#endif

        return 0;
}
int gen_pool_add_virt_test(struct gen_pool *pool, unsigned long virt, phys_addr_t phys,
                 size_t size, int nid)
{
        struct gen_pool_chunk *chunk;
        int nbits = size >> pool->min_alloc_order;
        int nbytes = sizeof(struct gen_pool_chunk) +
                                BITS_TO_LONGS(nbits) * sizeof(long);
        unsigned long *p, addr;
        int index = 0, total = BITS_TO_LONGS(nbits);
        pr_info("min alloc order %d ,nbits  %d and BITS_TO_LONGS(nbits) %ld \n", pool->min_alloc_order, nbits, BITS_TO_LONGS(nbits));
        pr_info("nbytes %d and sizeof(struct gen_pool_chunk) %d,sizeof(long) %ld, BITS_TO_LONGS* sizeof(long) %ld \n", nbytes, sizeof(struct gen_pool_chunk),sizeof(long), BITS_TO_LONGS(nbits) * sizeof(long));
#if 0
        if(nbytes > 32*1024)
        {
           pr_info("kzalloc too much memories \n");
           return -ENOMEM;

        }
#endif
        //chunk = kzalloc(nbytes, GFP_KERNEL);
        chunk = kzalloc_node(nbytes, GFP_KERNEL, nid);
        if (unlikely(chunk == NULL))
                return -ENOMEM;

        chunk->phys_addr = phys;
        chunk->start_addr = virt;
        chunk->end_addr = virt + size - 1;
#if 1
        //chunk->bits = (long unsigned int *)(chunk + 1);
        if (chunk->bits == (char *)chunk + sizeof(struct gen_pool_chunk))
        {
              pr_info("chunk->bits equals chunk + sizeof(struct gen_pool_chunk) \n");
        }
        //*((char *)chunk + sizeof(struct gen_pool_chunk)) = 0xff;
        bitmap_set_ll(chunk->bits, 0, 64);
        bitmap_clear_ll(chunk->bits, 0, 64);
        p = chunk->bits;
        while(index < total)
        {
            addr = (unsigned long)p & 0xF;
#if 0
            if(8 == addr || 0 == addr)
            {
                 pr_info("8 aligened addr %p \n", p);
                 *p  = 0xff;
                 *p = 0;
                 break;
            }
#endif
           ++index;
           *p  = 0xff;
           *p = 0;
           ++ p;
        }
#endif
        atomic_long_set(&chunk->avail, size);

        spin_lock(&pool->lock);
        list_add_rcu(&chunk->next_chunk, &pool->chunks);
        spin_unlock(&pool->lock);

        return 0;
}
static inline int gen_pool_add_test(struct gen_pool *pool, unsigned long addr,
                               size_t size, int nid)
{
        return gen_pool_add_virt_test(pool, addr, -1, size, nid);
}
void gen_pool_destroy_test(struct gen_pool *pool)
{
	struct list_head *_chunk, *_next_chunk;
	struct gen_pool_chunk *chunk;
	int order = pool->min_alloc_order;
	int bit, end_bit;
	//write_lock(&pool->lock);
	list_for_each_safe(_chunk, _next_chunk, &pool->chunks) {
		chunk = list_entry(_chunk, struct gen_pool_chunk, next_chunk);
		list_del(&chunk->next_chunk);
		end_bit = (chunk->end_addr - chunk->start_addr) >> order;
		bit = find_next_bit(chunk->bits, end_bit, 0);
		BUG_ON(bit < end_bit);
		kfree(chunk);
	}
        //kfree_const(pool->name);
	kfree(pool);
	return;
}
static void *mm_init(uint64_t addr, size_t size, uint32_t order)
{
    struct gen_pool *pool;
    // pool->algo = gen_pool_first_fit
    pool = gen_pool_create(order, -1);
    if (pool == NULL) {
        return NULL;
    }

    if (gen_pool_add_test(pool, addr, size, -1) != 0) {
        gen_pool_destroy_test(pool);

        return NULL;
    }

    return pool;
}

void gen_pool_info_print(struct gen_pool *pool)
{
	struct list_head *_chunk, *_next_chunk;
	struct gen_pool_chunk *chunk;
	int order = pool->min_alloc_order;
	int bit, end_bit;
	//write_lock(&pool->lock);
	list_for_each_safe(_chunk, _next_chunk, &pool->chunks) {
		chunk = list_entry(_chunk, struct gen_pool_chunk, next_chunk);
#if 0
		list_del(&chunk->next_chunk);
#endif
		end_bit = (chunk->end_addr - chunk->start_addr) >> order;
		bit = find_next_bit(chunk->bits, end_bit, 0);
		//BUG_ON(bit < end_bit);
                pr_info("chunk->end_addr 0x%lx -- chunk->start_addr 0x%lx \n", chunk->end_addr, chunk->start_addr);
                pr_info("chunk + 1 addr %p and chunk->bits %p \n", chunk + 1, chunk->bits);
#if 1
                //*(chunk->bits) = 0xffffff;
                //chunk->bits[0] = 0xffffff;
#endif
	}
	//write_unlock(&pool->lock);
	return;
}
static inline size_t chunk_size(const struct gen_pool_chunk *chunk)
{
        return chunk->end_addr - chunk->start_addr + 1;
}

static int set_bits_ll(unsigned long *addr, unsigned long mask_to_set)
{
        unsigned long val, nval;

        nval = *addr;
        do {
                val = nval;
                if (val & mask_to_set)
#if 1
		{
			pr_info("mask_to_clear %x, val %x  \n", mask_to_set, val);
                        return -EBUSY;
		}
#else
                        return -EBUSY;
#endif
                cpu_relax();
        } while ((nval = cmpxchg(addr, val, val | mask_to_set)) != val);

        return 0;
}

static int clear_bits_ll(unsigned long *addr, unsigned long mask_to_clear)
{
        unsigned long val, nval;

        nval = *addr;
        do {
                val = nval;
                if ((val & mask_to_clear) != mask_to_clear)
#if 1
		{
			pr_info("mask_to_clear %x, val %x  \n", mask_to_clear, val);
                        return -EBUSY;
		}
#else
                        return -EBUSY;
#endif
                cpu_relax();
        } while ((nval = cmpxchg(addr, val, val & ~mask_to_clear)) != val);

        return 0;
}

/*
 *  * bitmap_set_ll - set the specified number of bits at the specified position
 *   * @map: pointer to a bitmap
 *    * @start: a bit position in @map
 *     * @nr: number of bits to set
 *      *
 *       * Set @nr bits start from @start in @map lock-lessly. Several users
 *        * can set/clear the same bitmap simultaneously without lock. If two
 *         * users set the same bit, one user will return remain bits, otherwise
 *          * return 0.
 *           */
static int bitmap_set_ll(unsigned long *map, int start, int nr)
{
        unsigned long *p = map + BIT_WORD(start);
        const int size = start + nr;
        int bits_to_set = BITS_PER_LONG - (start % BITS_PER_LONG);
        unsigned long mask_to_set = BITMAP_FIRST_WORD_MASK(start);

        while (nr - bits_to_set >= 0) {
                if (set_bits_ll(p, mask_to_set))
                        return nr;
                nr -= bits_to_set;
                bits_to_set = BITS_PER_LONG;
                mask_to_set = ~0UL;
                p++;
        }
        if (nr) {
                mask_to_set &= BITMAP_LAST_WORD_MASK(size);
                if (set_bits_ll(p, mask_to_set))
                        return nr;
        }

        return 0;
}
/*
 *  * bitmap_clear_ll - clear the specified number of bits at the specified position
 *   * @map: pointer to a bitmap
 *    * @start: a bit position in @map
 *     * @nr: number of bits to set
 *      *
 *       * Clear @nr bits start from @start in @map lock-lessly. Several users
 *        * can set/clear the same bitmap simultaneously without lock. If two
 *         * users clear the same bit, one user will return remain bits,
 *          * otherwise return 0.
 *           */
static int bitmap_clear_ll(unsigned long *map, int start, int nr)
{
        unsigned long *p = map + BIT_WORD(start);
        const int size = start + nr;
        int bits_to_clear = BITS_PER_LONG - (start % BITS_PER_LONG);
        unsigned long mask_to_clear = BITMAP_FIRST_WORD_MASK(start);

        while (nr - bits_to_clear >= 0) {
                if (clear_bits_ll(p, mask_to_clear))
                        return nr;
                nr -= bits_to_clear;
                bits_to_clear = BITS_PER_LONG;
                mask_to_clear = ~0UL;
                p++;
        }
        if (nr) {
                mask_to_clear &= BITMAP_LAST_WORD_MASK(size);
                if (clear_bits_ll(p, mask_to_clear))
                        return nr;
        }

        return 0;
}
/**
 *  * gen_pool_alloc_algo - allocate special memory from the pool
 *   * @pool: pool to allocate from
 *    * @size: number of bytes to allocate from the pool
 *     * @algo: algorithm passed from caller
 *      * @data: data passed to algorithm
 *       *
 *        * Allocate the requested number of bytes from the specified pool.
 *         * Uses the pool allocation function (with first-fit algorithm by default).
 *          * Can not be used in NMI handler on architectures without
 *           * NMI-safe cmpxchg implementation.
 *            */
#if 0
unsigned long gen_pool_alloc_algo_test(struct gen_pool *pool, size_t size,
                genpool_algo_t algo, void *data)
{
        struct gen_pool_chunk *chunk;
        unsigned long addr = 0;
        int order = pool->min_alloc_order;
        int nbits, start_bit, end_bit, remain;

#ifndef CONFIG_ARCH_HAVE_NMI_SAFE_CMPXCHG
        BUG_ON(in_nmi());
#endif

        if (size == 0)
                return 0;

        nbits = (size + (1UL << order) - 1) >> order;
        rcu_read_lock();
        list_for_each_entry_rcu(chunk, &pool->chunks, next_chunk) {
                if (size > atomic_long_read(&chunk->avail))
                        continue;

                start_bit = 0;
                end_bit = chunk_size(chunk) >> order;
retry:
                start_bit = algo(chunk->bits, end_bit, start_bit,
                                 nbits, data, pool);
#if 0
                if (start_bit >= end_bit)
                        continue;
                remain = bitmap_set_ll(chunk->bits, start_bit, nbits);
                if (remain) {
                        remain = bitmap_clear_ll(chunk->bits, start_bit,
                                                 nbits - remain);
                        BUG_ON(remain);
                        goto retry;
                }
#else
                remain = bitmap_set_ll(chunk->bits, start_bit, nbits);
                pr_err("nbits %d, remain %d, nbits - remain: %d \n",  nbits, remain, nbits - remain);
                remain = bitmap_clear_ll(chunk->bits, start_bit, nbits);
                pr_err("nbits %d, remain %d, nbits - remain: %d \n",  nbits, remain, nbits - remain);
#endif
                addr = chunk->start_addr + ((unsigned long)start_bit << order);
                size = nbits << order;
                atomic_long_sub(size, &chunk->avail);
                break;
        }
        rcu_read_unlock();
        return addr;
}
#else
unsigned long gen_pool_alloc_algo_test(struct gen_pool *pool, size_t size,
                genpool_algo_t algo, void *data)
{
        struct gen_pool_chunk *chunk;
        unsigned long addr = 0;
        int order = pool->min_alloc_order;
        int nbits, start_bit, end_bit, remain;

#define  CONFIG_ARCH_HAVE_NMI_SAFE_CMPXCHG
#ifndef CONFIG_ARCH_HAVE_NMI_SAFE_CMPXCHG
        BUG_ON(in_nmi());
#endif

        if (size == 0)
                return 0;

        //pr_info("size %ld \n",  size );
        nbits = (size + (1UL << order) - 1) >> order;
        rcu_read_lock();
#if 1
        list_for_each_entry_rcu(chunk, &pool->chunks, next_chunk) {
#else

	struct list_head *_chunk, *_next_chunk;
	//write_lock(&pool->lock);
	list_for_each_safe(_chunk, _next_chunk, &pool->chunks) {
		chunk = list_entry(_chunk, struct gen_pool_chunk, next_chunk);
		//list_del(&chunk->next_chunk);
#endif
                pr_err("size %ld, avail  %d , bits addr %p and val %lx \n",  size , chunk->avail, chunk->bits, chunk->bits[0]);
                if (size > atomic_long_read(&chunk->avail))
                        continue;

                start_bit = 0;
                end_bit = chunk_size(chunk) >> order;
retry:
                start_bit = algo(chunk->bits, end_bit, start_bit,
                                 nbits, data, pool,chunk->start_addr);
                pr_info("start_bit %d, end_bit %d, nbits %d \n", start_bit , end_bit, nbits);
                if (start_bit >= end_bit)
                        continue;
#if 1
                remain = bitmap_set_ll(chunk->bits, start_bit, nbits);
                pr_err("start_bit %d, nbits %d, remain %d, nbits - remain: %d \n",start_bit,  nbits, remain, nbits - remain);
                //remain = bitmap_clear_ll(chunk->bits, start_bit, nbits);
                if (remain) {
                        remain = bitmap_clear_ll(chunk->bits, start_bit,
                                                 nbits - remain);
                        BUG_ON(remain);
                        goto retry;
                }
#else
#endif
                pr_info("find chunk \n");
                addr = chunk->start_addr + ((unsigned long)start_bit << order);
                size = nbits << order;
                atomic_long_sub(size, &chunk->avail);
                break;
        }
        rcu_read_unlock();
        return addr;
}
#endif
void gen_pool_free_owner_test(struct gen_pool *pool, unsigned long addr, size_t size,
		void **owner)
{
	struct gen_pool_chunk *chunk;
	int order = pool->min_alloc_order;
	unsigned long start_bit, nbits, remain;

#ifndef CONFIG_ARCH_HAVE_NMI_SAFE_CMPXCHG
	BUG_ON(in_nmi());
#endif

	if (owner)
		*owner = NULL;

	nbits = (size + (1UL << order) - 1) >> order;
	rcu_read_lock();
	list_for_each_entry_rcu(chunk, &pool->chunks, next_chunk) {
		if (addr >= chunk->start_addr && addr <= chunk->end_addr) {
			BUG_ON(addr + size - 1 > chunk->end_addr);
			start_bit = (addr - chunk->start_addr) >> order;
			remain = bitmap_clear_ll(chunk->bits, start_bit, nbits);
                        pr_err("start_bits %ld, nbits %ld, remain %d, nbits - remain: %ld \n", start_bit, nbits, remain, nbits - remain);
			//BUG_ON(remain);
			size = nbits << order;
			atomic_long_add(size, &chunk->avail);
			if (owner)
				*owner = chunk->owner;
			rcu_read_unlock();
			return;
		}
	}
	rcu_read_unlock();
	BUG();
}
/* 销毁内存池 */
static void mm_exit(void)
{
 
    if(sram_pool)
     {
        gen_pool_destroy_test(sram_pool);
     }
}
/* 分配函数 */

/* 释放函数 */
//    gen_pool_free(handle, addr, size);
//      va = gen_pool_alloc_algo(genpool, s, gen_pool_first_fit_align, &data);
//      paddr = gen_pool_virt_to_phys(genpool, va);
int order = 0;
int page_total =0;
#if 1
static int __init my_init(void)
{
    //char * dst;
    // PAGE_SHIFT
    int mini_order = 4; 
    const size_t obj_size = (1<<mini_order)*4;
    unsigned long addr = 0;
    size_t total ; 
    size_t index = 0;
    int nbits;
    char ch = '\0';
    struct gen_pool_chunk *chunk;
    page_total = 1<<order;
    total = PAGE_SIZE*page_total; 
   
    pr_info("genalloc test begin >>>>>>>>>>>>>>>>>  \n");
    pages = alloc_pages(GFP_KERNEL,order);  //分配1<<order个物理页
    if(!pages)
    {
        return -ENOMEM;
    }
    else
    {
        printk("alloc_pages Successfully! \n");
        printk("page_address(pages) = 0x%lx\n", (unsigned long)page_address(pages));
    }
    sram_pool = mm_init((uint64_t)page_address(pages),total, mini_order);
    if(!sram_pool)
    {
         pr_info("sram pool init fail \n");
         return 0;
    } 
    nbits = total >> sram_pool->min_alloc_order;
    pr_info("num of long  %d \n",nbits);
    gen_pool_set_algo(sram_pool, gen_pool_best_fit, NULL);
    gen_pool_info_print(sram_pool);
	//write_lock(&pool->lock);
    if(!sram_pool->algo)
    {
         pr_info("sram pool algo is NULL\n");
         return 0;
    } 
    while(1)
    {
         
         //addr = gen_pool_alloc(sram_pool, obj_size);
         addr = gen_pool_alloc_algo_test(sram_pool, obj_size, sram_pool->algo, sram_pool->data);
         if (!addr) {
             pr_info("gen pool alloc falied \n");
             break;
         }
         else {
              gen_pool_info_print(sram_pool);
              index = 0;
              pr_info("gen pool alloc addr %p \n",(void *) addr);
#if 0
              while(index < obj_size)
              {
                   memcpy((void*)addr + index, &ch,1);
                   ++index;
              } 
#endif 
              gen_pool_free_owner_test(sram_pool, addr, obj_size,NULL);
         }
         break;
    }
    if(pages)
    {
        __free_pages(pages,order);    //释放所分配的1<<order个页
        printk("__free_pages ok! \n");
    }
    mm_exit();
    return 0;
}
#else
static int __init my_init(void)
{
    gen_pool_chunk_test();
    return 0;
}
#endif
static void __exit my_exit(void)
{
}
module_init(my_init);
module_exit(my_exit);

MODULE_AUTHOR("Jerry Cooperstein");
MODULE_DESCRIPTION("LF331:1.6 s_18/lab8_uio_api.c");
MODULE_LICENSE("GPL v2");
