#include <linux/module.h>
#include <linux/slab.h>
#include <linux/genalloc.h>
#include <linux/kernel.h>
#include <linux/gfp.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
static struct gen_pool *sram_pool;
struct page * pages = NULL;
void gen_pool_destroy_test(struct gen_pool *pool)
{
	struct list_head *_chunk, *_next_chunk;
	struct gen_pool_chunk *chunk;
	int order = pool->min_alloc_order;
	int bit, end_bit;
	 printk("%s! \n",__func__);
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
int gen_pool_add_virt_test(struct gen_pool *pool, unsigned long virt, phys_addr_t phys,
                 size_t size, int nid)
{
        struct gen_pool_chunk *chunk;
        int nbits = size >> pool->min_alloc_order;
        int nbytes = sizeof(struct gen_pool_chunk) +
                                BITS_TO_LONGS(nbits) * sizeof(long);
        unsigned long *p, addr;
        pr_info("min alloc order %d ,nbits  %d and BITS_TO_LONGS(nbits) %ld \n", pool->min_alloc_order, nbits, BITS_TO_LONGS(nbits));
        pr_info("nbytes %d and sizeof(struct gen_pool_chunk) %ld,sizeof(long) %ld, BITS_TO_LONGS* sizeof(long) %ld \n", nbytes, sizeof(struct gen_pool_chunk),sizeof(long), BITS_TO_LONGS(nbits) * sizeof(long));
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
        p = chunk->bits;
        while(1)
        {
            addr = (unsigned long)p & 0xF;
            if(8 == addr || 0 == addr)
            {
                 pr_info("8 aligened addr %p \n", p);
                 *p  = 0xff;
                 *p  = 0x0;
                 break;
            }
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
/* 初始化内存池，需要创建以及加入内存块，参数为：起始地址、大小、最小分配阶数 */
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
                pr_info("chunk->end_addr 0x%lx -- chunk->start_addr 0x%lx, bit < end_bit: %d \n", chunk->end_addr, chunk->start_addr, bit < end_bit ? 1 : 0);
	}
	//write_unlock(&pool->lock);
	return;
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
    page_total = 1<<order;
    total = PAGE_SIZE*page_total; 
   
    pr_info("genalloc test begin >>>>>>>>>>>>>>>>>  \n");
    pages = alloc_pages(GFP_KERNEL,order);  //分配1<<order个物理页
    if(! pages)
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
    //gen_pool_set_algo(sram_pool, gen_pool_best_fit, NULL);
    nbits = total >> sram_pool->min_alloc_order;
    pr_info("num of long  %d \n",nbits);
    gen_pool_info_print(sram_pool);
    while(1)
    {
         
         addr = gen_pool_alloc(sram_pool, obj_size);
         if (!addr) {
             pr_info("gen pool alloc falied \n");
             break;
         }
         else {
              gen_pool_info_print(sram_pool);
              index = 0;
              pr_info("gen pool alloc addr %p \n",(void *) addr);
#if 1
              while(index < obj_size)
              {
                   memcpy((void*)addr + index, &ch,1);
                   ++index;
              } 
#endif
	      gen_pool_free(sram_pool, addr, obj_size);
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

static void __exit my_exit(void)
{
}
module_init(my_init);
module_exit(my_exit);

MODULE_AUTHOR("Jerry Cooperstein");
MODULE_DESCRIPTION("LF331:1.6 s_18/lab8_uio_api.c");
MODULE_LICENSE("GPL v2");
