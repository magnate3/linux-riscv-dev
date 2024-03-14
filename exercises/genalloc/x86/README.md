
# run

```
 uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
```

```
root@ubuntux86:/work/kernel_learn/genalloc_test# insmod  genalloc_test.ko 
root@ubuntux86:/work/kernel_learn/genalloc_test# insmod  genalloc_test2.ko 
root@ubuntux86:/work/kernel_learn/genalloc_test# insmod  genalloc_test3.ko 
root@ubuntux86:/work/kernel_learn/genalloc_test# insmod  genalloc_test4.ko 
root@ubuntux86:/work/kernel_learn/genalloc_test# lsmod | grep genalloc_test
genalloc_test4         20480  0
genalloc_test3         16384  0
genalloc_test2         16384  0
genalloc_test          16384  0
root@ubuntux86:/work/kernel_learn/genalloc_test# 
```




gen_pool_add_owner采用vzalloc_node分配struct gen_pool_chunk内存  
gen_pool_destroy应该采用vfree释放struct gen_pool_chunk内存    
```
// gen_pool_add -->  gen_pool_add_virt  --> gen_pool_add_owner use vzalloc_node
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
                // use vzalloc_node
                vfree(chunk);
                //kfree(chunk);
        }
        //kfree_const(pool->name);
        kfree(pool);
        return;
}
```