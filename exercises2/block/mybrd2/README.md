
# 软硬件队列的关系

```C
struct blk_mq_ctx {
        struct {
                spinlock_t              lock;
                struct list_head        rq_list;
        }  ____cacheline_aligned_in_smp;

        unsigned int            cpu;
        unsigned int            index_hw;

        /* incremented at dispatch time */
        unsigned long           rq_dispatched[2];
        unsigned long           rq_merged;

        /* incremented at completion time */
        unsigned long           ____cacheline_aligned_in_smp rq_completed[2];

        struct request_queue    *queue;
        struct kobject          kobj;
} ____cacheline_aligned_in_smp;

static inline struct blk_mq_hw_ctx *blk_mq_map_queue(struct request_queue *q,
                int cpu)
{
        return q->queue_hw_ctx[q->mq_map[cpu]];
}         
// refer to  blk_mq_init_cpu_queues  blk_mq_map_swqueue
static void print_request_queue(struct request_queue *q)
{        
    unsigned int i, hctx_idx;
    //unsigned int nr_queues = set->nr_hw_queues;
    struct blk_mq_ctx *ctx;  
    struct blk_mq_hw_ctx *hctx;
    struct blk_mq_tag_set *set = q->tag_set;
    for_each_possible_cpu(i) {
            hctx_idx = q->mq_map[i];
                    /* unmapped hw queue can be remapped after CPU topo changed */
                //if (!set->tags[hctx_idx] &&
                //    !__blk_mq_alloc_rq_map(set, hctx_idx)) {
                if (!set->tags[hctx_idx]){
                    /*
                   * If tags initialization fail for some hctx,
                   * that hctx won't be brought online.  In this
                   * case, remap the current ctx to hctx[0] which
                   * is guaranteed to always have tags allocated
                   */
                        q->mq_map[i] = 0;
                }
            hctx = blk_mq_map_queue(q, i);
            ctx = per_cpu_ptr(q->queue_ctx, i);

    }
}
static void mybrd_free(struct mybrd_device *mybrd)
{
        print_request_queue(global_mybrd->mybrd_queue);
        blk_cleanup_queue(global_mybrd->mybrd_queue);
        kfree(global_mybrd);
}
```

# rmmod  mybrd_test.ko
```
[root@centos7 mybrd]# insmod  mybrd_test.ko 
[root@centos7 mybrd]# rmmod  mybrd_test.ko 
```

成功运行