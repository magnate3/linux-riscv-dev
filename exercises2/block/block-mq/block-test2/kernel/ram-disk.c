/*
 * SO2 - Block device drivers lab (#7)
 * Linux - Exercise #1, #2, #3, #6 (RAM Disk)
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>

#include <linux/genhd.h>
#include <linux/fs.h>
#include <linux/blkdev.h>
#include <linux/blk_types.h>
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/bio.h>
#include <linux/vmalloc.h>

MODULE_DESCRIPTION("Simple RAM Disk");
MODULE_AUTHOR("SO2");
MODULE_LICENSE("GPL");


#define KERN_LOG_LEVEL		KERN_ALERT

#define MY_BLOCK_MAJOR		240
#define MY_BLKDEV_NAME		"mybdev"
#define MY_BLOCK_MINORS		1
#define NR_SECTORS		128

#define KERNEL_SECTOR_SIZE	512

/* TODO 6/0: use bios for read/write requests */
#define USE_BIO_TRANSFER	0


typedef __u32 __bitwise blk_opf_t;
enum {
      BLK_MQ_NO_TAG           = -1U,
      BLK_MQ_TAG_MIN          = 1,
      BLK_MQ_TAG_MAX          = BLK_MQ_NO_TAG - 1,
};
static struct my_block_dev {
	struct blk_mq_tag_set tag_set;
	struct request_queue *queue;
	struct gendisk *gd;
	u8 *data;
	size_t size;
} g_dev;
struct blk_mq_ctxs {
	        struct kobject kobj;
		        struct blk_mq_ctx __percpu      *queue_ctx;
};

/**
 *  * struct blk_mq_ctx - State for a software queue facing the submitting CPUs
 *   */
struct blk_mq_ctx {
	        struct {
			                spinlock_t              lock;
			                struct list_head        rq_lists[HCTX_MAX_TYPES];
			 } ____cacheline_aligned_in_smp;

		        unsigned int            cpu;
			unsigned short          index_hw[HCTX_MAX_TYPES];
			struct blk_mq_hw_ctx    *hctxs[HCTX_MAX_TYPES];

					        /* incremented at dispatch time */
			unsigned long           rq_dispatched[2];
			unsigned long           rq_merged;

							        /* incremented at completion time */
			unsigned long           ____cacheline_aligned_in_smp rq_completed[2];

			struct request_queue    *queue;
			struct blk_mq_ctxs      *ctxs;
			struct kobject          kobj;
} ____cacheline_aligned_in_smp;

struct blk_mq_alloc_data {
	        /* input parameter */
	        struct request_queue *q;
	        blk_mq_req_flags_t flags;
	        unsigned int shallow_depth;
	        unsigned int cmd_flags;

					        /* input & output parameter */
	        struct blk_mq_ctx *ctx;
                struct blk_mq_hw_ctx *hctx;
};
 struct blk_mq_tags {
	         unsigned int nr_tags;
		 unsigned int nr_reserved_tags;
                 atomic_t active_queues;
	         struct sbitmap_queue *bitmap_tags;
		 struct sbitmap_queue *breserved_tags;
		 struct sbitmap_queue __bitmap_tags;
	         struct sbitmap_queue __breserved_tags;
	         struct request **rqs;
		 struct request **static_rqs;
		 struct list_head page_list;
 };
static inline struct blk_mq_ctx *__blk_mq_get_ctx(struct request_queue *q,
		                                           unsigned int cpu)
{
	        return per_cpu_ptr(q->queue_ctx, cpu);
}
/*
 *  * This assumes per-cpu software queueing queues. They could be per-node
 *   * as well, for instance. For now this is hardcoded as-is. Note that we don't
 *    * care about preemption, since we know the ctx's are persistent. This does
 *     * mean that we can't rely on ctx always matching the currently running CPU.
 *      */
static inline struct blk_mq_ctx *blk_mq_get_ctx(struct request_queue *q)
{
	        return __blk_mq_get_ctx(q, raw_smp_processor_id());
}
static inline struct blk_mq_tags *blk_mq_tags_from_data(struct blk_mq_alloc_data *data)
{
     if (data->q->elevator)
         return data->hctx->sched_tags;
     return data->hctx->tags;
}
static inline struct blk_mq_hw_ctx *blk_mq_map_queue_type(struct request_queue *q, enum hctx_type type, unsigned int cpu)
{
	        return q->queue_hw_ctx[q->tag_set->map[type].mq_map[cpu]];
}
// refer to blk_mq_submit_bio __blk_mq_alloc_request
// blk_mq_map_swqueue(
static void blk_mq_test_req(struct request*rq)
{
     struct request_queue *q = rq->q;
     struct blk_mq_ctx *ctx = rq->mq_ctx;
     struct blk_mq_hw_ctx *hctx = rq->mq_hctx;
     unsigned int cmd_flags =  rq->cmd_flags ;
     struct blk_mq_ctx *ctx2;
     struct blk_mq_hw_ctx *hctx2;
     int tag;
     int internal_tag;
     unsigned int i, j, hctx_idx;
     struct blk_mq_tag_set *set = q->tag_set;
     struct blk_mq_alloc_data data = {
        .q              = q,
	.flags          = BLK_MQ_REQ_NOWAIT,
	.cmd_flags      = cmd_flags,
	.ctx = ctx,
	.hctx = hctx,
	  };
    struct blk_mq_tags * tags;
    if(blk_mq_get_ctx(q) == ctx)
    {
            pr_err("soft ctx, blk_mq_get_ctx(q) == ctx ? %d \n", blk_mq_get_ctx(q) == ctx);
    }
    for_each_possible_cpu(i) {
	ctx2 = per_cpu_ptr(q->queue_ctx, i);
	if(NULL == ctx2){

           pr_err("soft ctx is null on cpu  %u \n",i);
           continue;																
	}
	for (j = 0; j < set->nr_maps; j++) {
            if (!set->map[j].nr_queues) {
               continue;																
            }
	    hctx_idx = set->map[j].mq_map[i]; // refer to blk_mq_map_queue_type
            //data.hctx = q->queue_hw_ctx[hctx_idx];
	    if(hctx == q->queue_hw_ctx[hctx_idx]){
	            pr_err("queue_hw_ctx [index %u] \n",hctx_idx);
	    }
            hctx2 = blk_mq_map_queue_type(q, j, i);
            if(ctx->hctxs[j] == hctx2){
            pr_err("ctx hctxs [index %u] equal hctx2 \n",j);
	}
      }
    }
        if (q->elevator) {
            //rq->tag = BLK_MQ_NO_TAG;
	    //rq->internal_tag = tag;
	    tag = rq->internal_tag;
            pr_err("the request has the IO  elevator scheduler queue,rq tag == BLK_MQ_NO_TAG? %d,internal_tag %d \n", BLK_MQ_NO_TAG == rq->tag, rq->internal_tag);
	 } else {
	    //rq->tag = tag;
	    //rq->internal_tag = BLK_MQ_NO_TAG;
            pr_err("the request does not have the IO  elevator scheduler queue,rq internal tag == BLK_MQ_NO_TAG? %d, tag %d \n", BLK_MQ_NO_TAG == rq->internal_tag, rq->tag);
	    tag = rq->tag;
	}
    tags =  blk_mq_tags_from_data(&data);
    if(rq == tags->static_rqs[tag]){
        pr_err("staic_rqs[index  %u] euqal rq  \n",tag);
    }
}
static void blk_mq_test(struct request_queue *q)
{
#if 0
       blk_opf_t opf;
       blk_mq_req_flags_t flags;
	        struct blk_mq_alloc_data data = {
			                .q              = q,
					.flags          = flags,
					.cmd_flags      = opf,
					 .nr_tags        = 1,
					  };
#endif
      if(q->elevator){
	    pr_err("the request has the IO  elevator scheduler queue");
      }
}
static int my_block_open(struct block_device *bdev, fmode_t mode)
{
	return 0;
}

static void my_block_release(struct gendisk *gd, fmode_t mode)
{
}

static const struct block_device_operations my_block_ops = {
	.owner = THIS_MODULE,
	.open = my_block_open,
	.release = my_block_release
};

static void my_block_transfer(struct my_block_dev *dev, sector_t sector,
		unsigned long len, char *buffer, int dir)
{
	unsigned long offset = sector * KERNEL_SECTOR_SIZE;

	/* check for read/write beyond end of block device */
	if ((offset + len) > dev->size)
		return;

	/* TODO 3/4: read/write to dev buffer depending on dir */
	if (dir == 1)		/* write */
		memcpy(dev->data + offset, buffer, len);
	else
		memcpy(buffer, dev->data + offset, len);
}

/* to transfer data using bio structures enable USE_BIO_TRANFER */
#if USE_BIO_TRANSFER == 1
static void my_xfer_request(struct my_block_dev *dev, struct request *req)
{
	/* TODO 6/10: iterate segments */
	struct bio_vec bvec;
	struct req_iterator iter;

	rq_for_each_segment(bvec, req, iter) {
		sector_t sector = iter.iter.bi_sector;
		unsigned long offset = bvec.bv_offset;
		size_t len = bvec.bv_len;
		int dir = bio_data_dir(iter.bio);
		char *buffer = kmap_atomic(bvec.bv_page);
		printk(KERN_LOG_LEVEL "%s: buf %8p offset %lu len %u dir %d\n", __func__, buffer, offset, len, dir);

		/* TODO 6/3: copy bio data to device buffer */
		my_block_transfer(dev, sector, len, buffer + offset, dir);
		kunmap_atomic(buffer);
	}
}
#endif

static blk_status_t my_block_request(struct blk_mq_hw_ctx *hctx,
				     const struct blk_mq_queue_data *bd)
{
	struct request *rq;
	struct my_block_dev *dev = hctx->queue->queuedata;

	/* TODO 2: get pointer to request */
	rq = bd->rq;
        blk_mq_test_req(rq);
	/* TODO 2: start request processing. */
	blk_mq_start_request(rq);

	/* TODO 2/5: check fs request. Return if passthrough. */
	if (blk_rq_is_passthrough(rq)) {
		printk(KERN_NOTICE "Skip non-fs request\n");
		blk_mq_end_request(rq, BLK_STS_IOERR);
		goto out;
	}

	/* TODO 2/6: print request information */
	printk(KERN_LOG_LEVEL
		"request received: pos=%llu bytes=%u "
		"cur_bytes=%u dir=%c\n",
		(unsigned long long) blk_rq_pos(rq),
		blk_rq_bytes(rq), blk_rq_cur_bytes(rq),
		rq_data_dir(rq) ? 'W' : 'R');

#if USE_BIO_TRANSFER == 1
	/* TODO 6/1: process the request by calling my_xfer_request */
	my_xfer_request(dev, rq);
#else
	/* TODO 3/3: process the request by calling my_block_transfer */
	my_block_transfer(dev, blk_rq_pos(rq),
			  blk_rq_bytes(rq),
			  bio_data(rq->bio), rq_data_dir(rq));
#endif

	/* TODO 2/1: end request successfully */
	blk_mq_end_request(rq, BLK_STS_OK);

out:
	return BLK_STS_OK;
}

static struct blk_mq_ops my_queue_ops = {
	.queue_rq = my_block_request,
};

static int create_block_device(struct my_block_dev *dev)
{
	int err;

	dev->size = NR_SECTORS * KERNEL_SECTOR_SIZE;
	dev->data = vmalloc(dev->size);
	if (dev->data == NULL) {
		printk(KERN_ERR "vmalloc: out of memory\n");
		err = -ENOMEM;
		goto out_vmalloc;
	}

	/* Initialize tag set. */
	dev->tag_set.ops = &my_queue_ops;
	dev->tag_set.nr_hw_queues = 1;
	dev->tag_set.queue_depth = 128;
	dev->tag_set.numa_node = NUMA_NO_NODE;
	dev->tag_set.cmd_size = 0;
	dev->tag_set.flags = BLK_MQ_F_SHOULD_MERGE;
	err = blk_mq_alloc_tag_set(&dev->tag_set);
	if (err) {
	    printk(KERN_ERR "blk_mq_alloc_tag_set: can't allocate tag set\n");
	    goto out_alloc_tag_set;
	}

	/* Allocate queue. */
	dev->queue = blk_mq_init_queue(&dev->tag_set);
	if (IS_ERR(dev->queue)) {
		printk(KERN_ERR "blk_mq_init_queue: out of memory\n");
		err = -ENOMEM;
		goto out_blk_init;
	}
	blk_queue_logical_block_size(dev->queue, KERNEL_SECTOR_SIZE);
	dev->queue->queuedata = dev;

	/* initialize the gendisk structure */
	dev->gd = alloc_disk(MY_BLOCK_MINORS);
	if (!dev->gd) {
		printk(KERN_ERR "alloc_disk: failure\n");
		err = -ENOMEM;
		goto out_alloc_disk;
	}

	dev->gd->major = MY_BLOCK_MAJOR;
	dev->gd->first_minor = 0;
	dev->gd->fops = &my_block_ops;
	dev->gd->queue = dev->queue;
	dev->gd->private_data = dev;
	snprintf(dev->gd->disk_name, DISK_NAME_LEN, "myblock");
	set_capacity(dev->gd, NR_SECTORS);

	add_disk(dev->gd);

	return 0;

out_alloc_disk:
	blk_cleanup_queue(dev->queue);
out_blk_init:
	blk_mq_free_tag_set(&dev->tag_set);
out_alloc_tag_set:
	vfree(dev->data);
out_vmalloc:
	return err;
}

static int __init my_block_init(void)
{
	int err = 0;

	/* TODO 1/5: register block device */
	err = register_blkdev(MY_BLOCK_MAJOR, MY_BLKDEV_NAME);
	if (err < 0) {
		printk(KERN_ERR "register_blkdev: unable to register\n");
		return err;
	}

	/* TODO 2/3: create block device using create_block_device */
	err = create_block_device(&g_dev);
	if (err < 0)
		goto out;

	return 0;

out:
	/* TODO 2/1: unregister block device in case of an error */
	unregister_blkdev(MY_BLOCK_MAJOR, MY_BLKDEV_NAME);
	return err;
}

static void delete_block_device(struct my_block_dev *dev)
{
	if (dev->gd) {
		del_gendisk(dev->gd);
		put_disk(dev->gd);
	}

	if (dev->queue)
		blk_cleanup_queue(dev->queue);
	if (dev->tag_set.tags)
		blk_mq_free_tag_set(&dev->tag_set);
	if (dev->data)
		vfree(dev->data);
}

static void __exit my_block_exit(void)
{
	/* TODO 2/1: cleanup block device using delete_block_device */
	delete_block_device(&g_dev);

	/* TODO 1/1: unregister block device */
	unregister_blkdev(MY_BLOCK_MAJOR, MY_BLKDEV_NAME);
}

module_init(my_block_init);
module_exit(my_block_exit);
