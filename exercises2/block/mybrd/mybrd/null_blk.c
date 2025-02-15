#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/blkdev.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/blk-mq.h>
#include <linux/hrtimer.h>
#include <linux/smp.h>

struct nullb_cmd {
	struct list_head list;
	struct llist_node ll_list;
	call_single_data_t csd;
	//struct call_single_data csd;
	struct request *rq;
	struct bio *bio;
	unsigned int tag;
	struct nullb_queue *nq;
};

struct nullb_queue {
	unsigned long *tag_map;
	wait_queue_head_t wait;
	unsigned int queue_depth;

	struct nullb_cmd *cmds;
};

struct nullb {
	struct list_head list;
	unsigned int index;
	struct request_queue *q;
	struct gendisk *disk;
	struct hrtimer timer;
	unsigned int queue_depth;
	spinlock_t lock;

	struct nullb_queue *queues;
	unsigned int nr_queues;
};

static LIST_HEAD(nullb_list);
static struct mutex lock;
static int null_major;
static int nullb_indexes;

struct completion_queue {
	struct llist_head list;
	struct hrtimer timer;
};

/*
 * These are per-cpu for now, they will need to be configured by the
 * complete_queues parameter and appropriately mapped.
 */
static DEFINE_PER_CPU(struct completion_queue, completion_queues);

enum {
	NULL_IRQ_NONE		= 0,
	NULL_IRQ_SOFTIRQ	= 1,
	NULL_IRQ_TIMER		= 2,

	NULL_Q_BIO		= 0,
	NULL_Q_RQ		= 1,
	NULL_Q_MQ		= 2,
};

static int submit_queues = 1;
module_param(submit_queues, int, S_IRUGO);
MODULE_PARM_DESC(submit_queues, "Number of submission queues");

static int home_node = NUMA_NO_NODE;
module_param(home_node, int, S_IRUGO);
MODULE_PARM_DESC(home_node, "Home node for the device");

static int queue_mode = NULL_Q_MQ;
module_param(queue_mode, int, S_IRUGO);
MODULE_PARM_DESC(use_mq, "Use blk-mq interface (0=bio,1=rq,2=multiqueue)");

static int gb = 250;
module_param(gb, int, S_IRUGO);
MODULE_PARM_DESC(gb, "Size in GB");

static int bs = 512;
module_param(bs, int, S_IRUGO);
MODULE_PARM_DESC(bs, "Block size (in bytes)");

static int nr_devices = 2;
module_param(nr_devices, int, S_IRUGO);
MODULE_PARM_DESC(nr_devices, "Number of devices to register");

static int irqmode = NULL_IRQ_SOFTIRQ;
module_param(irqmode, int, S_IRUGO);
MODULE_PARM_DESC(irqmode, "IRQ completion handler. 0-none, 1-softirq, 2-timer");

static int completion_nsec = 10000;
module_param(completion_nsec, int, S_IRUGO);
MODULE_PARM_DESC(completion_nsec, "Time in ns to complete a request in hardware. Default: 10,000ns");

static int hw_queue_depth = 64;
module_param(hw_queue_depth, int, S_IRUGO);
MODULE_PARM_DESC(hw_queue_depth, "Queue depth for each hardware queue. Default: 64");

static bool use_per_node_hctx = true;
module_param(use_per_node_hctx, bool, S_IRUGO);
MODULE_PARM_DESC(use_per_node_hctx, "Use per-node allocation for hardware context queues. Default: true");

static void put_tag(struct nullb_queue *nq, unsigned int tag)
{
	clear_bit_unlock(tag, nq->tag_map);

	if (waitqueue_active(&nq->wait))
		wake_up(&nq->wait);
}

static unsigned int get_tag(struct nullb_queue *nq)
{
	unsigned int tag;

	do {
		tag = find_first_zero_bit(nq->tag_map, nq->queue_depth);
		if (tag >= nq->queue_depth)
			return -1U;
	} while (test_and_set_bit_lock(tag, nq->tag_map));

	return tag;
}

static void free_cmd(struct nullb_cmd *cmd)
{
	put_tag(cmd->nq, cmd->tag);
}

static struct nullb_cmd *__alloc_cmd(struct nullb_queue *nq)
{
	struct nullb_cmd *cmd;
	unsigned int tag;

	tag = get_tag(nq);
	if (tag != -1U) {
		cmd = &nq->cmds[tag];
		cmd->tag = tag;
		cmd->nq = nq;
		return cmd;
	}

	return NULL;
}

static struct nullb_cmd *alloc_cmd(struct nullb_queue *nq, int can_wait)
{
	struct nullb_cmd *cmd;
	DEFINE_WAIT(wait);

	cmd = __alloc_cmd(nq);
	if (cmd || !can_wait)
		return cmd;

	do {
		prepare_to_wait(&nq->wait, &wait, TASK_UNINTERRUPTIBLE);
		cmd = __alloc_cmd(nq);
		if (cmd)
			break;

		io_schedule();
	} while (1);

	finish_wait(&nq->wait, &wait);
	return cmd;
}

static void end_cmd(struct nullb_cmd *cmd)
{
	if (cmd->rq) {
		if (queue_mode == NULL_Q_MQ)
			blk_mq_end_io(cmd->rq, 0);
		else {
			INIT_LIST_HEAD(&cmd->rq->queuelist);
			blk_end_request_all(cmd->rq, 0);
		}
	} else if (cmd->bio)
		bio_endio(cmd->bio, 0);

	if (queue_mode != NULL_Q_MQ)
		free_cmd(cmd);
}

static enum hrtimer_restart null_cmd_timer_expired(struct hrtimer *timer)
{
	struct completion_queue *cq;
	struct llist_node *entry;
	struct nullb_cmd *cmd;

	cq = &per_cpu(completion_queues, smp_processor_id());

	while ((entry = llist_del_all(&cq->list)) != NULL) {
		do {
			cmd = container_of(entry, struct nullb_cmd, ll_list);
			end_cmd(cmd);
			entry = entry->next;
		} while (entry);
	}

	return HRTIMER_NORESTART;
}

static void null_cmd_end_timer(struct nullb_cmd *cmd)
{
	struct completion_queue *cq = &per_cpu(completion_queues, get_cpu());

	cmd->ll_list.next = NULL;
	if (llist_add(&cmd->ll_list, &cq->list)) {
		ktime_t kt = ktime_set(0, completion_nsec);

		hrtimer_start(&cq->timer, kt, HRTIMER_MODE_REL);
	}

	put_cpu();
}

static void null_ipi_cmd_end_io(void *data)
{
	struct completion_queue *cq;
	struct llist_node *entry, *next;
	struct nullb_cmd *cmd;

	cq = &per_cpu(completion_queues, smp_processor_id());

	entry = llist_del_all(&cq->list);

	while (entry) {
		next = entry->next;
		cmd = llist_entry(entry, struct nullb_cmd, ll_list);
		end_cmd(cmd);
		entry = next;
	}
}

static void null_softirq_done_fn(struct request *rq)
{
	blk_end_request_all(rq, 0);
}

static void null_cmd_end_ipi(struct nullb_cmd *cmd)
{
	struct call_single_data *data = &cmd->csd;
	int cpu = get_cpu();
	struct completion_queue *cq = &per_cpu(completion_queues, cpu);

	cmd->ll_list.next = NULL;

	if (llist_add(&cmd->ll_list, &cq->list)) {
		data->func = null_ipi_cmd_end_io;
		data->flags = 0;
		__smp_call_function_single(cpu, data, 0);
	}

	put_cpu();
}

static inline void null_handle_cmd(struct nullb_cmd *cmd)
{
	/* Complete IO by inline, softirq or timer */
	switch (irqmode) {
	case NULL_IRQ_NONE:
		end_cmd(cmd);
		break;
	case NULL_IRQ_SOFTIRQ:
		null_cmd_end_ipi(cmd);
		break;
	case NULL_IRQ_TIMER:
		null_cmd_end_timer(cmd);
		break;
	}
}

static struct nullb_queue *nullb_to_queue(struct nullb *nullb)
{
	int index = 0;

	if (nullb->nr_queues != 1)
		index = raw_smp_processor_id() / ((nr_cpu_ids + nullb->nr_queues - 1) / nullb->nr_queues);

	return &nullb->queues[index];
}

static void null_queue_bio(struct request_queue *q, struct bio *bio)
{
	struct nullb *nullb = q->queuedata;
	struct nullb_queue *nq = nullb_to_queue(nullb);
	struct nullb_cmd *cmd;

	cmd = alloc_cmd(nq, 1);
	cmd->bio = bio;

	null_handle_cmd(cmd);
}

static int null_rq_prep_fn(struct request_queue *q, struct request *req)
{
	struct nullb *nullb = q->queuedata;
	struct nullb_queue *nq = nullb_to_queue(nullb);
	struct nullb_cmd *cmd;

	cmd = alloc_cmd(nq, 0);
	if (cmd) {
		cmd->rq = req;
		req->special = cmd;
		return BLKPREP_OK;
	}

	return BLKPREP_DEFER;
}

static void null_request_fn(struct request_queue *q)
{
	struct request *rq;

	while ((rq = blk_fetch_request(q)) != NULL) {
		struct nullb_cmd *cmd = rq->special;

		spin_unlock_irq(q->queue_lock);
		null_handle_cmd(cmd);
		spin_lock_irq(q->queue_lock);
	}
}

static int null_queue_rq(struct blk_mq_hw_ctx *hctx, struct request *rq)
{
	struct nullb_cmd *cmd = rq->special;

	cmd->rq = rq;
	cmd->nq = hctx->driver_data;

	null_handle_cmd(cmd);
	return BLK_MQ_RQ_QUEUE_OK;
}

static struct blk_mq_hw_ctx *null_alloc_hctx(struct blk_mq_reg *reg, unsigned int hctx_index)
{
	return kmalloc_node(sizeof(struct blk_mq_hw_ctx),
				GFP_KERNEL | __GFP_ZERO, hctx_index);
}

static void null_free_hctx(struct blk_mq_hw_ctx *hctx, unsigned int hctx_index)
{
	kfree(hctx);
}

/*
 * Map each per-cpu software queue to a per-node hardware queue
 */
struct blk_mq_hw_ctx *null_queue_map_per_node(struct request_queue *q,
					      const int ctx_index)
{
	return q->queue_hw_ctx[cpu_to_node(ctx_index)];
}

static int null_init_hctx(struct blk_mq_hw_ctx *hctx, void *data,
			  unsigned int index)
{
	struct nullb *nullb = data;
	struct nullb_queue *nq = &nullb->queues[index];

	init_waitqueue_head(&nq->wait);
	nq->queue_depth = nullb->queue_depth;
	nullb->nr_queues++;
	hctx->driver_data = nq;

	return 0;
}

static struct blk_mq_ops null_mq_ops = {
	.queue_rq       = null_queue_rq,
	.map_queue      = blk_mq_map_queue,
	.init_hctx	= null_init_hctx,
};

static struct blk_mq_reg null_mq_reg = {
	.ops		= &null_mq_ops,
	.queue_depth	= 64,
	.cmd_size	= sizeof(struct nullb_cmd),
	.flags		= BLK_MQ_F_SHOULD_MERGE,
};

static void null_del_dev(struct nullb *nullb)
{
	list_del_init(&nullb->list);

	del_gendisk(nullb->disk);
	if (queue_mode == NULL_Q_MQ)
		blk_mq_free_queue(nullb->q);
	else
		blk_cleanup_queue(nullb->q);
	put_disk(nullb->disk);
	kfree(nullb);
}

static int null_open(struct block_device *bdev, fmode_t mode)
{
	return 0;
}

static void null_release(struct gendisk *disk, fmode_t mode)
{
}

static const struct block_device_operations null_fops = {
	.owner =	THIS_MODULE,
	.open =		null_open,
	.release =	null_release,
};

static int setup_commands(struct nullb_queue *nq)
{
	struct nullb_cmd *cmd;
	int i, tag_size;

	nq->cmds = kzalloc(nq->queue_depth * sizeof(*cmd), GFP_KERNEL);
	if (!nq->cmds)
		return 1;

	tag_size = ALIGN(nq->queue_depth, BITS_PER_LONG) / BITS_PER_LONG;
	nq->tag_map = kzalloc(tag_size * sizeof(unsigned long), GFP_KERNEL);
	if (!nq->tag_map) {
		kfree(nq->cmds);
		return 1;
	}

	for (i = 0; i < nq->queue_depth; i++) {
		cmd = &nq->cmds[i];
		INIT_LIST_HEAD(&cmd->list);
		cmd->ll_list.next = NULL;
		cmd->tag = -1U;
	}

	return 0;
}

static void cleanup_queue(struct nullb_queue *nq)
{
	kfree(nq->tag_map);
	kfree(nq->cmds);
}

static void cleanup_queues(struct nullb *nullb)
{
	int i;

	for (i = 0; i < nullb->nr_queues; i++)
		cleanup_queue(&nullb->queues[i]);

	kfree(nullb->queues);
}

static int setup_queues(struct nullb *nullb)
{
	struct nullb_queue *nq;
	int i;

	nullb->queues = kzalloc(submit_queues * sizeof(*nq), GFP_KERNEL);
	if (!nullb->queues)
		return 1;

	nullb->nr_queues = 0;
	nullb->queue_depth = hw_queue_depth;

	if (queue_mode == NULL_Q_MQ)
		return 0;

	for (i = 0; i < submit_queues; i++) {
		nq = &nullb->queues[i];
		init_waitqueue_head(&nq->wait);
		nq->queue_depth = hw_queue_depth;
		if (setup_commands(nq))
			break;
		nullb->nr_queues++;
	}

	if (i == submit_queues)
		return 0;

	cleanup_queues(nullb);
	return 1;
}

static int null_add_dev(void)
{
	struct gendisk *disk;
	struct nullb *nullb;
	sector_t size;

	nullb = kmalloc_node(sizeof(*nullb), GFP_KERNEL, home_node);
	if (!nullb)
		return -ENOMEM;

	memset(nullb, 0, sizeof(*nullb));

	spin_lock_init(&nullb->lock);

	if (setup_queues(nullb))
		goto err;

	if (queue_mode == NULL_Q_MQ) {
		null_mq_reg.numa_node = home_node;
		null_mq_reg.queue_depth = hw_queue_depth;

		if (use_per_node_hctx) {
			null_mq_reg.ops->alloc_hctx = null_alloc_hctx;
			null_mq_reg.ops->free_hctx = null_free_hctx;

			null_mq_reg.nr_hw_queues = nr_online_nodes;
		} else {
			null_mq_reg.ops->alloc_hctx = blk_mq_alloc_single_hw_queue;
			null_mq_reg.ops->free_hctx = blk_mq_free_single_hw_queue;

			null_mq_reg.nr_hw_queues = submit_queues;
		}

		nullb->q = blk_mq_init_queue(&null_mq_reg, nullb);
	} else if (queue_mode == NULL_Q_BIO) {
		nullb->q = blk_alloc_queue_node(GFP_KERNEL, home_node);
		blk_queue_make_request(nullb->q, null_queue_bio);
	} else {
		nullb->q = blk_init_queue_node(null_request_fn, &nullb->lock, home_node);
		blk_queue_prep_rq(nullb->q, null_rq_prep_fn);
		if (nullb->q)
			blk_queue_softirq_done(nullb->q, null_softirq_done_fn);
	}

	if (!nullb->q)
		goto queue_fail;

	nullb->q->queuedata = nullb;
	queue_flag_set_unlocked(QUEUE_FLAG_NONROT, nullb->q);

	disk = nullb->disk = alloc_disk_node(1, home_node);
	if (!disk) {
queue_fail:
		if (queue_mode == NULL_Q_MQ)
			blk_mq_free_queue(nullb->q);
		else
			blk_cleanup_queue(nullb->q);
		cleanup_queues(nullb);
err:
		kfree(nullb);
		return -ENOMEM;
	}

	mutex_lock(&lock);
	list_add_tail(&nullb->list, &nullb_list);
	nullb->index = nullb_indexes++;
	mutex_unlock(&lock);

	blk_queue_logical_block_size(nullb->q, bs);
	blk_queue_physical_block_size(nullb->q, bs);

	size = gb * 1024 * 1024 * 1024ULL;
	size /= (sector_t) bs;
	set_capacity(disk, size);

	disk->flags |= GENHD_FL_EXT_DEVT;
	spin_lock_init(&nullb->lock);
	disk->major		= null_major;
	disk->first_minor	= nullb->index;
	disk->fops		= &null_fops;
	disk->private_data	= nullb;
	disk->queue		= nullb->q;
	sprintf(disk->disk_name, "nullb%d", nullb->index);
	add_disk(disk);
	return 0;
}

static int __init null_init(void)
{
	unsigned int i;

	if (submit_queues > nr_cpu_ids)
		submit_queues = nr_cpu_ids;
	else if (!submit_queues)
		submit_queues = 1;

	mutex_init(&lock);

	/* Initialize a separate list for each CPU for issuing softirqs */
	for_each_possible_cpu(i) {
		struct completion_queue *cq = &per_cpu(completion_queues, i);

		init_llist_head(&cq->list);

		if (irqmode != NULL_IRQ_TIMER)
			continue;

		hrtimer_init(&cq->timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
		cq->timer.function = null_cmd_timer_expired;
	}

	null_major = register_blkdev(0, "nullb");
	if (null_major < 0)
		return null_major;

	for (i = 0; i < nr_devices; i++) {
		if (null_add_dev()) {
			unregister_blkdev(null_major, "nullb");
			return -EINVAL;
		}
	}

	pr_info("null: module loaded\n");
	return 0;
}

static void __exit null_exit(void)
{
	struct nullb *nullb;

	unregister_blkdev(null_major, "nullb");

	mutex_lock(&lock);
	while (!list_empty(&nullb_list)) {
		nullb = list_entry(nullb_list.next, struct nullb, list);
		null_del_dev(nullb);
	}
	mutex_unlock(&lock);
}

module_init(null_init);
module_exit(null_exit);

//MODULE_AUTHOR("Jens Axboe <jaxboe@fusionio.com>");
//MODULE_LICENSE("GPL");
