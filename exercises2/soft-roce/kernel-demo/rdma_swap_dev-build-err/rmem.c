/*
 * A sample, extra-simple block driver. Updated for kernel 2.6.31.
 *
 * (C) 2003 Eklektix, Inc.
 * (C) 2010 Pat Patterson <pat at superpat dot com>
 * Redistributable under the terms of the GNU GPL.
 * Modified by Sangjin Han (sangjin@eecs.berkeley.edu) and Peter Gao (petergao@berkeley.edu)
 */

#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>

#include <linux/kernel.h> /* printk() */
#include <linux/fs.h>     /* everything... */
#include <linux/errno.h>  /* error codes */
#include <linux/types.h>  /* size_t */
#include <linux/vmalloc.h>
#include <linux/genhd.h>
#include <linux/blkdev.h>
#include <linux/hdreg.h>
#include <linux/random.h>
#include <linux/un.h>
#include <net/sock.h>
#include <linux/socket.h>
#include <linux/delay.h>
#include <linux/bio.h>
#include <linux/version.h>

#include <linux/time.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/cpufreq.h>
#include "rdma_library.h"
#include "log.h"
#include "conf.h"

MODULE_LICENSE("Dual BSD/GPL");


static char* servers = "localhost:8888";
module_param(servers, charp, 0);


struct rmem_device {
  unsigned long size;
  spinlock_t lock;
  spinlock_t rdma_lock;
  spinlock_t arr_lock; 
  struct gendisk *gd;
  int major_num;
  rdma_ctx_t rdma_ctx;
  #if COPY_LESS
  struct ib_send_wr wrs[MAX_REQ];
  struct ib_sge sges[MAX_REQ];
  #else
  # if MODE == MODE_ASYNC || MODE == MODE_ONE
  rdma_request *rdma_req[REQ_ARR_SIZE];
  # else
  rdma_request rdma_req[MAX_REQ];
  # endif
  #endif
  volatile int head;
  volatile int tail;
  struct request *blk_req[MAX_REQ];
};

struct rmem_device* devices[DEVICE_BOUND];

int initd = 0;

static struct proc_dir_entry* proc_entry;

#if MODE == MODE_ASYNC || MODE == MODE_ONE
struct rdma_request* get_rdma_request_arr(struct rmem_device* dev)
{
    struct rdma_request* ret = NULL;
    spin_lock(&dev->arr_lock);
    if(dev->head == dev->tail)
    {
        pr_err("Pool is almost empty");
        BUG();
    }
    else
    {
        ret = dev->rdma_req[dev->head];
        BUG_ON(ret == NULL);
        dev->rdma_req[dev->head] = NULL;
        dev->head = (dev->head + 1) % REQ_ARR_SIZE;
    }
    spin_unlock(&dev->arr_lock);
    return ret;
}

void return_rdma_request_arr(struct rmem_device* pool, rdma_request* req)
{
    int new_tail;
    BUG_ON(req == NULL);
    spin_lock(&pool->arr_lock);
    new_tail = (pool->tail + 1) % REQ_ARR_SIZE;
    if(new_tail == pool->head)
    {
        pr_err("Err: new_tail == head == %d", new_tail);
        BUG();
    }
    if(pool->rdma_req[new_tail])
    {
        pr_err("Err: New tail value %p", pool->rdma_req[new_tail]);
        BUG();
    }
    pool->rdma_req[new_tail] = req;
    pool->tail = new_tail;
    spin_unlock(&pool->arr_lock);
}
#endif

#if MODE == MODE_ASYNC
static void rmem_request_async(struct request_queue *q)
{
  struct request *req;
  struct bio *bio;
  struct bio_vec *bvec;
  sector_t sector;
  rdma_req_t last_rdma_req, cur_rdma_req = NULL;
  int i, rdma_req_count = 0;
  struct rmem_device *dev = q->queuedata;
  char* buffer;
  struct batch_request* batch_req = NULL, *last_batch_req;
  rdma_request *rdma_req;
  bool first_req = true;

  LOG_KERN(LOG_INFO, "alloc");
  rdma_req = get_rdma_request_arr(dev);


  LOG_KERN(LOG_INFO, "=======Start of rmem request======");

  while ((req = blk_fetch_request(q)) != NULL) 
  {
    if (req->cmd_type != REQ_TYPE_FS) 
    {
      printk (KERN_NOTICE "Skip non-fs request\n");
      __blk_end_request(req, -EIO, blk_rq_cur_bytes(req));
      continue;
    }
    spin_unlock_irq(q->queue_lock); 
    #if DEBUG_OUT_REQ
    debug_pool_insert(dev->rdma_ctx->pool, req); 
    #endif

    last_batch_req = batch_req;
    batch_req = get_batch_request(dev->rdma_ctx->pool);
    LOG_KERN(LOG_INFO, "obtained batch req %d", batch_req->id);
    batch_req->req = req;
    batch_req->outstanding_reqs = 0;
    batch_req->next = NULL;
    batch_req->nsec = 0;
    #if MEASURE_LATENCY
    batch_req->first = first_req;
    first_req = false;
    #endif
    __rq_for_each_bio(bio, req) 
    {
      sector = bio->bi_sector;
      bio_for_each_segment(bvec, bio, i) 
      {
        buffer = __bio_kmap_atomic(bio, i);
        //sbull_transfer(dev, sector, bio_cur_bytes(bio) >> 9, buffer, bio_data_dir(bio) == WRITE);
        cur_rdma_req = rdma_req + rdma_req_count;
        cur_rdma_req->rw = bio_data_dir(bio)?RDMA_WRITE:RDMA_READ;
        cur_rdma_req->length = (bio_cur_bytes(bio) >> 9) * KERNEL_SECTOR_SIZE;
        cur_rdma_req->dma_addr = rdma_map_address(buffer, cur_rdma_req->length);
        cur_rdma_req->remote_offset = (uint64_t)sector * KERNEL_SECTOR_SIZE;
        cur_rdma_req->batch_req = batch_req;

        last_rdma_req = rdma_req + rdma_req_count - 1;
        if(MERGE_REQ && rdma_req_count > 0 && cur_rdma_req->rw == last_rdma_req->rw && 
            last_rdma_req->dma_addr + last_rdma_req->length == cur_rdma_req->dma_addr &&
            last_rdma_req->remote_offset + last_rdma_req->length == cur_rdma_req->remote_offset)
        {
          last_rdma_req->length += cur_rdma_req->length;
          //LOG_KERN(LOG_INFO, "Merging RDMA req %p w: %d  addr: %llu (ptr: %p)  offset: %u  len: %d", req, last_rdma_req_p->rw == RDMA_WRITE, last_rdma_req_p->dma_addr, bio_data(req->bio), last_rdma_req_p->remote_offset, last_rdma_req_p->length);
        }
        else
        {
          //LOG_KERN(LOG_INFO, "Constructing RDMA req %p w: %d  addr: %llu (ptr: %p)  offset: %u  len: %d", req, cur_rdma_req->rw == RDMA_WRITE, cur_rdma_req->dma_addr, buffer, cur_rdma_req->remote_offset, cur_rdma_req->length);
          batch_req->outstanding_reqs++;
          rdma_req_count++;
        }

        sector += bio_cur_bytes(bio) >> 9;
        __bio_kunmap_atomic(buffer);
        BUG_ON(rdma_req_count >= MAX_REQ);

      }
      batch_req->nsec += bio->bi_size/KERNEL_SECTOR_SIZE;
    }

    if(batch_req->outstanding_reqs == 0)
    {
      last_batch_req->next = batch_req;
      LOG_KERN(LOG_INFO, "batch req %d -> %d", last_batch_req->id, batch_req->id);
    }

    spin_lock_irq(q->queue_lock);
  }

  if(rdma_req_count)
  {
    rdma_op(dev->rdma_ctx, rdma_req, rdma_req_count);
  } 
  return_rdma_request_arr(dev, rdma_req);
  LOG_KERN(LOG_INFO, "======End of rmem request======");
}
#endif

#if MODE == MODE_SYNC
static void rmem_request_sync(struct request_queue *q)
{
  struct request *req;
  struct bio *bio;
  #if LINUX_VERSION_CODE < KERNEL_VERSION(3,19,0)
  struct bio_vec *bvec;
  int iter;
  #else
  struct bio_vec bvec;
  struct bvec_iter iter;
  #endif
  sector_t sector;
  rdma_req_t last_rdma_req, cur_rdma_req = NULL;
  int i, blk_req_count = 0, rdma_req_count = 0;
  struct rmem_device *dev = q->queuedata;
  char* buffer;
  #if COPY_LESS
  struct ib_send_wr** bad_wr;
  uint32_t curr_length;
  #else
  rdma_request *rdma_req;
  #endif

  LOG_KERN(LOG_INFO, "=======Start of rmem request======");
  #if !COPY_LESS
  rdma_req = dev->rdma_req;
  #endif

  while ((req = blk_fetch_request(q)) != NULL) 
  {
    if (req->cmd_type != REQ_TYPE_FS) 
    {
      printk (KERN_NOTICE "Skip non-fs request\n");
      __blk_end_request(req, -EIO, blk_rq_cur_bytes(req));
      continue;
    }

    dev->blk_req[blk_req_count++] = req;

    __rq_for_each_bio(bio, req) 
    {
      #if LINUX_VERSION_CODE < KERNEL_VERSION(3,19,0)
      sector = bio->bi_sector;
      #else
      sector = bio->bi_iter.bi_sector;
      #endif
      bio_for_each_segment(bvec, bio, iter) 
      {
        buffer = __bio_kmap_atomic(bio, iter);
        #if COPY_LESS
        curr_length = bio_cur_bytes(bio);
        #if SIMPLE_MAKE_WR
        simple_make_wr(dev->rdma_ctx, dev->wrs+rdma_req_count, dev->sges+rdma_req_count, bio_data_dir(bio)?RDMA_WRITE:RDMA_READ, rdma_map_address(buffer, curr_length), (uint64_t)sector * KERNEL_SECTOR_SIZE, curr_length, NULL); 
        #else
        make_wr(dev->rdma_ctx, dev->wrs+rdma_req_count, dev->sges+rdma_req_count, bio_data_dir(bio)?RDMA_WRITE:RDMA_READ, rdma_map_address(buffer, curr_length), (uint64_t)sector * KERNEL_SECTOR_SIZE, curr_length, NULL); 
        #endif
        if(MERGE_REQ && rdma_req_count > 0 && merge_wr(dev->wrs+rdma_req_count-1, dev->sges+rdma_req_count-1, dev->wrs+rdma_req_count, dev->sges+rdma_req_count))
        {
            LOG_KERN(LOG_INFO, "Merged rdma req");
        }
        else
        {
            LOG_KERN(LOG_INFO, "New rdma req");
            if(rdma_req_count > 0)
            {
                dev->wrs[rdma_req_count-1].next = dev->wrs+rdma_req_count;
            }
            rdma_req_count++;
        }
        if(rdma_req_count >= MAX_REQ)
        {
          LOG_KERN(LOG_INFO, "Sending %d rdma reqs", rdma_req_count);
          #if SIMPLE_MAKE_WR
          dev->wrs[rdma_req_count-1].next = NULL;
          #endif
          dev->rdma_ctx->outstanding_requests = rdma_req_count;
          #if MEASURE_LATENCY && MAX_REQ == 1
          if(rdma_req_count == 1 && dev->wrs[0].sg_list->length == 4096)
            dev->wrs[0].wr_id = (u64)get_cycle();
          #endif
          ib_post_send(dev->rdma_ctx->qp, dev->wrs, bad_wr);
          poll_cq(dev->rdma_ctx);
          rdma_req_count = 0;
        }
        #else
        cur_rdma_req = rdma_req + rdma_req_count;
        cur_rdma_req->rw = bio_data_dir(bio)?RDMA_WRITE:RDMA_READ;
        cur_rdma_req->length = bio_cur_bytes(bio);
        cur_rdma_req->dma_addr = rdma_map_address(buffer, cur_rdma_req->length);
        cur_rdma_req->remote_offset = (uint64_t)sector * KERNEL_SECTOR_SIZE;

        last_rdma_req = rdma_req + rdma_req_count - 1;
        if(MERGE_REQ && rdma_req_count > 0 && cur_rdma_req->rw == last_rdma_req->rw && 
            last_rdma_req->dma_addr + last_rdma_req->length == cur_rdma_req->dma_addr &&
            last_rdma_req->remote_offset + last_rdma_req->length == cur_rdma_req->remote_offset)
        {
          last_rdma_req->length += cur_rdma_req->length;
          LOG_KERN(LOG_INFO, "Merging RDMA req %p w: %d  addr: %llu (ptr: %p)  offset: %u  len: %d", req, last_rdma_req->rw == RDMA_WRITE, last_rdma_req->dma_addr, bio_data(req->bio), last_rdma_req->remote_offset, last_rdma_req->length);
        }
        else
        {
          LOG_KERN(LOG_INFO, "Constructing RDMA req %p w: %d  addr: %llu (ptr: %p)  offset: %u  len: %d", req, cur_rdma_req->rw == RDMA_WRITE, cur_rdma_req->dma_addr, buffer, cur_rdma_req->remote_offset, cur_rdma_req->length);
          rdma_req_count++;
        }
        if(rdma_req_count >= MAX_REQ)
        {
          rdma_op(dev->rdma_ctx, rdma_req, rdma_req_count);
          rdma_req_count = 0;
        }
        #endif
        sector += bio_cur_bytes(bio) >> 9;
        __bio_kunmap_atomic(buffer);
      }
    }

  }

  if(rdma_req_count)
  {
    #if COPY_LESS
    LOG_KERN(LOG_INFO, "Sending %d rdma reqs", rdma_req_count);

    #if SIMPLE_MAKE_WR
    dev->wrs[rdma_req_count-1].next = NULL;
    #endif
    dev->rdma_ctx->outstanding_requests = rdma_req_count;
    #if MEASURE_LATENCY
    if(rdma_req_count == 1 && dev->wrs[0].sg_list->length == 4096 )
      dev->wrs[0].wr_id = (u64)get_cycle();
    #endif
    ib_post_send(dev->rdma_ctx->qp, dev->wrs, bad_wr);
    poll_cq(dev->rdma_ctx);
    #else
    rdma_op(dev->rdma_ctx, rdma_req, rdma_req_count);
    #endif
  }

  for(i = 0; i < blk_req_count; i++)
  {
    __blk_end_request_all(dev->blk_req[i], 0);
  }
  LOG_KERN(LOG_INFO, "======End of rmem request======");
}
#endif

#if MODE == MODE_ONE
static void rmem_request_one(struct request_queue *q)
{
  struct request *req;
  struct bio *bio;
  struct bio_vec *bvec;
  sector_t sector;
  rdma_request cur_rdma_req, temp_rdma_req;
  bool has_req = false;
  #if MEASURE_LATENCY
  bool first_req = true;
  #endif
  int i;
  struct rmem_device *dev = q->queuedata;
  char* buffer;
  struct batch_request* batch_req = NULL;

  LOG_KERN(LOG_INFO, "======Start of rmem request======");

  while ((req = blk_fetch_request(q)) != NULL) 
  {
    if (req->cmd_type != REQ_TYPE_FS) 
    {
      printk (KERN_NOTICE "Skip non-fs request\n");
      __blk_end_request(req, -EIO, blk_rq_cur_bytes(req));
      continue;
    }
    spin_unlock_irq(q->queue_lock); 

    #if DEBUG_OUT_REQ
    debug_pool_insert(dev->rdma_ctx->pool, req);
    #endif

    batch_req = get_batch_request(dev->rdma_ctx->pool);
    batch_req->req = req;
    batch_req->outstanding_reqs = 0;
    batch_req->comp_reqs = 0;
    batch_req->next = NULL;
    batch_req->nsec = 0;
    batch_req->all_request_sent = false;
    #if MEASURE_LATENCY
    batch_req->first = first_req;
    first_req = false;
    #endif

    __rq_for_each_bio(bio, req) 
    {
      sector = bio->bi_sector;
      bio_for_each_segment(bvec, bio, i) 
      {
        buffer = __bio_kmap_atomic(bio, i);
        if(has_req)  
        {
          temp_rdma_req.rw = bio_data_dir(bio)?RDMA_WRITE:RDMA_READ;
          temp_rdma_req.length = (bio_cur_bytes(bio) >> 9) * KERNEL_SECTOR_SIZE;
          temp_rdma_req.dma_addr = rdma_map_address(buffer, cur_rdma_req.length);
          temp_rdma_req.remote_offset = sector * KERNEL_SECTOR_SIZE;
          temp_rdma_req.batch_req = batch_req;
          
          if(temp_rdma_req.rw == cur_rdma_req.rw &&
            cur_rdma_req.dma_addr + cur_rdma_req.length == temp_rdma_req.dma_addr &&
            cur_rdma_req.remote_offset + cur_rdma_req.length == temp_rdma_req.remote_offset)
          {
            cur_rdma_req.length += temp_rdma_req.length;
            LOG_KERN(LOG_INFO, "Merging RDMA req %p w: %d  addr: %llu (ptr: %p)  offset: %u  len: %d", req, cur_rdma_req.rw == RDMA_WRITE, cur_rdma_req.dma_addr, bio_data(req->bio), cur_rdma_req.remote_offset, cur_rdma_req.length);
          }
          else
          {
            LOG_KERN(LOG_INFO, "Flushing RDMA req %p w: %d  addr: %llu (ptr: %p)  offset: %u  len: %d", req, cur_rdma_req.rw == RDMA_WRITE, cur_rdma_req.dma_addr, bio_data(req->bio), cur_rdma_req.remote_offset, cur_rdma_req.length);
            batch_req->outstanding_reqs++;
            rdma_op(dev->rdma_ctx, &cur_rdma_req, 1);
            cur_rdma_req = temp_rdma_req;
          }
        }
        else
        {
          cur_rdma_req.rw = bio_data_dir(bio)?RDMA_WRITE:RDMA_READ;
          cur_rdma_req.length = (bio_cur_bytes(bio) >> 9) * KERNEL_SECTOR_SIZE;
          cur_rdma_req.dma_addr = rdma_map_address(buffer, cur_rdma_req.length);
          cur_rdma_req.remote_offset = sector * KERNEL_SECTOR_SIZE;
          cur_rdma_req.batch_req = batch_req;
          has_req = true;
          batch_req->outstanding_reqs++;
          LOG_KERN(LOG_INFO, "New RDMA req %p w: %d  addr: %llu (ptr: %p)  offset: %u  len: %d", req, cur_rdma_req.rw == RDMA_WRITE, cur_rdma_req.dma_addr, bio_data(req->bio), cur_rdma_req.remote_offset, cur_rdma_req.length);
        }

        sector += bio_cur_bytes(bio) >> 9;
        __bio_kunmap_atomic(buffer);
      }
      batch_req->nsec += bio->bi_size/KERNEL_SECTOR_SIZE;
    }

    batch_req->all_request_sent = true;
    rdma_op(dev->rdma_ctx, &cur_rdma_req, 1);
    
    has_req = false;
    spin_lock_irq(q->queue_lock);
  }

  LOG_KERN(LOG_INFO, "======End of rmem request======");
}
#endif

/*
 * The HDIO_GETGEO ioctl is handled in blkdev_ioctl(), which
 * calls this. We need to implement getgeo, since we can't
 * use tools such as fdisk to partition the drive otherwise.
 */
int rmem_getgeo(struct block_device * block_device, struct hd_geometry * geo) {
  long size;

  /* We have no real geometry, of course, so make something up. */
  //size = device.size * (PAGE_SIZE / KERNEL_SECTOR_SIZE);
  size = ((struct rmem_device*)(block_device->bd_queue->queuedata))->size  * PAGE_SIZE * (PAGE_SIZE / KERNEL_SECTOR_SIZE);
  geo->cylinders = (size & ~0x3f) >> 6;
  geo->heads = 4;
  geo->sectors = 16;
  geo->start = 0;
  return 0;
}

/*
 * The device operations structure.
 */
static struct block_device_operations rmem_ops = {
  .owner  = THIS_MODULE,
  .getgeo = rmem_getgeo
};

/*
static void rdma_transfer(struct rmem_device
*dev, unsigned long sector,
    unsigned long nsect, char *buffer, int write, struct batch_request* batch_req, rdma_request* rdma_reqs)
{
  rdma_request* req = rdma_reqs + batch_req->outstanding_reqs;
  unsigned long offset = sector*KERNEL_SECTOR_SIZE;
  unsigned long nbytes = nsect*KERNEL_SECTOR_SIZE;

  if ((offset + nbytes) > dev->size) {
    printk (KERN_NOTICE "Beyond-end write (%ld %ld)\n", offset, nbytes);
    return;
  }
  req->rw = write?RDMA_WRITE:RDMA_READ;
  req->length = nbytes;
  req->dma_addr = rdma_map_address(buffer, nbytes);
  req->remote_offset = offset;
  req->batch_req = batch_req;
  batch_req->outstanding_reqs++;
  BUG_ON(batch_req->outstanding_reqs > MAX_REQ);
}

static int rdma_xfer_bio(struct rmem_device *dev, struct bio *bio, struct batch_request* batch_req, rdma_request* rdma_reqs)
{
  int i;
  struct bio_vec *bvec;
  sector_t sector = bio->bi_sector;

  bio_for_each_segment(bvec, bio, i) {
    char *buffer = __bio_kmap_atomic(bio, i);
    rdma_transfer(dev, sector, bio_cur_bytes(bio) >> 9, buffer, bio_data_dir(bio) == WRITE, batch_req, rdma_reqs);
    sector += bio_cur_bytes(bio) >> 9;
    __bio_kunmap_atomic(buffer);
  }
  return 0; 
}

static void rdma_make_request(struct request_queue *q, struct bio *bio)
{
  struct rmem_device *dev;
  int status;
  struct batch_request* batch_req;

  LOG_KERN(LOG_INFO, "======New rmem request======");
  dev = q->queuedata;

  batch_req = get_batch_request(dev->rdma_ctx->pool);
  LOG_KERN(LOG_INFO, "batch req %d", batch_req->id);
  batch_req->outstanding_reqs = 0;
  batch_req->next = NULL;
  batch_req->bio = bio;
  status = rdma_xfer_bio(dev, bio, batch_req, dev->rdma_req);

  rdma_op(dev->rdma_ctx, dev->rdma_req, batch_req->outstanding_reqs);
  LOG_KERN(LOG_INFO, "======End of rmem request======");

}
*/

#if MEASURE_LATENCY
static int debug_show_latency_dist(struct seq_file *m, void *v)
{
  int i,j;
  for(i = 0; i < DEVICE_BOUND; i++)
  {
    if(devices[i])
    {
      seq_printf(m, "-----Found device %d display latency-----\n", i);
      for(j = 0; j < LATENCY_BUCKET; j++)
      {
        seq_printf(m, "%d\t%d\n", j, devices[i]->rdma_ctx->pool->latency_dist[j]);
      }
    }
  }
  return 0;
}

#elif DEBUG_OUT_REQ
static int debug_show_out_req(struct seq_file *m, void *v)
{
  int i,j;
  struct batch_request** reqs;
  for(i = 0; i < DEVICE_BOUND; i++)
  {
    if(devices[i]){
      seq_printf(m, "-----Found device %d-----\n", i);
      spin_lock_irq(&devices[i]->rdma_ctx->pool->lock);
      reqs = vmalloc(sizeof(batch_request*) * devices[i]->rdma_ctx->pool->size);
      memset(reqs, 0, sizeof(batch_request*) * devices[i]->rdma_ctx->pool->size);
      //put batch_request_pool objects to reqs, index by id
      seq_printf(m, "Indexing batch_reqs\n");
      for(j = 0; j < devices[i]->rdma_ctx->pool->size; j++)
      {
        if(devices[i]->rdma_ctx->pool->data[j])
        {
          reqs[devices[i]->rdma_ctx->pool->data[j]->id] = devices[i]->rdma_ctx->pool->data[j];
        }
      }
      //print missing pool object, so they are outgoing batch_reqs
      seq_printf(m, "Outstanding batch reqs\n");
      for(j = 0; j < devices[i]->rdma_ctx->pool->size; j++)
      {
        if(reqs[j] == NULL)
          seq_printf(m, "%d:%d\n", j, devices[i]->rdma_ctx->pool->all[j]->outstanding_reqs);
      }
      vfree(reqs);
      //print outgoing block reqs
      seq_printf(m, "Outstanding block reqs\n");
      for(j = 0; j < 1024; j++)
      {
          if(devices[i]->rdma_ctx->pool->io_req[i])
              seq_printf(m, "%p\n", devices[i]->rdma_ctx->pool->io_req[i]);
      }
      spin_unlock_irq(&devices[i]->rdma_ctx->pool->lock);
    }
  }
  return 0;
}
#else
static int debug_show_null(struct seq_file *m, void *v)
{
  return 0;
}
#endif

static int debug_open(struct inode * sp_inode, struct file *sp_file)
{
#if MEASURE_LATENCY
  return single_open(sp_file, debug_show_latency_dist, NULL);
#elif DEBUG_OUT_REQ
  return single_open(sp_file, debug_show_out_req, NULL);
#else
  return single_open(sp_file, debug_show_null, NULL);
#endif
}

static ssize_t debug_write(struct file *sp_file, const char __user *buf, size_t size, loff_t *offset)
{
  LOG_KERN(LOG_INFO, "debug");
  return 0;
}

static struct file_operations debug_fops = {
  .open = debug_open,
  .read = seq_read,
  .write = debug_write,
  .llseek = seq_lseek,
  .release = single_release
};

static int __init rmem_init(void) {
  int c,major_num,i,ret;
  struct rmem_device* device;
  struct request_queue *queue;
  char dev_name[20];
  char *servers_str_p = servers;
  char delim[] = ",:";
  char *tmp_srv, *tmp_port, *tmp_npages;
  char* tmp_end_p;
  int port, npages;

  for(c = 0; c < DEVICE_BOUND; c++) {
    devices[c] = NULL;
  }

  LOG_KERN(LOG_INFO, "Start rmem_rdma. rdma_library_init() CPU freq %d kHz\n", cpu_khz);
  ret = rdma_library_init();
  pr_err("init success? %d\n", ret);

  while(!rdma_library_ready());
  LOG_KERN(LOG_INFO, "init done\n");


  while(1){
    queue = NULL;
    tmp_srv = strsep(&servers_str_p, delim);
    tmp_port = strsep(&servers_str_p, delim);
    tmp_npages = strsep(&servers_str_p, delim);
    if (tmp_srv && tmp_port && tmp_npages){
      port = simple_strtol(tmp_port, &tmp_end_p, 10);
      if(tmp_end_p == NULL){
        pr_info("Incorrect port %s\n", tmp_port);
        goto out;
      }
      npages = simple_strtol(tmp_npages, &tmp_end_p, 10);
      if(tmp_end_p == NULL){
        pr_info("Incorrect npage %s\n", tmp_npages);
        goto out;
      }
      pr_err("Connecting to server %s port %d, requesting %d pages \n", tmp_srv, port, npages);

      device = vmalloc(sizeof(*device));
      

      device->size = npages * PAGE_SIZE;
      spin_lock_init(&device->lock);
      spin_lock_init(&device->rdma_lock);
      spin_lock_init(&device->arr_lock);
      device->head = 0;
      device->tail = REQ_ARR_SIZE - 1;


      #if MODE == MODE_ASYNC || MODE == MODE_ONE
      for(i = 0; i < REQ_ARR_SIZE; i++)
        device->rdma_req[i] = vmalloc(sizeof(rdma_request) * MAX_REQ);
      #endif    

      device->rdma_ctx = rdma_init(npages, tmp_srv, port, REQ_POOL_SIZE);
      if(device->rdma_ctx == NULL){
        pr_info("rdma_init() failed\n");
        goto out;
      }
      #if SIMPLE_MAKE_WR
      memset(device->wrs, 0, MAX_REQ * sizeof(struct ib_send_wr));
      memset(device->sges, 0, MAX_REQ * sizeof(struct ib_sge));
      for(i = 0; i < MAX_REQ; i++)
      {
        device->sges[i].lkey = device->rdma_ctx->mr->lkey;
        device->wrs[i].sg_list = device->sges + i;
        device->wrs[i].num_sge = 1;
        device->wrs[i].send_flags = IB_SEND_SIGNALED;
        device->wrs[i].wr.rdma.rkey = device->rdma_ctx->rem_rkey;
      }
      #endif      /*
       * Get a request queue.
       */
      if(CUSTOM_MAKE_REQ_FN)
      {
        //queue = blk_alloc_queue(GFP_KERNEL);
        //if (queue == NULL)
        //  goto out_rdma_exit;
        //blk_queue_make_request(queue, rdma_make_request);
      }
      else
      {
#if MODE == MODE_SYNC
        queue = blk_init_queue(rmem_request_sync, &device->lock);
#elif MODE == MODE_ASYNC
        queue = blk_init_queue(rmem_request_async, &device->lock);
#elif MODE == MODE_ONE
        queue = blk_init_queue(rmem_request_one, &device->lock);
#else
        #error "Wrong Mode"
#endif
        if (queue == NULL)
          goto out_rdma_exit;
      }
      queue->queuedata = device;
      pr_info("init queue id %d\n", queue->id);
      if (queue->id >= DEVICE_BOUND) 
        goto out_rdma_exit;
      scnprintf(dev_name, 20, "rmem_rdma%d", queue->id);
      devices[queue->id] = device;
      blk_queue_physical_block_size(queue, PAGE_SIZE);
      blk_queue_logical_block_size(queue, PAGE_SIZE);
      blk_queue_io_min(queue, PAGE_SIZE);
      blk_queue_io_opt(queue, PAGE_SIZE * 4);
      /*
       * Get registered.
       */
      major_num = register_blkdev(0, dev_name);
      device->major_num = major_num;
      pr_info("Registering blkdev %s major_num %d\n", dev_name, major_num);
      if (major_num < 0) {
        printk(KERN_WARNING "rmem: unable to get major number\n");
        goto out_rdma_exit;
      }
      /*
       * And the gendisk structure.
       */
      device->gd = alloc_disk(16);
      if (!device->gd)
        goto out_unregister;
      device->gd->major = major_num;
      device->gd->first_minor = 0;
      device->gd->fops = &rmem_ops;
      device->gd->private_data = device;
      strcpy(device->gd->disk_name, dev_name);
      set_capacity(device->gd, 0);
      device->gd->queue = queue;
      add_disk(device->gd);
      set_capacity(device->gd, npages * SECTORS_PER_PAGE);

      
    }
    else
      break;
  }
  initd = 1;

  proc_entry = proc_create("rmem_debug", 0666, NULL, &debug_fops);


  pr_info("rmem_rdma successfully loaded!\n");
  return 0;

out_unregister:
  unregister_blkdev(major_num, dev_name);
out_rdma_exit:
  rdma_exit(device->rdma_ctx);
out:
  if(queue && devices[queue->id])
    devices[queue->id] = NULL;
  for(c = 0; c < DEVICE_BOUND; c++) {
    if(devices[c]){
      unregister_blkdev(devices[c]->gd->major, devices[c]->gd->disk_name);
      rdma_exit(devices[c]->rdma_ctx);
    }
  }
  rdma_library_exit();
  return -ENOMEM;
}

static void __exit rmem_exit(void)
{
  int c, i;
  for(c = 0; c < DEVICE_BOUND; c++) {
    if(devices[c] != NULL){
      del_gendisk(devices[c]->gd);
      put_disk(devices[c]->gd);
      pr_info("Unregistering blkdev %s major_num %d\n", devices[c]->gd->disk_name, devices[c]->major_num);
      unregister_blkdev(devices[c]->major_num, devices[c]->gd->disk_name);
      blk_cleanup_queue(devices[c]->gd->queue);

      rdma_exit(devices[c]->rdma_ctx);
    
      #if MODE == MODE_ASYNC || MODE == MODE_ONE 
      for(i = 0; i < REQ_ARR_SIZE; i++)
        if(devices[c]->rdma_req[i])
          vfree(devices[c]->rdma_req[i]);
      #endif

      vfree(devices[c]);
      devices[c] = NULL;
    }
  }
  rdma_library_exit();
  
  remove_proc_entry("rmem_debug", NULL);

  pr_info("rmem: bye!\n");
}

module_init(rmem_init);
module_exit(rmem_exit);
