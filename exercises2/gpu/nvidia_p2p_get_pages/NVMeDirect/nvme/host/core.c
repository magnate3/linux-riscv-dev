/*
 * NVM Express device driver
 * Copyright (c) 2011-2014, Intel Corporation.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 */

#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/delay.h>
#include <linux/errno.h>
#include <linux/hdreg.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/list_sort.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/pr.h>
#include <linux/ptrace.h>
#include <linux/nvme_ioctl.h>
#include <linux/t10-pi.h>
#include <scsi/sg.h>
#include <asm/unaligned.h>

// Adding this for NVMeD
#include "nv-p2p.h"
#include "nvme_direct.h"
#include "page_handle.h"

#include "nvme.h"


#define NVME_MINORS		(1U << MINORBITS)

static int nvme_major;
module_param(nvme_major, int, 0);

static int nvme_char_major;
module_param(nvme_char_major, int, 0);

static LIST_HEAD(nvme_ctrl_list);
DEFINE_SPINLOCK(dev_list_lock);

static struct class *nvme_class;

static void nvme_free_ns(struct kref *kref)
{
	struct nvme_ns *ns = container_of(kref, struct nvme_ns, kref);

	if (ns->type == NVME_NS_LIGHTNVM)
		nvme_nvm_unregister(ns->queue, ns->disk->disk_name);

	spin_lock(&dev_list_lock);
	ns->disk->private_data = NULL;
	spin_unlock(&dev_list_lock);

	nvme_put_ctrl(ns->ctrl);
	put_disk(ns->disk);
	kfree(ns);
}

static void nvme_put_ns(struct nvme_ns *ns)
{
	kref_put(&ns->kref, nvme_free_ns);
}

static struct nvme_ns *nvme_get_ns_from_disk(struct gendisk *disk)
{
	struct nvme_ns *ns;

	spin_lock(&dev_list_lock);
	ns = disk->private_data;
	if (ns && !kref_get_unless_zero(&ns->kref))
		ns = NULL;
	spin_unlock(&dev_list_lock);

	return ns;
}

void nvme_requeue_req(struct request *req)
{
	unsigned long flags;

	blk_mq_requeue_request(req);
	spin_lock_irqsave(req->q->queue_lock, flags);
	if (!blk_queue_stopped(req->q))
		blk_mq_kick_requeue_list(req->q);
	spin_unlock_irqrestore(req->q->queue_lock, flags);
}

struct request *nvme_alloc_request(struct request_queue *q,
		struct nvme_command *cmd, unsigned int flags)
{
	bool write = cmd->common.opcode & 1;
	struct request *req;

	req = blk_mq_alloc_request(q, write, flags);
	if (IS_ERR(req))
		return req;

	req->cmd_type = REQ_TYPE_DRV_PRIV;
	req->cmd_flags |= REQ_FAILFAST_DRIVER;
	req->__data_len = 0;
	req->__sector = (sector_t) -1;
	req->bio = req->biotail = NULL;

	req->cmd = (unsigned char *)cmd;
	req->cmd_len = sizeof(struct nvme_command);
	req->special = (void *)0;

	return req;
}

/*
 * Returns 0 on success.  If the result is negative, it's a Linux error code;
 * if the result is positive, it's an NVM Express status code
 */
int __nvme_submit_sync_cmd(struct request_queue *q, struct nvme_command *cmd,
		void *buffer, unsigned bufflen, u32 *result, unsigned timeout)
{
	struct request *req;
	int ret;

	req = nvme_alloc_request(q, cmd, 0);
	if (IS_ERR(req))
		return PTR_ERR(req);

	req->timeout = timeout ? timeout : ADMIN_TIMEOUT;

	if (buffer && bufflen) {
		ret = blk_rq_map_kern(q, req, buffer, bufflen, GFP_KERNEL);
		if (ret)
			goto out;
	}

	blk_execute_rq(req->q, NULL, req, 0);
	if (result)
		*result = (u32)(uintptr_t)req->special;
	ret = req->errors;
 out:
	blk_mq_free_request(req);
	return ret;
}

int nvme_submit_sync_cmd(struct request_queue *q, struct nvme_command *cmd,
		void *buffer, unsigned bufflen)
{
	return __nvme_submit_sync_cmd(q, cmd, buffer, bufflen, NULL, 0);
}

int __nvme_submit_user_cmd(struct request_queue *q, struct nvme_command *cmd,
		void __user *ubuffer, unsigned bufflen,
		void __user *meta_buffer, unsigned meta_len, u32 meta_seed,
		u32 *result, unsigned timeout)
{
	bool write = cmd->common.opcode & 1;
	struct nvme_ns *ns = q->queuedata;
	struct gendisk *disk = ns ? ns->disk : NULL;
	struct request *req;
	struct bio *bio = NULL;
	void *meta = NULL;
	int ret;

	req = nvme_alloc_request(q, cmd, 0);
	if (IS_ERR(req))
		return PTR_ERR(req);

	req->timeout = timeout ? timeout : ADMIN_TIMEOUT;

	if (ubuffer && bufflen) {
		ret = blk_rq_map_user(q, req, NULL, ubuffer, bufflen,
				GFP_KERNEL);
		if (ret)
			goto out;
		bio = req->bio;

		if (!disk)
			goto submit;
		bio->bi_bdev = bdget_disk(disk, 0);
		if (!bio->bi_bdev) {
			ret = -ENODEV;
			goto out_unmap;
		}

		if (meta_buffer) {
			struct bio_integrity_payload *bip;

			meta = kmalloc(meta_len, GFP_KERNEL);
			if (!meta) {
				ret = -ENOMEM;
				goto out_unmap;
			}

			if (write) {
				if (copy_from_user(meta, meta_buffer,
						meta_len)) {
					ret = -EFAULT;
					goto out_free_meta;
				}
			}

			bip = bio_integrity_alloc(bio, GFP_KERNEL, 1);
			if (IS_ERR(bip)) {
				ret = PTR_ERR(bip);
				goto out_free_meta;
			}

			bip->bip_iter.bi_size = meta_len;
			bip->bip_iter.bi_sector = meta_seed;

			ret = bio_integrity_add_page(bio, virt_to_page(meta),
					meta_len, offset_in_page(meta));
			if (ret != meta_len) {
				ret = -ENOMEM;
				goto out_free_meta;
			}
		}
	}
 submit:
	blk_execute_rq(req->q, disk, req, 0);
	ret = req->errors;
	if (result)
		*result = (u32)(uintptr_t)req->special;
	if (meta && !ret && !write) {
		if (copy_to_user(meta_buffer, meta, meta_len))
			ret = -EFAULT;
	}
 out_free_meta:
	kfree(meta);
 out_unmap:
	if (bio) {
		if (disk && bio->bi_bdev)
			bdput(bio->bi_bdev);
		blk_rq_unmap_user(bio);
	}
 out:
	blk_mq_free_request(req);
	return ret;
}


int nvme_submit_user_cmd(struct request_queue *q, struct nvme_command *cmd,
		void __user *ubuffer, unsigned bufflen, u32 *result,
		unsigned timeout)
{
	return __nvme_submit_user_cmd(q, cmd, ubuffer, bufflen, NULL, 0, 0,
			result, timeout);
}

int nvme_identify_ctrl(struct nvme_ctrl *dev, struct nvme_id_ctrl **id)
{
	struct nvme_command c = { };
	int error;

	/* gcc-4.4.4 (at least) has issues with initializers and anon unions */
	c.identify.opcode = nvme_admin_identify;
	c.identify.cns = cpu_to_le32(1);

	*id = kmalloc(sizeof(struct nvme_id_ctrl), GFP_KERNEL);
	if (!*id)
		return -ENOMEM;

	error = nvme_submit_sync_cmd(dev->admin_q, &c, *id,
			sizeof(struct nvme_id_ctrl));
	if (error)
		kfree(*id);
	return error;
}

static int nvme_identify_ns_list(struct nvme_ctrl *dev, unsigned nsid, __le32 *ns_list)
{
	struct nvme_command c = { };

	c.identify.opcode = nvme_admin_identify;
	c.identify.cns = cpu_to_le32(2);
	c.identify.nsid = cpu_to_le32(nsid);
	return nvme_submit_sync_cmd(dev->admin_q, &c, ns_list, 0x1000);
}

int nvme_identify_ns(struct nvme_ctrl *dev, unsigned nsid,
		struct nvme_id_ns **id)
{
	struct nvme_command c = { };
	int error;

	/* gcc-4.4.4 (at least) has issues with initializers and anon unions */
	c.identify.opcode = nvme_admin_identify,
	c.identify.nsid = cpu_to_le32(nsid),

	*id = kmalloc(sizeof(struct nvme_id_ns), GFP_KERNEL);
	if (!*id)
		return -ENOMEM;

	error = nvme_submit_sync_cmd(dev->admin_q, &c, *id,
			sizeof(struct nvme_id_ns));
	if (error)
		kfree(*id);
	return error;
}

int nvme_get_features(struct nvme_ctrl *dev, unsigned fid, unsigned nsid,
					dma_addr_t dma_addr, u32 *result)
{
	struct nvme_command c;

	memset(&c, 0, sizeof(c));
	c.features.opcode = nvme_admin_get_features;
	c.features.nsid = cpu_to_le32(nsid);
	c.features.dptr.prp1 = cpu_to_le64(dma_addr);
	c.features.fid = cpu_to_le32(fid);

	return __nvme_submit_sync_cmd(dev->admin_q, &c, NULL, 0, result, 0);
}

int nvme_set_features(struct nvme_ctrl *dev, unsigned fid, unsigned dword11,
					dma_addr_t dma_addr, u32 *result)
{
	struct nvme_command c;

	memset(&c, 0, sizeof(c));
	c.features.opcode = nvme_admin_set_features;
	c.features.dptr.prp1 = cpu_to_le64(dma_addr);
	c.features.fid = cpu_to_le32(fid);
	c.features.dword11 = cpu_to_le32(dword11);

	return __nvme_submit_sync_cmd(dev->admin_q, &c, NULL, 0, result, 0);
}

int nvme_get_log_page(struct nvme_ctrl *dev, struct nvme_smart_log **log)
{
	struct nvme_command c = { };
	int error;

	c.common.opcode = nvme_admin_get_log_page,
	c.common.nsid = cpu_to_le32(0xFFFFFFFF),
	c.common.cdw10[0] = cpu_to_le32(
			(((sizeof(struct nvme_smart_log) / 4) - 1) << 16) |
			 NVME_LOG_SMART),

	*log = kmalloc(sizeof(struct nvme_smart_log), GFP_KERNEL);
	if (!*log)
		return -ENOMEM;

	error = nvme_submit_sync_cmd(dev->admin_q, &c, *log,
			sizeof(struct nvme_smart_log));
	if (error)
		kfree(*log);
	return error;
}

int nvme_set_queue_count(struct nvme_ctrl *ctrl, int *count)
{
	u32 q_count = (*count - 1) | ((*count - 1) << 16);
	u32 result;
	int status, nr_io_queues;

	status = nvme_set_features(ctrl, NVME_FEAT_NUM_QUEUES, q_count, 0,
			&result);
	if (status)
		return status;

	nr_io_queues = min(result & 0xffff, result >> 16) + 1;
	*count = min(*count, nr_io_queues);
	return 0;
}

static int nvme_submit_io(struct nvme_ns *ns, struct nvme_user_io __user *uio)
{
	struct nvme_user_io io;
	struct nvme_command c;
	unsigned length, meta_len;
	void __user *metadata;

	if (copy_from_user(&io, uio, sizeof(io)))
		return -EFAULT;

	switch (io.opcode) {
	case nvme_cmd_write:
	case 0x42:
//		printk("Readx rsvd %x\n", io.rsvd);
		c.rw.rsvd2 = io.rsvd;
		c.rw.rsvd2 = c.rw.rsvd2 << 32 ;		
//		printk("Readx received\n");
	case nvme_cmd_read:
	case nvme_cmd_compare:
		break;
	default:
		return -EINVAL;
	}

	length = (io.nblocks + 1) << ns->lba_shift;
	meta_len = (io.nblocks + 1) * ns->ms;
	metadata = (void __user *)(uintptr_t)io.metadata;

	if (ns->ext) {
		length += meta_len;
		meta_len = 0;
	} else if (meta_len) {
		if ((io.metadata & 3) || !io.metadata)
			return -EINVAL;
	}

	memset(&c, 0, sizeof(c));
	c.rw.opcode = io.opcode;
	c.rw.flags = io.flags;
	c.rw.nsid = cpu_to_le32(ns->ns_id);
	c.rw.slba = cpu_to_le64(io.slba);
	c.rw.length = cpu_to_le16(io.nblocks);
	c.rw.control = cpu_to_le16(io.control);
	c.rw.dsmgmt = cpu_to_le32(io.dsmgmt);
	c.rw.reftag = cpu_to_le32(io.reftag);
	c.rw.apptag = cpu_to_le16(io.apptag);
	c.rw.appmask = cpu_to_le16(io.appmask);
	return __nvme_submit_user_cmd(ns->queue, &c,
			(void __user *)(uintptr_t)io.addr, length,
			metadata, meta_len, io.slba, NULL, 0);
}

// Calculate DMA pages
static int nvme_npages(unsigned size)
{
	unsigned nprps = DIV_ROUND_UP(size + PAGE_SIZE, PAGE_SIZE);
	return DIV_ROUND_UP(8 * nprps, PAGE_SIZE - 8);
}

#define NVME_INT_PAGES		2
#define NVME_INT_BYTES(dev)	(NVME_INT_PAGES * (dev)->ctrl.page_size)

// Initialize IOD
struct nvme_iod *
nvme_alloc_iod(unsigned nseg, unsigned nbytes, gfp_t gfp)
{
	struct nvme_iod *iod = kmalloc(sizeof(struct nvme_iod) +
				sizeof(__le64 *) * nvme_npages(nbytes) +
				sizeof(struct scatterlist) * nseg, gfp);

	if (iod) {
		iod->offset = offsetof(struct nvme_iod, inline_sg[nseg]);
		iod->npages = -1;
		iod->length = nbytes;
		iod->nents = 0;
		iod->first_dma = 0ULL;
	}
		iod->sg = iod->inline_sg;

	return iod;
}
struct nvme_dev {
	struct nvme_queue **queues;
	struct blk_mq_tag_set tagset;
	struct blk_mq_tag_set admin_tagset;
	u32 __iomem *dbs;
	struct device *dev;
	struct dma_pool *prp_page_pool;
	struct dma_pool *prp_small_pool;
	unsigned queue_count;
	unsigned online_queues;
	unsigned max_qid;
	int q_depth;
	u32 db_stride;
	struct msix_entry *entry;
	void __iomem *bar;
	struct work_struct reset_work;
	struct work_struct scan_work;
	struct work_struct remove_work;
	struct work_struct async_work;
	struct timer_list watchdog_timer;
	struct mutex shutdown_lock;
	bool subsystem;
	void __iomem *cmb;
	dma_addr_t cmb_dma_addr;
	u64 cmb_size;
	u32 cmbsz;
	unsigned long flags;

#define NVME_CTRL_RESETTING    0
#define NVME_CTRL_REMOVING     1

	struct nvme_ctrl ctrl;
	struct completion ioq_wait;
#ifdef CONFIG_NVME_VENDOR_EXT_GOOGLE
	u32 *db_mem;
	dma_addr_t doorbell;
	u32 *ei_mem;
	dma_addr_t eventidx;
#endif
};
static __le64 **iod_list(struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	return (__le64 **)(iod->sg + req->nr_phys_segments);
}
static __le64 **gpu_iod_list(struct nvme_iod *iod)
{
	return ((void *)iod) + iod->offset;
}

static inline struct nvme_dev *to_nvme_dev(struct nvme_ctrl *ctrl)
{
	return container_of(ctrl, struct nvme_dev, ctrl);
}

static void nvme_free_iod(struct nvme_dev *dev, struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	const int last_prp = dev->ctrl.page_size / 8 - 1;
	int i;
	__le64 **list = iod_list(req);
	dma_addr_t prp_dma = iod->first_dma;

	if (iod->npages == 0)
		dma_pool_free(dev->prp_small_pool, list[0], prp_dma);
	for (i = 0; i < iod->npages; i++) {
		__le64 *prp_list = list[i];
		dma_addr_t next_prp_dma = le64_to_cpu(prp_list[last_prp]);
		dma_pool_free(dev->prp_page_pool, prp_list, prp_dma);
		prp_dma = next_prp_dma;
	}

	if (iod->sg != iod->inline_sg)
		kfree(iod->sg);
}

void nvme_free_gpu_iod(struct nvme_dev *dev, struct nvme_iod *iod)
{
	const int last_prp = PAGE_SIZE / 8 - 1;
	int i;
	__le64 **list = gpu_iod_list(iod);
	dma_addr_t prp_dma = iod->first_dma;

	if (iod->npages == 0)
		dma_pool_free(dev->prp_small_pool, list[0], prp_dma);
	for (i = 0; i < iod->npages; i++) {
		__le64 *prp_list = list[i];
		dma_addr_t next_prp_dma = le64_to_cpu(prp_list[last_prp]);
		dma_pool_free(dev->prp_page_pool, prp_list, prp_dma);
		prp_dma = next_prp_dma;
	}
	kfree(iod);
}

static struct nvme_iod *map_gpu_pages(struct nvme_dev *dev,
                                      struct nvidia_p2p_page_table *pages,
                                      __u64 offset,
                                      unsigned length)
{
    int i, of;
    struct nvme_iod *iod;
    struct scatterlist *sg;
    uint32_t page_size;

    if (!length || length > INT_MAX - PAGE_SIZE)
        return ERR_PTR(-EINVAL);

    iod = nvme_alloc_iod(pages->entries, length, GFP_KERNEL);
    if (!iod)
        return ERR_PTR(-ENOMEM);

//    	printk("ESCAL NVMeD: iod allocated\n");
    switch(pages->page_size) {
    case NVIDIA_P2P_PAGE_SIZE_4KB:   page_size =   4*1024; break;
    case NVIDIA_P2P_PAGE_SIZE_64KB:  page_size =  64*1024; break;
    case NVIDIA_P2P_PAGE_SIZE_128KB: page_size = 128*1024; break;
    default:
        nvme_free_gpu_iod(dev, iod);
        return ERR_PTR(-EIO);
    }

    sg = iod->sg;
//    	printk("ESCAL NVMeD: Initing SG\n");
    sg_init_table(sg, pages->entries);
//    	printk("ESCAL NVMeD: SG initiated\n");

    of = offset / page_size;
    offset -= of * page_size;

    for (i = 0; i < pages->entries-of; i++) {
        if (!length) break;

        sg[i].page_link = 0;
        sg[i].dma_address = pages->pages[of+i]->physical_address + offset;
        sg[i].length = min_t(unsigned, length, page_size - offset);
        sg[i].dma_length = sg[i].length;
        sg[i].offset = 0;

        offset = 0;

        length -= sg[i].length;
    }

    if (length) {
        nvme_free_gpu_iod(dev, iod);
        return ERR_PTR(-EINVAL);
    }

    sg_mark_end(&sg[i]);
    iod->nents = i;

    return iod;
}


struct nvme_iod *nvmed_init_gpu_iod(struct nvme_ctrl *ctrl,
							  struct request *rq,
							  struct nvidia_p2p_page_table *pages,
                              __u64 offset,
                              unsigned length)
{
    int i, of;
    struct nvme_iod *iod;
    struct scatterlist *sg;
    uint32_t page_size;
	struct nvme_dev *dev = to_nvme_dev(ctrl);

    if (!length || length > INT_MAX - PAGE_SIZE)
        return ERR_PTR(-EINVAL);
//    iod = nvme_alloc_iod(pages->entries, length, GFP_KERNEL);
	iod = blk_mq_rq_to_pdu(rq);
    if (!iod)
    {
    	printk("iod is NULL\n");
        return ERR_PTR(-ENOMEM);
	}

//	int nseg = rq->nr_phys_segments;
//	printk("Init iod: iod: %p nseg -> %d, rq->nr_phys_segments %d\n", iod, pages->entries, rq->nr_phys_segments);
	int nseg = pages->entries;
	rq->nr_phys_segments = nseg;
	unsigned size;

	if (rq->cmd_flags & REQ_DISCARD)
		size = sizeof(struct nvme_dsm_range);
	else
		size = blk_rq_bytes(rq);

	if (nseg > NVME_INT_PAGES || size > NVME_INT_BYTES(dev)) {
		iod->sg = kmalloc(sizeof(struct nvme_iod) + 
						  sizeof(__le64 *) * nvme_npages(length) + 
						  sizeof(struct scatterlist) * nseg, GFP_ATOMIC);
		if (!iod->sg)
			return BLK_MQ_RQ_QUEUE_BUSY;
	} else {
		iod->sg = iod->inline_sg;
	}

	iod->aborted = 0;
	iod->npages = -1;
	iod->nents = 0;
	iod->length = length;
	iod->first_dma = 0ULL;
//	iod->offset = offsetof(struct nvme_iod, sg[nseg]);


    switch(pages->page_size) {
    case NVIDIA_P2P_PAGE_SIZE_4KB:   page_size =   4*1024; break;
    case NVIDIA_P2P_PAGE_SIZE_64KB:  page_size =  64*1024; break;
    case NVIDIA_P2P_PAGE_SIZE_128KB: page_size = 128*1024; break;
    default:
        nvme_free_gpu_iod(dev, iod);
        return ERR_PTR(-EIO);
    }

    sg = iod->sg;
    sg_init_table(sg, pages->entries);
    of = offset / page_size;
    offset -= of * page_size;

    for (i = 0; i < pages->entries-of; i++) {
        if (!length) break;

        sg[i].page_link = 0;
        sg[i].dma_address = pages->pages[of+i]->physical_address + offset;
        sg[i].length = min_t(unsigned, length, page_size - offset);
        sg[i].dma_length = sg[i].length;
        sg[i].offset = 0;

        offset = 0;

        length -= sg[i].length;
    }

    if (length) {
//        nvme_free_iod(dev, iod);
		printk("Got wrong here -- length >0\n");
        return ERR_PTR(-EINVAL);
    }

    sg_mark_end(&sg[i]);
    iod->nents = i;
//	printk("IOD initialized\n");

    return iod;
}


static bool nvme_setup_prps(struct nvme_ctrl *ctrl, struct request *req,
		int total_len)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct dma_pool *pool;
	int length = total_len;
	struct scatterlist *sg = iod->sg;
	int dma_len = sg_dma_len(sg);
//	printk("sg: %p iod: %p\n", sg,iod);
//	printk("Did we even get here? I\n");
	u64 dma_addr = sg_dma_address(sg);
//	printk("Did we even get here? II\n");
	u32 page_size = ctrl->page_size;
	int offset = dma_addr & (page_size - 1);
	__le64 *prp_list;
	__le64 **list = iod_list(req);
//	printk("Did we even get here? III\n");
	dma_addr_t prp_dma;
	int nprps, i;
//	printk("page size: %d dma_len: %d dma_addr: %llx\n", page_size,dma_len,dma_addr);
	struct nvme_dev *dev = to_nvme_dev(ctrl);
	length -= (page_size - offset);
	if (length <= 0)
		return true;

	dma_len -= (page_size - offset);
	if (dma_len) {
		dma_addr += (page_size - offset);
	} else {
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}

	if (length <= page_size) {
		iod->first_dma = dma_addr;
		return true;
	}

	nprps = DIV_ROUND_UP(length, page_size);
	if (nprps <= (256 / 8)) {
		pool = dev->prp_small_pool;
		iod->npages = 0;
	} else {
		pool = dev->prp_page_pool;
		iod->npages = 1;
	}

	prp_list = dma_pool_alloc(pool, GFP_ATOMIC, &prp_dma);
	if (!prp_list) {
		iod->first_dma = dma_addr;
		iod->npages = -1;
		return false;
	}
	list[0] = prp_list;
	iod->first_dma = prp_dma;
	i = 0;
	for (;;) {
		if (i == page_size >> 3) {
			__le64 *old_prp_list = prp_list;
			prp_list = dma_pool_alloc(pool, GFP_ATOMIC, &prp_dma);
			if (!prp_list)
				return false;
			list[iod->npages++] = prp_list;
			prp_list[0] = old_prp_list[i - 1];
			old_prp_list[i - 1] = cpu_to_le64(prp_dma);
			i = 1;
		}
		prp_list[i++] = cpu_to_le64(dma_addr);
		dma_len -= page_size;
		dma_addr += page_size;
		length -= page_size;
		if (length <= 0)
			break;
		if (dma_len > 0)
			continue;
		BUG_ON(dma_len < 0);
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}

	return true;
}

void nvme_unmap_user_pages(struct nvme_ctrl *ctrl, int write,
			struct nvme_iod *iod)
{
	int i;
	struct nvme_dev *dev = to_nvme_dev(ctrl);
	dma_unmap_sg(dev->dev, iod->sg, iod->nents,	write ? DMA_TO_DEVICE : DMA_FROM_DEVICE);

	for (i = 0; i < iod->nents; i++)
		put_page(sg_page(&iod->sg[i]));
}

static int handle_pfn_pages(struct scatterlist *sg, unsigned long start,
                            unsigned long length)
{
    int i = 0;
    struct vm_area_struct *vma = NULL;
    unsigned long pfn;
    struct mm_struct *mm = current->mm;

    do {
        vma = find_vma(mm, start);
        if (!vma || !(vma->vm_flags & VM_PFNMAP))
            return -EFAULT;

        sg[i].page_link = 0;

        if (follow_pfn(vma, start, &pfn))
            return -EINVAL;

        sg[i].dma_address = pfn << PAGE_SHIFT;
        sg[i].length = min_t(unsigned, length, vma->vm_end - start);
        sg[i].dma_length = sg[i].length;
        sg[i].offset = 0;

        length -= sg[i].dma_length;
        start += sg[i].dma_length;
        i++;

    } while (length);

    sg_mark_end(&sg[i - 1]);

    return 0;
}


int nvme_setup_gpu_prps(struct nvme_dev *dev, struct nvme_iod *iod, int total_len,
								gfp_t gfp)
{
	struct dma_pool *pool;
	int length = total_len;
	struct scatterlist *sg = iod->sg;
	int dma_len = sg_dma_len(sg);
	u64 dma_addr = sg_dma_address(sg);
	int offset = offset_in_page(dma_addr);
	__le64 *prp_list;
	__le64 **list = gpu_iod_list(iod);
	dma_addr_t prp_dma;
	int nprps, i;

	length -= (PAGE_SIZE - offset);
	if (length <= 0)
		return total_len;

	dma_len -= (PAGE_SIZE - offset);
	if (dma_len) {
		dma_addr += (PAGE_SIZE - offset);
	} else {
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}

	if (length <= PAGE_SIZE) {
		iod->first_dma = dma_addr;
		return total_len;
	}

	nprps = DIV_ROUND_UP(length, PAGE_SIZE);
	if (nprps <= (256 / 8)) {
		pool = dev->prp_small_pool;
		iod->npages = 0;
	} else {
		pool = dev->prp_page_pool;
		iod->npages = 1;
	}

	prp_list = dma_pool_alloc(pool, gfp, &prp_dma);
	if (!prp_list) {
		iod->first_dma = dma_addr;
		iod->npages = -1;
		return (total_len - length) + PAGE_SIZE;
	}
	list[0] = prp_list;
	iod->first_dma = prp_dma;
	i = 0;
	for (;;) {
		if (i == PAGE_SIZE / 8) {
			__le64 *old_prp_list = prp_list;
			prp_list = dma_pool_alloc(pool, gfp, &prp_dma);
			if (!prp_list)
				return total_len - length;
			list[iod->npages++] = prp_list;
			prp_list[0] = old_prp_list[i - 1];
			old_prp_list[i - 1] = cpu_to_le64(prp_dma);
			i = 1;
		}
		prp_list[i++] = cpu_to_le64(dma_addr);
		dma_len -= PAGE_SIZE;
		dma_addr += PAGE_SIZE;
		length -= PAGE_SIZE;
		if (length <= 0)
			break;
		if (dma_len > 0)
			continue;
		BUG_ON(dma_len < 0);
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}

	return total_len;
}

// Issue I/O commands between device and GPU memory
static int nvme_submit_gpu_io(struct nvme_ns *ns, struct nvme_user_gpu_io __user *uio)
{
	struct nvme_dev *dev = to_nvme_dev(ns->ctrl);
	struct nvme_user_gpu_io io;
	struct nvme_command c;
	unsigned length, meta_len;
	void __user *metadata;
	int status;
	struct nvme_iod *iod;
	struct page_handle *p;

	if (copy_from_user(&io, uio, sizeof(io)))
		return -EFAULT;
    p = io.gpu_mem_handle;
	length = (io.nblocks + 1) << ns->lba_shift;
	meta_len = (io.nblocks + 1) * ns->ms;
	metadata = (void __user *)(uintptr_t)io.metadata;

	switch (io.opcode) {
	case nvme_cmd_write:
	case nvme_cmd_read:
	case nvme_cmd_compare:
        iod = map_gpu_pages(dev, p->page_table, io.gpu_mem_offset, length);
		break;
	default:
		return -EINVAL;
	}


	if (ns->ext) {
		length += meta_len;
		meta_len = 0;
	} else if (meta_len) {
		if ((io.metadata & 3) || !io.metadata)
			return -EINVAL;
	}

	memset(&c, 0, sizeof(c));
	c.rw.opcode = io.opcode;
	c.rw.flags = io.flags;
	c.rw.nsid = cpu_to_le32(ns->ns_id);
	c.rw.slba = cpu_to_le64(io.slba);
	c.rw.length = cpu_to_le16(io.nblocks);
	c.rw.control = cpu_to_le16(io.control);
	c.rw.dsmgmt = cpu_to_le32(io.dsmgmt);
	c.rw.reftag = cpu_to_le32(io.reftag);
	c.rw.apptag = cpu_to_le16(io.apptag);
	c.rw.appmask = cpu_to_le16(io.appmask);
//	if(iod == NULL)
//        printk("ESCAL NVMeD: NULL IOD\n");
	
//        printk("ESCAL NVMeD: Setting PRP\n");
	length = nvme_setup_gpu_prps(dev, iod, length, GFP_KERNEL);
//        printk("ESCAL NVMeD: PRP Set\n");

    c.rw.dptr.prp1 = cpu_to_le64(sg_dma_address(iod->sg));
    c.rw.dptr.prp2 = cpu_to_le64(iod->first_dma);
	if (length != (io.nblocks + 1) << ns->lba_shift)
		status = -ENOMEM;
	else
		status = nvme_submit_sync_cmd(ns->queue, &c, NULL, 0);

	nvme_free_gpu_iod(dev, iod);
	return status;
}


static int nvme_user_cmd(struct nvme_ctrl *ctrl, struct nvme_ns *ns,
			struct nvme_passthru_cmd __user *ucmd)
{
	struct nvme_passthru_cmd cmd;
	struct nvme_command c;
	unsigned timeout = 0;
	int status;

	if (!capable(CAP_SYS_ADMIN))
		return -EACCES;
	if (copy_from_user(&cmd, ucmd, sizeof(cmd)))
		return -EFAULT;

	memset(&c, 0, sizeof(c));
	c.common.opcode = cmd.opcode;
	c.common.flags = cmd.flags;
	c.common.nsid = cpu_to_le32(cmd.nsid);
	c.common.cdw2[0] = cpu_to_le32(cmd.cdw2);
	c.common.cdw2[1] = cpu_to_le32(cmd.cdw3);
	c.common.cdw10[0] = cpu_to_le32(cmd.cdw10);
	c.common.cdw10[1] = cpu_to_le32(cmd.cdw11);
	c.common.cdw10[2] = cpu_to_le32(cmd.cdw12);
	c.common.cdw10[3] = cpu_to_le32(cmd.cdw13);
	c.common.cdw10[4] = cpu_to_le32(cmd.cdw14);
	c.common.cdw10[5] = cpu_to_le32(cmd.cdw15);

	if (cmd.timeout_ms)
		timeout = msecs_to_jiffies(cmd.timeout_ms);

	status = nvme_submit_user_cmd(ns ? ns->queue : ctrl->admin_q, &c,
			(void __user *)(uintptr_t)cmd.addr, cmd.data_len,
			&cmd.result, timeout);
	if (status >= 0) {
		if (put_user(cmd.result, &ucmd->result))
			return -EFAULT;
	}

	return status;
}

static int nvme_ioctl(struct block_device *bdev, fmode_t mode,
		unsigned int cmd, unsigned long arg)
{
	struct nvme_ns *ns = bdev->bd_disk->private_data;
	int ret;
	switch (cmd) {
	case NVME_IOCTL_ID:
		force_successful_syscall_return();
		return ns->ns_id;
	case NVME_IOCTL_ADMIN_CMD:
		return nvme_user_cmd(ns->ctrl, NULL, (void __user *)arg);
	case NVME_IOCTL_IO_CMD:
		return nvme_user_cmd(ns->ctrl, ns, (void __user *)arg);
	case NVME_IOCTL_SUBMIT_IO:
//		printk("We are here!\n");
		ret = nvme_submit_io(ns, (void __user *)arg);
//		printk("Return %d!\n", ret);
		return ret;
	case NVME_IOCTL_SUBMIT_GPU_IO:
		//printk("We are here! -- gpu io\n");
		return nvme_submit_gpu_io(ns, (void __user *)arg);
#ifdef CONFIG_BLK_DEV_NVME_SCSI
	case SG_GET_VERSION_NUM:
		return nvme_sg_get_version_num((void __user *)arg);
	case SG_IO:
		return nvme_sg_io(ns, (void __user *)arg);
#endif
	default:
		return -ENOTTY;
	}
}

#ifdef CONFIG_COMPAT
static int nvme_compat_ioctl(struct block_device *bdev, fmode_t mode,
			unsigned int cmd, unsigned long arg)
{
	switch (cmd) {
	case SG_IO:
		return -ENOIOCTLCMD;
	}
	return nvme_ioctl(bdev, mode, cmd, arg);
}
#else
#define nvme_compat_ioctl	NULL
#endif

static int nvme_open(struct block_device *bdev, fmode_t mode)
{
	return nvme_get_ns_from_disk(bdev->bd_disk) ? 0 : -ENXIO;
}

static void nvme_release(struct gendisk *disk, fmode_t mode)
{
	nvme_put_ns(disk->private_data);
}

static int nvme_getgeo(struct block_device *bdev, struct hd_geometry *geo)
{
	/* some standard values */
	geo->heads = 1 << 6;
	geo->sectors = 1 << 5;
	geo->cylinders = get_capacity(bdev->bd_disk) >> 11;
	return 0;
}

#ifdef CONFIG_BLK_DEV_INTEGRITY
static void nvme_init_integrity(struct nvme_ns *ns)
{
	struct blk_integrity integrity;

	switch (ns->pi_type) {
	case NVME_NS_DPS_PI_TYPE3:
		integrity.profile = &t10_pi_type3_crc;
		break;
	case NVME_NS_DPS_PI_TYPE1:
	case NVME_NS_DPS_PI_TYPE2:
		integrity.profile = &t10_pi_type1_crc;
		break;
	default:
		integrity.profile = NULL;
		break;
	}
	integrity.tuple_size = ns->ms;
	blk_integrity_register(ns->disk, &integrity);
	blk_queue_max_integrity_segments(ns->queue, 1);
}
#else
static void nvme_init_integrity(struct nvme_ns *ns)
{
}
#endif /* CONFIG_BLK_DEV_INTEGRITY */

static void nvme_config_discard(struct nvme_ns *ns)
{
	struct nvme_ctrl *ctrl = ns->ctrl;
	u32 logical_block_size = queue_logical_block_size(ns->queue);

	if (ctrl->quirks & NVME_QUIRK_DISCARD_ZEROES)
		ns->queue->limits.discard_zeroes_data = 1;
	else
		ns->queue->limits.discard_zeroes_data = 0;

	ns->queue->limits.discard_alignment = logical_block_size;
	ns->queue->limits.discard_granularity = logical_block_size;
	blk_queue_max_discard_sectors(ns->queue, 0xffffffff);
	queue_flag_set_unlocked(QUEUE_FLAG_DISCARD, ns->queue);
}

static int nvme_revalidate_disk(struct gendisk *disk)
{
	struct nvme_ns *ns = disk->private_data;
	struct nvme_id_ns *id;
	u8 lbaf, pi_type;
	u16 old_ms;
	unsigned short bs;

	if (test_bit(NVME_NS_DEAD, &ns->flags)) {
		set_capacity(disk, 0);
		return -ENODEV;
	}
	if (nvme_identify_ns(ns->ctrl, ns->ns_id, &id)) {
		dev_warn(ns->ctrl->dev, "%s: Identify failure nvme%dn%d\n",
				__func__, ns->ctrl->instance, ns->ns_id);
		return -ENODEV;
	}
	if (id->ncap == 0) {
		kfree(id);
		return -ENODEV;
	}

	if (nvme_nvm_ns_supported(ns, id) && ns->type != NVME_NS_LIGHTNVM) {
		if (nvme_nvm_register(ns->queue, disk->disk_name)) {
			dev_warn(ns->ctrl->dev,
				"%s: LightNVM init failure\n", __func__);
			kfree(id);
			return -ENODEV;
		}
		ns->type = NVME_NS_LIGHTNVM;
	}

	if (ns->ctrl->vs >= NVME_VS(1, 1))
		memcpy(ns->eui, id->eui64, sizeof(ns->eui));
	if (ns->ctrl->vs >= NVME_VS(1, 2))
		memcpy(ns->uuid, id->nguid, sizeof(ns->uuid));

	old_ms = ns->ms;
	lbaf = id->flbas & NVME_NS_FLBAS_LBA_MASK;
	ns->lba_shift = id->lbaf[lbaf].ds;
	ns->ms = le16_to_cpu(id->lbaf[lbaf].ms);
	ns->ext = ns->ms && (id->flbas & NVME_NS_FLBAS_META_EXT);

	/*
	 * If identify namespace failed, use default 512 byte block size so
	 * block layer can use before failing read/write for 0 capacity.
	 */
	if (ns->lba_shift == 0)
		ns->lba_shift = 9;
	bs = 1 << ns->lba_shift;
	/* XXX: PI implementation requires metadata equal t10 pi tuple size */
	pi_type = ns->ms == sizeof(struct t10_pi_tuple) ?
					id->dps & NVME_NS_DPS_PI_MASK : 0;

	blk_mq_freeze_queue(disk->queue);
	if (blk_get_integrity(disk) && (ns->pi_type != pi_type ||
				ns->ms != old_ms ||
				bs != queue_logical_block_size(disk->queue) ||
				(ns->ms && ns->ext)))
		blk_integrity_unregister(disk);

	ns->pi_type = pi_type;
	blk_queue_logical_block_size(ns->queue, bs);

	if (ns->ms && !blk_get_integrity(disk) && !ns->ext)
		nvme_init_integrity(ns);
	if (ns->ms && !(ns->ms == 8 && ns->pi_type) && !blk_get_integrity(disk))
		set_capacity(disk, 0);
	else
		set_capacity(disk, le64_to_cpup(&id->nsze) << (ns->lba_shift - 9));

	if (ns->ctrl->oncs & NVME_CTRL_ONCS_DSM)
		nvme_config_discard(ns);
	blk_mq_unfreeze_queue(disk->queue);

	kfree(id);
	return 0;
}

static char nvme_pr_type(enum pr_type type)
{
	switch (type) {
	case PR_WRITE_EXCLUSIVE:
		return 1;
	case PR_EXCLUSIVE_ACCESS:
		return 2;
	case PR_WRITE_EXCLUSIVE_REG_ONLY:
		return 3;
	case PR_EXCLUSIVE_ACCESS_REG_ONLY:
		return 4;
	case PR_WRITE_EXCLUSIVE_ALL_REGS:
		return 5;
	case PR_EXCLUSIVE_ACCESS_ALL_REGS:
		return 6;
	default:
		return 0;
	}
};

static int nvme_pr_command(struct block_device *bdev, u32 cdw10,
				u64 key, u64 sa_key, u8 op)
{
	struct nvme_ns *ns = bdev->bd_disk->private_data;
	struct nvme_command c;
	u8 data[16] = { 0, };

	put_unaligned_le64(key, &data[0]);
	put_unaligned_le64(sa_key, &data[8]);

	memset(&c, 0, sizeof(c));
	c.common.opcode = op;
	c.common.nsid = cpu_to_le32(ns->ns_id);
	c.common.cdw10[0] = cpu_to_le32(cdw10);

	return nvme_submit_sync_cmd(ns->queue, &c, data, 16);
}

static int nvme_pr_register(struct block_device *bdev, u64 old,
		u64 new, unsigned flags)
{
	u32 cdw10;

	if (flags & ~PR_FL_IGNORE_KEY)
		return -EOPNOTSUPP;

	cdw10 = old ? 2 : 0;
	cdw10 |= (flags & PR_FL_IGNORE_KEY) ? 1 << 3 : 0;
	cdw10 |= (1 << 30) | (1 << 31); /* PTPL=1 */
	return nvme_pr_command(bdev, cdw10, old, new, nvme_cmd_resv_register);
}

static int nvme_pr_reserve(struct block_device *bdev, u64 key,
		enum pr_type type, unsigned flags)
{
	u32 cdw10;

	if (flags & ~PR_FL_IGNORE_KEY)
		return -EOPNOTSUPP;

	cdw10 = nvme_pr_type(type) << 8;
	cdw10 |= ((flags & PR_FL_IGNORE_KEY) ? 1 << 3 : 0);
	return nvme_pr_command(bdev, cdw10, key, 0, nvme_cmd_resv_acquire);
}

static int nvme_pr_preempt(struct block_device *bdev, u64 old, u64 new,
		enum pr_type type, bool abort)
{
	u32 cdw10 = nvme_pr_type(type) << 8 | abort ? 2 : 1;
	return nvme_pr_command(bdev, cdw10, old, new, nvme_cmd_resv_acquire);
}

static int nvme_pr_clear(struct block_device *bdev, u64 key)
{
	u32 cdw10 = 1 | (key ? 1 << 3 : 0);
	return nvme_pr_command(bdev, cdw10, key, 0, nvme_cmd_resv_register);
}

static int nvme_pr_release(struct block_device *bdev, u64 key, enum pr_type type)
{
	u32 cdw10 = nvme_pr_type(type) << 8 | key ? 1 << 3 : 0;
	return nvme_pr_command(bdev, cdw10, key, 0, nvme_cmd_resv_release);
}

static const struct pr_ops nvme_pr_ops = {
	.pr_register	= nvme_pr_register,
	.pr_reserve	= nvme_pr_reserve,
	.pr_release	= nvme_pr_release,
	.pr_preempt	= nvme_pr_preempt,
	.pr_clear	= nvme_pr_clear,
};

static const struct block_device_operations nvme_fops = {
	.owner		= THIS_MODULE,
	.ioctl		= nvme_ioctl,
	.compat_ioctl	= nvme_compat_ioctl,
	.open		= nvme_open,
	.release	= nvme_release,
	.getgeo		= nvme_getgeo,
	.revalidate_disk= nvme_revalidate_disk,
	.pr_ops		= &nvme_pr_ops,
};

static int nvme_wait_ready(struct nvme_ctrl *ctrl, u64 cap, bool enabled)
{
	unsigned long timeout =
		((NVME_CAP_TIMEOUT(cap) + 1) * HZ / 2) + jiffies;
	u32 csts, bit = enabled ? NVME_CSTS_RDY : 0;
	int ret;

	while ((ret = ctrl->ops->reg_read32(ctrl, NVME_REG_CSTS, &csts)) == 0) {
		if ((csts & NVME_CSTS_RDY) == bit)
			break;

		msleep(100);
		if (fatal_signal_pending(current))
			return -EINTR;
		if (time_after(jiffies, timeout)) {
			dev_err(ctrl->dev,
				"Device not ready; aborting %s\n", enabled ?
						"initialisation" : "reset");
			return -ENODEV;
		}
	}

	return ret;
}

/*
 * If the device has been passed off to us in an enabled state, just clear
 * the enabled bit.  The spec says we should set the 'shutdown notification
 * bits', but doing so may cause the device to complete commands to the
 * admin queue ... and we don't know what memory that might be pointing at!
 */
int nvme_disable_ctrl(struct nvme_ctrl *ctrl, u64 cap)
{
	int ret;

	ctrl->ctrl_config &= ~NVME_CC_SHN_MASK;
	ctrl->ctrl_config &= ~NVME_CC_ENABLE;

	ret = ctrl->ops->reg_write32(ctrl, NVME_REG_CC, ctrl->ctrl_config);
	if (ret)
		return ret;

	/* Checking for ctrl->tagset is a trick to avoid sleeping on module
	 * load, since we only need the quirk on reset_controller. Notice
	 * that the HGST device needs this delay only in firmware activation
	 * procedure; unfortunately we have no (easy) way to verify this.
	 */
	if ((ctrl->quirks & NVME_QUIRK_DELAY_BEFORE_CHK_RDY) && ctrl->tagset)
		msleep(NVME_QUIRK_DELAY_AMOUNT);

	return nvme_wait_ready(ctrl, cap, false);
}

int nvme_enable_ctrl(struct nvme_ctrl *ctrl, u64 cap)
{
	/*
	 * Default to a 4K page size, with the intention to update this
	 * path in the future to accomodate architectures with differing
	 * kernel and IO page sizes.
	 */
	unsigned dev_page_min = NVME_CAP_MPSMIN(cap) + 12, page_shift = 12;
	int ret;

	if (page_shift < dev_page_min) {
		dev_err(ctrl->dev,
			"Minimum device page size %u too large for host (%u)\n",
			1 << dev_page_min, 1 << page_shift);
		return -ENODEV;
	}

	ctrl->page_size = 1 << page_shift;

	ctrl->ctrl_config = NVME_CC_CSS_NVM;
	ctrl->ctrl_config |= (page_shift - 12) << NVME_CC_MPS_SHIFT;
	ctrl->ctrl_config |= NVME_CC_ARB_RR | NVME_CC_SHN_NONE;
	ctrl->ctrl_config |= NVME_CC_IOSQES | NVME_CC_IOCQES;
	ctrl->ctrl_config |= NVME_CC_ENABLE;

	ret = ctrl->ops->reg_write32(ctrl, NVME_REG_CC, ctrl->ctrl_config);
	if (ret)
		return ret;
	return nvme_wait_ready(ctrl, cap, true);
}

int nvme_shutdown_ctrl(struct nvme_ctrl *ctrl)
{
	unsigned long timeout = SHUTDOWN_TIMEOUT + jiffies;
	u32 csts;
	int ret;

	ctrl->ctrl_config &= ~NVME_CC_SHN_MASK;
	ctrl->ctrl_config |= NVME_CC_SHN_NORMAL;

	ret = ctrl->ops->reg_write32(ctrl, NVME_REG_CC, ctrl->ctrl_config);
	if (ret)
		return ret;

	while ((ret = ctrl->ops->reg_read32(ctrl, NVME_REG_CSTS, &csts)) == 0) {
		if ((csts & NVME_CSTS_SHST_MASK) == NVME_CSTS_SHST_CMPLT)
			break;

		msleep(100);
		if (fatal_signal_pending(current))
			return -EINTR;
		if (time_after(jiffies, timeout)) {
			dev_err(ctrl->dev,
				"Device shutdown incomplete; abort shutdown\n");
			return -ENODEV;
		}
	}

	return ret;
}

static void nvme_set_queue_limits(struct nvme_ctrl *ctrl,
		struct request_queue *q)
{
	if (ctrl->max_hw_sectors) {
		u32 max_segments =
			(ctrl->max_hw_sectors / (ctrl->page_size >> 9)) + 1;

		blk_queue_max_hw_sectors(q, ctrl->max_hw_sectors);
		blk_queue_max_segments(q, min_t(u32, max_segments, USHRT_MAX));
	}
	if (ctrl->stripe_size)
		blk_queue_chunk_sectors(q, ctrl->stripe_size >> 9);
	if (ctrl->vwc & NVME_CTRL_VWC_PRESENT)
		blk_queue_flush(q, REQ_FLUSH | REQ_FUA);
	blk_queue_virt_boundary(q, ctrl->page_size - 1);
}

/*
 * Initialize the cached copies of the Identify data and various controller
 * register in our nvme_ctrl structure.  This should be called as soon as
 * the admin queue is fully up and running.
 */
int nvme_init_identify(struct nvme_ctrl *ctrl)
{
	struct nvme_id_ctrl *id;
	u64 cap;
	int ret, page_shift;

	ret = ctrl->ops->reg_read32(ctrl, NVME_REG_VS, &ctrl->vs);
	if (ret) {
		dev_err(ctrl->dev, "Reading VS failed (%d)\n", ret);
		return ret;
	}

	ret = ctrl->ops->reg_read64(ctrl, NVME_REG_CAP, &cap);
	if (ret) {
		dev_err(ctrl->dev, "Reading CAP failed (%d)\n", ret);
		return ret;
	}
	page_shift = NVME_CAP_MPSMIN(cap) + 12;

	if (ctrl->vs >= NVME_VS(1, 1))
		ctrl->subsystem = NVME_CAP_NSSRC(cap);

	ret = nvme_identify_ctrl(ctrl, &id);
	if (ret) {
		dev_err(ctrl->dev, "Identify Controller failed (%d)\n", ret);
		return -EIO;
	}

	ctrl->oncs = le16_to_cpup(&id->oncs);
	atomic_set(&ctrl->abort_limit, id->acl + 1);
	ctrl->vwc = id->vwc;
	memcpy(ctrl->serial, id->sn, sizeof(id->sn));
	memcpy(ctrl->model, id->mn, sizeof(id->mn));
	memcpy(ctrl->firmware_rev, id->fr, sizeof(id->fr));
	if (id->mdts)
		ctrl->max_hw_sectors = 1 << (id->mdts + page_shift - 9);
	else
		ctrl->max_hw_sectors = UINT_MAX;

	if ((ctrl->quirks & NVME_QUIRK_STRIPE_SIZE) && id->vs[3]) {
		unsigned int max_hw_sectors;

		ctrl->stripe_size = 1 << (id->vs[3] + page_shift);
		max_hw_sectors = ctrl->stripe_size >> (page_shift - 9);
		if (ctrl->max_hw_sectors) {
			ctrl->max_hw_sectors = min(max_hw_sectors,
							ctrl->max_hw_sectors);
		} else {
			ctrl->max_hw_sectors = max_hw_sectors;
		}
	}

	nvme_set_queue_limits(ctrl, ctrl->admin_q);

	kfree(id);
	return 0;
}

static int nvme_dev_open(struct inode *inode, struct file *file)
{
	struct nvme_ctrl *ctrl;
	int instance = iminor(inode);
	int ret = -ENODEV;

	spin_lock(&dev_list_lock);
	list_for_each_entry(ctrl, &nvme_ctrl_list, node) {
		if (ctrl->instance != instance)
			continue;

		if (!ctrl->admin_q) {
			ret = -EWOULDBLOCK;
			break;
		}
		if (!kref_get_unless_zero(&ctrl->kref))
			break;
		file->private_data = ctrl;
		ret = 0;
		break;
	}
	spin_unlock(&dev_list_lock);

	return ret;
}

static int nvme_dev_release(struct inode *inode, struct file *file)
{
	nvme_put_ctrl(file->private_data);
	return 0;
}

static int nvme_dev_user_cmd(struct nvme_ctrl *ctrl, void __user *argp)
{
	struct nvme_ns *ns;
	int ret;

	mutex_lock(&ctrl->namespaces_mutex);
	if (list_empty(&ctrl->namespaces)) {
		ret = -ENOTTY;
		goto out_unlock;
	}

	ns = list_first_entry(&ctrl->namespaces, struct nvme_ns, list);
	if (ns != list_last_entry(&ctrl->namespaces, struct nvme_ns, list)) {
		dev_warn(ctrl->dev,
			"NVME_IOCTL_IO_CMD not supported when multiple namespaces present!\n");
		ret = -EINVAL;
		goto out_unlock;
	}

	dev_warn(ctrl->dev,
		"using deprecated NVME_IOCTL_IO_CMD ioctl on the char device!\n");
	kref_get(&ns->kref);
	mutex_unlock(&ctrl->namespaces_mutex);

	ret = nvme_user_cmd(ctrl, ns, argp);
	nvme_put_ns(ns);
	return ret;

out_unlock:
	mutex_unlock(&ctrl->namespaces_mutex);
	return ret;
}

static long nvme_dev_ioctl(struct file *file, unsigned int cmd,
		unsigned long arg)
{
	struct nvme_ctrl *ctrl = file->private_data;
	void __user *argp = (void __user *)arg;

	switch (cmd) {
	case NVME_IOCTL_ADMIN_CMD:
		return nvme_user_cmd(ctrl, NULL, argp);
	case NVME_IOCTL_IO_CMD:
		return nvme_dev_user_cmd(ctrl, argp);
	case NVME_IOCTL_RESET:
		dev_warn(ctrl->dev, "resetting controller\n");
		return ctrl->ops->reset_ctrl(ctrl);
	case NVME_IOCTL_SUBSYS_RESET:
		return nvme_reset_subsystem(ctrl);
	default:
		return -ENOTTY;
	}
}

static const struct file_operations nvme_dev_fops = {
	.owner		= THIS_MODULE,
	.open		= nvme_dev_open,
	.release	= nvme_dev_release,
	.unlocked_ioctl	= nvme_dev_ioctl,
	.compat_ioctl	= nvme_dev_ioctl,
};

static ssize_t nvme_sysfs_reset(struct device *dev,
				struct device_attribute *attr, const char *buf,
				size_t count)
{
	struct nvme_ctrl *ctrl = dev_get_drvdata(dev);
	int ret;

	ret = ctrl->ops->reset_ctrl(ctrl);
	if (ret < 0)
		return ret;
	return count;
}
static DEVICE_ATTR(reset_controller, S_IWUSR, NULL, nvme_sysfs_reset);

static ssize_t uuid_show(struct device *dev, struct device_attribute *attr,
								char *buf)
{
	struct nvme_ns *ns = dev_to_disk(dev)->private_data;
	return sprintf(buf, "%pU\n", ns->uuid);
}
static DEVICE_ATTR(uuid, S_IRUGO, uuid_show, NULL);

static ssize_t eui_show(struct device *dev, struct device_attribute *attr,
								char *buf)
{
	struct nvme_ns *ns = dev_to_disk(dev)->private_data;
	return sprintf(buf, "%8phd\n", ns->eui);
}
static DEVICE_ATTR(eui, S_IRUGO, eui_show, NULL);

static ssize_t nsid_show(struct device *dev, struct device_attribute *attr,
								char *buf)
{
	struct nvme_ns *ns = dev_to_disk(dev)->private_data;
	return sprintf(buf, "%d\n", ns->ns_id);
}
static DEVICE_ATTR(nsid, S_IRUGO, nsid_show, NULL);

static struct attribute *nvme_ns_attrs[] = {
	&dev_attr_uuid.attr,
	&dev_attr_eui.attr,
	&dev_attr_nsid.attr,
	NULL,
};

static umode_t nvme_attrs_are_visible(struct kobject *kobj,
		struct attribute *a, int n)
{
	struct device *dev = container_of(kobj, struct device, kobj);
	struct nvme_ns *ns = dev_to_disk(dev)->private_data;

	if (a == &dev_attr_uuid.attr) {
		if (!memchr_inv(ns->uuid, 0, sizeof(ns->uuid)))
			return 0;
	}
	if (a == &dev_attr_eui.attr) {
		if (!memchr_inv(ns->eui, 0, sizeof(ns->eui)))
			return 0;
	}
	return a->mode;
}

static const struct attribute_group nvme_ns_attr_group = {
	.attrs		= nvme_ns_attrs,
	.is_visible	= nvme_attrs_are_visible,
};

#define nvme_show_function(field)						\
static ssize_t  field##_show(struct device *dev,				\
			    struct device_attribute *attr, char *buf)		\
{										\
        struct nvme_ctrl *ctrl = dev_get_drvdata(dev);				\
        return sprintf(buf, "%.*s\n", (int)sizeof(ctrl->field), ctrl->field);	\
}										\
static DEVICE_ATTR(field, S_IRUGO, field##_show, NULL);

nvme_show_function(model);
nvme_show_function(serial);
nvme_show_function(firmware_rev);

static struct attribute *nvme_dev_attrs[] = {
	&dev_attr_reset_controller.attr,
	&dev_attr_model.attr,
	&dev_attr_serial.attr,
	&dev_attr_firmware_rev.attr,
	NULL
};

static struct attribute_group nvme_dev_attrs_group = {
	.attrs = nvme_dev_attrs,
};

static const struct attribute_group *nvme_dev_attr_groups[] = {
	&nvme_dev_attrs_group,
	NULL,
};

static int ns_cmp(void *priv, struct list_head *a, struct list_head *b)
{
	struct nvme_ns *nsa = container_of(a, struct nvme_ns, list);
	struct nvme_ns *nsb = container_of(b, struct nvme_ns, list);

	return nsa->ns_id - nsb->ns_id;
}

static struct nvme_ns *nvme_find_ns(struct nvme_ctrl *ctrl, unsigned nsid)
{
	struct nvme_ns *ns;

	lockdep_assert_held(&ctrl->namespaces_mutex);

	list_for_each_entry(ns, &ctrl->namespaces, list) {
		if (ns->ns_id == nsid)
			return ns;
		if (ns->ns_id > nsid)
			break;
	}
	return NULL;
}

static void nvme_alloc_ns(struct nvme_ctrl *ctrl, unsigned nsid)
{
	struct nvme_ns *ns;
	struct gendisk *disk;
	int node = dev_to_node(ctrl->dev);

	lockdep_assert_held(&ctrl->namespaces_mutex);

	ns = kzalloc_node(sizeof(*ns), GFP_KERNEL, node);
	if (!ns)
		return;

	ns->queue = blk_mq_init_queue(ctrl->tagset);
	if (IS_ERR(ns->queue))
		goto out_free_ns;
	queue_flag_set_unlocked(QUEUE_FLAG_NOMERGES, ns->queue);
	queue_flag_set_unlocked(QUEUE_FLAG_NONROT, ns->queue);
	ns->queue->queuedata = ns;
	ns->ctrl = ctrl;

	disk = alloc_disk_node(0, node);
	if (!disk)
		goto out_free_queue;

	kref_init(&ns->kref);
	ns->ns_id = nsid;
	ns->disk = disk;
	ns->lba_shift = 9; /* set to a default value for 512 until disk is validated */


	blk_queue_logical_block_size(ns->queue, 1 << ns->lba_shift);
	nvme_set_queue_limits(ctrl, ns->queue);

	disk->major = nvme_major;
	disk->first_minor = 0;
	disk->fops = &nvme_fops;
	disk->private_data = ns;
	disk->queue = ns->queue;
	disk->driverfs_dev = ctrl->device;
	disk->flags = GENHD_FL_EXT_DEVT;
	sprintf(disk->disk_name, "nvme%dn%d", ctrl->instance, nsid);

	if (nvme_revalidate_disk(ns->disk))
		goto out_free_disk;

	list_add_tail(&ns->list, &ctrl->namespaces);
	kref_get(&ctrl->kref);
	if (ns->type == NVME_NS_LIGHTNVM)
		return;

	add_disk(ns->disk);
	if (sysfs_create_group(&disk_to_dev(ns->disk)->kobj,
					&nvme_ns_attr_group))
		pr_warn("%s: failed to create sysfs group for identification\n",
			ns->disk->disk_name);
	return;
 out_free_disk:
	kfree(disk);
 out_free_queue:
	blk_cleanup_queue(ns->queue);
 out_free_ns:
	kfree(ns);
}

static void nvme_ns_remove(struct nvme_ns *ns)
{
	if (test_and_set_bit(NVME_NS_REMOVING, &ns->flags))
		return;

	if (ns->disk->flags & GENHD_FL_UP) {
		if (blk_get_integrity(ns->disk))
			blk_integrity_unregister(ns->disk);
		sysfs_remove_group(&disk_to_dev(ns->disk)->kobj,
					&nvme_ns_attr_group);
		del_gendisk(ns->disk);
		blk_mq_abort_requeue_list(ns->queue);
		blk_cleanup_queue(ns->queue);
	}
	mutex_lock(&ns->ctrl->namespaces_mutex);
	list_del_init(&ns->list);
	mutex_unlock(&ns->ctrl->namespaces_mutex);
	nvme_put_ns(ns);
}

static void nvme_validate_ns(struct nvme_ctrl *ctrl, unsigned nsid)
{
	struct nvme_ns *ns;

	ns = nvme_find_ns(ctrl, nsid);
	if (ns) {
		if (revalidate_disk(ns->disk))
			nvme_ns_remove(ns);
	} else
		nvme_alloc_ns(ctrl, nsid);
}

static int nvme_scan_ns_list(struct nvme_ctrl *ctrl, unsigned nn)
{
	struct nvme_ns *ns;
	__le32 *ns_list;
	unsigned i, j, nsid, prev = 0, num_lists = DIV_ROUND_UP(nn, 1024);
	int ret = 0;

	ns_list = kzalloc(0x1000, GFP_KERNEL);
	if (!ns_list)
		return -ENOMEM;

	for (i = 0; i < num_lists; i++) {
		ret = nvme_identify_ns_list(ctrl, prev, ns_list);
		if (ret)
			goto out;

		for (j = 0; j < min(nn, 1024U); j++) {
			nsid = le32_to_cpu(ns_list[j]);
			if (!nsid)
				goto out;

			nvme_validate_ns(ctrl, nsid);

			while (++prev < nsid) {
				ns = nvme_find_ns(ctrl, prev);
				if (ns)
					nvme_ns_remove(ns);
			}
		}
		nn -= j;
	}
 out:
	kfree(ns_list);
	return ret;
}

static void __nvme_scan_namespaces(struct nvme_ctrl *ctrl, unsigned nn)
{
	struct nvme_ns *ns, *next;
	unsigned i;

	lockdep_assert_held(&ctrl->namespaces_mutex);

	for (i = 1; i <= nn; i++)
		nvme_validate_ns(ctrl, i);

	list_for_each_entry_safe(ns, next, &ctrl->namespaces, list) {
		if (ns->ns_id > nn)
			nvme_ns_remove(ns);
	}
}

void nvme_scan_namespaces(struct nvme_ctrl *ctrl)
{
	struct nvme_id_ctrl *id;
	unsigned nn;

	if (nvme_identify_ctrl(ctrl, &id))
		return;

	mutex_lock(&ctrl->namespaces_mutex);
	nn = le32_to_cpu(id->nn);
	if (ctrl->vs >= NVME_VS(1, 1) &&
	    !(ctrl->quirks & NVME_QUIRK_IDENTIFY_CNS)) {
		if (!nvme_scan_ns_list(ctrl, nn))
			goto done;
	}
	__nvme_scan_namespaces(ctrl, le32_to_cpup(&id->nn));
 done:
	list_sort(NULL, &ctrl->namespaces, ns_cmp);
	mutex_unlock(&ctrl->namespaces_mutex);
	kfree(id);
}

void nvme_remove_namespaces(struct nvme_ctrl *ctrl)
{
	struct nvme_ns *ns, *next;

	list_for_each_entry_safe(ns, next, &ctrl->namespaces, list)
		nvme_ns_remove(ns);
}

static DEFINE_IDA(nvme_instance_ida);

static int nvme_set_instance(struct nvme_ctrl *ctrl)
{
	int instance, error;

	do {
		if (!ida_pre_get(&nvme_instance_ida, GFP_KERNEL))
			return -ENODEV;

		spin_lock(&dev_list_lock);
		error = ida_get_new(&nvme_instance_ida, &instance);
		spin_unlock(&dev_list_lock);
	} while (error == -EAGAIN);

	if (error)
		return -ENODEV;

	ctrl->instance = instance;
	return 0;
}

static void nvme_release_instance(struct nvme_ctrl *ctrl)
{
	spin_lock(&dev_list_lock);
	ida_remove(&nvme_instance_ida, ctrl->instance);
	spin_unlock(&dev_list_lock);
}

void nvme_uninit_ctrl(struct nvme_ctrl *ctrl)
 {
	device_destroy(nvme_class, MKDEV(nvme_char_major, ctrl->instance));

	spin_lock(&dev_list_lock);
	list_del(&ctrl->node);
	spin_unlock(&dev_list_lock);
}

static void nvme_free_ctrl(struct kref *kref)
{
	struct nvme_ctrl *ctrl = container_of(kref, struct nvme_ctrl, kref);

	put_device(ctrl->device);
	nvme_release_instance(ctrl);

	ctrl->ops->free_ctrl(ctrl);
}

void nvme_put_ctrl(struct nvme_ctrl *ctrl)
{
	kref_put(&ctrl->kref, nvme_free_ctrl);
}

/*
 * Initialize a NVMe controller structures.  This needs to be called during
 * earliest initialization so that we have the initialized structured around
 * during probing.
 */
int nvme_init_ctrl(struct nvme_ctrl *ctrl, struct device *dev,
		const struct nvme_ctrl_ops *ops, unsigned long quirks)
{
	int ret;

	INIT_LIST_HEAD(&ctrl->namespaces);
	mutex_init(&ctrl->namespaces_mutex);
	kref_init(&ctrl->kref);
	ctrl->dev = dev;
	ctrl->ops = ops;
	ctrl->quirks = quirks;

	ret = nvme_set_instance(ctrl);
	if (ret)
		goto out;

	ctrl->device = device_create_with_groups(nvme_class, ctrl->dev,
				MKDEV(nvme_char_major, ctrl->instance),
				dev, nvme_dev_attr_groups,
				"nvme%d", ctrl->instance);
	if (IS_ERR(ctrl->device)) {
		ret = PTR_ERR(ctrl->device);
		goto out_release_instance;
	}
	get_device(ctrl->device);
	dev_set_drvdata(ctrl->device, ctrl);

	spin_lock(&dev_list_lock);
	list_add_tail(&ctrl->node, &nvme_ctrl_list);
	spin_unlock(&dev_list_lock);

	return 0;
out_release_instance:
	nvme_release_instance(ctrl);
out:
	return ret;
}

/**
 * nvme_kill_queues(): Ends all namespace queues
 * @ctrl: the dead controller that needs to end
 *
 * Call this function when the driver determines it is unable to get the
 * controller in a state capable of servicing IO.
 */
void nvme_kill_queues(struct nvme_ctrl *ctrl)
{
	struct nvme_ns *ns;

	mutex_lock(&ctrl->namespaces_mutex);
	list_for_each_entry(ns, &ctrl->namespaces, list) {
		if (!kref_get_unless_zero(&ns->kref))
			continue;

		/*
		 * Revalidating a dead namespace sets capacity to 0. This will
		 * end buffered writers dirtying pages that can't be synced.
		 */
		if (!test_and_set_bit(NVME_NS_DEAD, &ns->flags))
			revalidate_disk(ns->disk);

		blk_set_queue_dying(ns->queue);
		blk_mq_abort_requeue_list(ns->queue);
		blk_mq_start_stopped_hw_queues(ns->queue, true);

		nvme_put_ns(ns);
	}
	mutex_unlock(&ctrl->namespaces_mutex);
}

void nvme_stop_queues(struct nvme_ctrl *ctrl)
{
	struct nvme_ns *ns;

	mutex_lock(&ctrl->namespaces_mutex);
	list_for_each_entry(ns, &ctrl->namespaces, list) {
		spin_lock_irq(ns->queue->queue_lock);
		queue_flag_set(QUEUE_FLAG_STOPPED, ns->queue);
		spin_unlock_irq(ns->queue->queue_lock);

		blk_mq_cancel_requeue_work(ns->queue);
		blk_mq_stop_hw_queues(ns->queue);
	}
	mutex_unlock(&ctrl->namespaces_mutex);
}

void nvme_start_queues(struct nvme_ctrl *ctrl)
{
	struct nvme_ns *ns;

	mutex_lock(&ctrl->namespaces_mutex);
	list_for_each_entry(ns, &ctrl->namespaces, list) {
		queue_flag_clear_unlocked(QUEUE_FLAG_STOPPED, ns->queue);
		blk_mq_start_stopped_hw_queues(ns->queue, true);
		blk_mq_kick_requeue_list(ns->queue);
	}
	mutex_unlock(&ctrl->namespaces_mutex);
}

int __init nvme_core_init(void)
{
	int result;

	result = register_blkdev(nvme_major, "nvme");
	if (result < 0)
		return result;
	else if (result > 0)
		nvme_major = result;

	result = __register_chrdev(nvme_char_major, 0, NVME_MINORS, "nvme",
							&nvme_dev_fops);
	if (result < 0)
		goto unregister_blkdev;
	else if (result > 0)
		nvme_char_major = result;

	nvme_class = class_create(THIS_MODULE, "nvme");
	if (IS_ERR(nvme_class)) {
		result = PTR_ERR(nvme_class);
		goto unregister_chrdev;
	}
	printk("ESCAL NVMe Driver\n");
	return 0;

 unregister_chrdev:
	__unregister_chrdev(nvme_char_major, 0, NVME_MINORS, "nvme");
 unregister_blkdev:
	unregister_blkdev(nvme_major, "nvme");
	return result;
}

void nvme_core_exit(void)
{
	unregister_blkdev(nvme_major, "nvme");
	class_destroy(nvme_class);
	__unregister_chrdev(nvme_char_major, 0, NVME_MINORS, "nvme");
}
