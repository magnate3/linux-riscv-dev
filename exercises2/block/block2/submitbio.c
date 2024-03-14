/*
 * This file provides functions for block I/O operations on swap/file.
 *
 * Copyright (C) 1998,2001-2005 Pavel Machek <pavel@ucw.cz>
 * Copyright (C) 2006 Rafael J. Wysocki <rjw@sisk.pl>
 * Copyright (C) 2014 Vincent Wan<vincent.wan@amd.com>
 * This file is released under the GPLv2.
 */

#include <linux/bio.h>
#include <linux/kernel.h>
#include <linux/pagemap.h>
#include<linux/blkdev.h> 
#include <linux/device.h>
/*start to debug for submit_bio*/
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/backing-dev.h>
#include <linux/bio.h>
#include <linux/blkdev.h>
#include <linux/highmem.h>
#include <linux/mm.h>
#include <linux/kernel_stat.h>
#include <linux/string.h>
#include <linux/init.h>
#include <linux/completion.h>
#include <linux/slab.h>
#include <linux/swap.h>
#include <linux/writeback.h>
#include <linux/task_io_accounting_ops.h>
#include <linux/fault-inject.h>
#include <linux/list_sort.h>
#include <linux/delay.h>
#include <linux/ratelimit.h>
#include <linux/pm_runtime.h>
/*end*/
#define DEVICE_BYTES_PER_BLOCK PAGE_SIZE
unsigned int max_vecs = 16;
unsigned int real_max_vecs = 4;
static int block_want_dump = 1;

int i = 0, j = 0;

blk_qc_t submit_bio_me(int rw, struct bio *bio)
{
	//bio->bi_rw |= rw;
        // bio->bi_opf = REQ_OP_READ
        if (rw & WRITE){
         //bio->bi_opf = REQ_OP_WRITE;
         bio_set_op_attrs(bio, REQ_OP_WRITE, 0);
        }
        else {
         //bio->bi_opf = REQ_OP_READ;
         bio_set_op_attrs(bio, REQ_OP_READ,0);
        }
	/*
	 * If it's a regular read/write or a barrier with data attached,
	 * go through the normal accounting stuff before submission.
	 */
	if (bio_has_data(bio)) {
		unsigned int count;

		if (unlikely(rw & REQ_OP_WRITE_SAME))
			count = queue_logical_block_size(bio->bi_disk->queue) >> 9;
		else
			count = bio_sectors(bio);

		if (rw & WRITE) {
			count_vm_events(PGPGOUT, count);
		} else {
			task_io_account_read(bio->bi_iter.bi_size);
			count_vm_events(PGPGIN, count);
		}

		if (block_want_dump) {
			char b[BDEVNAME_SIZE];
			printk(KERN_INFO "%s(%d): %s block %Lu on %s (%u sectors)\n",
			current->comm, task_pid_nr(current),
				(rw & WRITE) ? "WRITE" : "READ",
				(unsigned long long)bio->bi_iter.bi_sector,
				bio_devname(bio, b),
				count);
		}
	}

	return generic_make_request(bio);
}
EXPORT_SYMBOL(submit_bio_me);

int bio_add_page_me(struct bio *bio, struct page *page,
		 unsigned int len, unsigned int offset)
{
	struct bio_vec *bv;

	/*
	 * cloned bio must not modify vec list
	 */
	if (WARN_ON_ONCE(bio_flagged(bio, BIO_CLONED)))
		return 0;

	/*
	 * For filesystems with a blocksize smaller than the pagesize
	 * we will often be called with the same page as last time and
	 * a consecutive offset.  Optimize this special case.
	 */
	if (bio->bi_vcnt > 0) {
		bv = &bio->bi_io_vec[bio->bi_vcnt - 1];

		if (page == bv->bv_page &&
		    offset == bv->bv_offset + bv->bv_len) {
			bv->bv_len += len;
			goto done;
		}
	}

	if (bio->bi_vcnt >= bio->bi_max_vecs)
		return 0;

	bv		= &bio->bi_io_vec[bio->bi_vcnt];
	bv->bv_page	= page;
	bv->bv_len	= len;
	bv->bv_offset	= offset;

	bio->bi_vcnt++;
done:
	bio->bi_iter.bi_size += len;
	return len;
}

static void end_bio_read(struct bio *bio)
{
	bio_put(bio);
	printk("5. SUBMITIO: Call bio->bi_end_io-------------\n");
}

/**
 *	submit - submit BIO request.
 *	blocks - start device's block num.
 *	ptr, len - memory space.
 */
static int submit(int rw, struct block_device *bdev, unsigned int blocks,
		char *ptr, int len)
{
	const int bio_rw = rw | REQ_SYNC;
	struct bio *bio;
	int bi_sizes = 0, length = 0;

submit_retry:

	bio = bio_alloc(__GFP_RECLAIM | __GFP_HIGH, max_vecs);
	bio->bi_iter.bi_sector = blocks * (DEVICE_BYTES_PER_BLOCK >> 9);

	printk("1.SUBMITIO:\n");

	printk("bio->bi_max_vecs:%d\n", bio->bi_max_vecs);
	printk("bio->bi_vcnt:%d\n", bio->bi_vcnt);
	printk("bio->bi_iter.bi_sector:%ld\n", bio->bi_iter.bi_sector);

	//bio->bi_bdev = bdev;
         bio_set_dev(bio, bdev);
	bio->bi_end_io = end_bio_read;

	do {
		length = len < PAGE_SIZE? len:PAGE_SIZE;

		if (!bio_add_page_me(bio, virt_to_page(ptr), length, 
				virt_to_phys(ptr) & (PAGE_SIZE-1))) {

		printk("2.SUBMITIO:error!!!,bio->bi_vcnt=%d,\
                        bio->bi_iter.bi_sector = %ld,\
                        bio->bi_max_vecs = %d------\n",
			bio->bi_vcnt, bio->bi_iter.bi_sector, bio->bi_max_vecs);

			bio_put(bio);

			goto submit_retry;
		}

		bi_sizes += length;

		printk("3.SUBMITIO:\n");

		printk("bio->bi_max_vecs:%d\n", bio->bi_max_vecs);
		printk("bio->bi_vcnt:%d\n", bio->bi_vcnt);
		printk("bio->bi_iter.bi_sector:%ld\n", bio->bi_iter.bi_sector);

		len -= PAGE_SIZE;
		ptr += PAGE_SIZE;

	} while(len > 0);

	printk("4.SUBMITIO: bio->bi_max_vecs = %d,\
		bi_sizes = %d, bio->bi_iter.bi_size = %dn",
		bio->bi_max_vecs, bi_sizes, bio->bi_iter.bi_size);

	submit_bio_me(bio_rw, bio);

	return 0;
}

int read_bio_page(struct block_device *simplebdev,
		  pgoff_t page_off, void *addr, int len)
{
	return submit(READ, simplebdev, page_off ,addr, len);
}

int write_bio_page(struct block_device *simplebdev,
		  pgoff_t page_off, void *addr, int len)
{
	return submit(WRITE, simplebdev, page_off, addr, len);
}

