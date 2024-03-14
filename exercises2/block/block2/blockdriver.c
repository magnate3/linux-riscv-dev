#include <linux/module.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/cdev.h>
#include <linux/errno.h>
#include <linux/sched.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include<linux/genhd.h>

#include<linux/blkdev.h> 
#include <linux/device.h>

#define SIMP_BLKDEV_DISKNAME "simp_blkdev2"
#define SIMP_BLKDEV_DEVICEMAJOR 220
#define SIMP_BLKDEV_BYTES (4*1024*1024)

static DEFINE_SPINLOCK(rq_lock);
unsigned char simp_blkdev_data[SIMP_BLKDEV_BYTES];

static struct gendisk *simp_blkdev_disk;
static struct request_queue *simp_blkdev_queue;//device's request queue

struct block_device_operations simp_blkdev_fops = {
	.owner = THIS_MODULE,
};

void print_requestbuffer(char *buffer) {
	int i, val;
	for (i = 0; i < 40; i++) {
		val = *(buffer + i);
		
		printk("0x%x ,", val);

		if((i+1) % 10 == 0)
			printk("\n");
	}
}

//handle request that pass to this device
static void simp_blkdev_do_request(struct request_queue *q){

	struct request *req;

	req = blk_fetch_request(q);

	while (req) {
		unsigned int start = blk_rq_pos(req) << 9;
		unsigned int len  = blk_rq_cur_bytes(req);
		int err = 0;

		if (start + len > SIMP_BLKDEV_BYTES) {
			printk("BLKDEV: simp_blkdev_do_request : bad access: block=%llu, "
			       "count=%u\n",
			       (unsigned long long)blk_rq_pos(req),
			       blk_rq_cur_sectors(req));
			err = -EIO;
			goto done;
		}

			if (rq_data_dir(req) == READ) {
				printk("BLKDEV:Read-sector:%d-- size = %d--\n", start >> 9, len);

				printk("BLKDEV:source, simp_blkdev_data:\n");
				print_requestbuffer((char*)(simp_blkdev_data + start));

				memcpy(bio_data(req->bio), simp_blkdev_data + start, len);

				printk("BLKDEV:dest, req->buffer:\n");
				print_requestbuffer(bio_data(req->bio));
			}
			else {
				printk("BLKDEV:Write-sector:%d--size = %d--\n", start >> 9, len);

				printk("BLKDEV:source, req->buffer:\n");
				print_requestbuffer(bio_data(req->bio));

				memcpy(simp_blkdev_data + start, bio_data(req->bio), len);

				printk("BLKDEV:dest, simp_blkdev_data:\n");
				print_requestbuffer((char*)(simp_blkdev_data + start));
			}

	done:
		if (!__blk_end_request_cur(req, err))
			req = blk_fetch_request(q);
	}

}

static int simp_blkdev_init(void){

	int ret;

	//init the request queue by the handler function
	simp_blkdev_queue = blk_init_queue(simp_blkdev_do_request,&rq_lock);
	if(!simp_blkdev_queue){
		ret = -ENOMEM;
		goto error_init_queue;
	}

	//alloc the resource of gendisk
	simp_blkdev_disk = alloc_disk(1);
	if(!simp_blkdev_disk){
		ret = -ENOMEM;
		goto error_alloc_disk;
	}

	//populate the gendisk structure
	strcpy(simp_blkdev_disk->disk_name,SIMP_BLKDEV_DISKNAME);
	simp_blkdev_disk->major = SIMP_BLKDEV_DEVICEMAJOR;
	simp_blkdev_disk->first_minor = 0;
	simp_blkdev_disk->fops = &simp_blkdev_fops;
	simp_blkdev_disk->queue = simp_blkdev_queue;
	set_capacity(simp_blkdev_disk,SIMP_BLKDEV_BYTES>>9);
	
	add_disk(simp_blkdev_disk);

	memset(simp_blkdev_data, 0, SIMP_BLKDEV_BYTES);

	printk("BLKDEV: module simp_blkdev added.\n");
	return 0;

error_init_queue:
	blk_cleanup_queue(simp_blkdev_queue);

error_alloc_disk:
	return ret;	

}
static void simp_blkdev_exit(void){

	del_gendisk(simp_blkdev_disk);

	put_disk(simp_blkdev_disk);

	blk_cleanup_queue(simp_blkdev_queue);

	printk("BLKDEV: module simp_blkdev romoved.\n");
}

module_init(simp_blkdev_init);
module_exit(simp_blkdev_exit);
MODULE_LICENSE("GPLV2");
