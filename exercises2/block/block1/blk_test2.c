#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/delay.h>
#include <linux/ide.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/errno.h>
#include <linux/gpio.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/of_gpio.h>
#include <linux/semaphore.h>
#include <linux/timer.h>
#include <linux/i2c.h>
#include <linux/genhd.h>
#include <linux/blkdev.h>
#include <linux/hdreg.h>

//#include <asm/mach/map.h>
#include <asm/uaccess.h>
#include <asm/io.h>


#define RAMDISK_SIZE	(2 * 1024 * 1024) 	/* 容量大小为2MB */
#define RAMDISK_NAME	"ramdisk"			/* 名字 */
#define RADMISK_MINOR	3					/* 表示有三个磁盘分区！不是次设备号为3！ */

/* ramdisk设备结构体 */
struct ramdisk_dev{
	int major;					/* 主设备号 */
	unsigned char *ramdiskbuf;	/* ramdisk内存空间,用于模拟块设备 */
	spinlock_t lock;			/* 自旋锁 */
	struct gendisk *gendisk; 	/* gendisk */
	struct request_queue *queue;/* 请求队列 */

};

struct ramdisk_dev ramdisk;		/* ramdisk设备 */

/*
 * @description		: 打开块设备
 * @param - dev 	: 块设备
 * @param - mode 	: 打开模式
 * @return 			: 0 成功;其他 失败
 */
int ramdisk_open(struct block_device *dev, fmode_t mode)
{
	printk("ramdisk open\r\n");
	return 0;
}

/*
 * @description		: 释放块设备
 * @param - disk 	: gendisk
 * @param - mode 	: 模式
 * @return 			: 0 成功;其他 失败
 */
void ramdisk_release(struct gendisk *disk, fmode_t mode)
{
	printk("ramdisk release\r\n");
}

/*
 * @description		: 获取磁盘信息
 * @param - dev 	: 块设备
 * @param - geo 	: 模式
 * @return 			: 0 成功;其他 失败
 */
int ramdisk_getgeo(struct block_device *dev, struct hd_geometry *geo)
{
	/* 这是相对于机械硬盘的概念 */
	geo->heads = 2;			/* 磁头 */
	geo->cylinders = 32;	/* 柱面 */
	geo->sectors = RAMDISK_SIZE / (2 * 32 *512); /* 一个磁道上的扇区数量 */
	return 0;
}

/* 
 * 块设备操作函数 
 */
static struct block_device_operations ramdisk_fops =
{
	.owner	 = THIS_MODULE,
	.open	 = ramdisk_open,
	.release = ramdisk_release,
	.getgeo  = ramdisk_getgeo,
};

/*
 * @description	: “制造请求”函数
 * @param-q 	: 请求队列
 * @return 		: 无
 */
blk_qc_t ramdisk_make_request_fn(struct request_queue *q, struct bio *bio)
{
	int offset;
	struct bio_vec bvec;
	struct bvec_iter iter;
	unsigned long len = 0;
        
		if ((bio->bi_iter.bi_sector << 9) + bio->bi_iter.bi_size > RAMDISK_SIZE) {
                        printk(KERN_ERR RAMDISK_NAME": bad request: block=%llu, count=%u\n",
            (unsigned long long)bio->bi_iter.bi_sector,  bio->bi_iter.bi_size);
                         bio_endio(bio);
			goto done;
		}

	//set_capacity设置扇区大小 512	
	offset = (bio->bi_iter.bi_sector) << 9;	/* 获取要操作的设备的偏移地址 */

	/* 处理bio中的每个段 */
	bio_for_each_segment(bvec, bio, iter){
		char *ptr = page_address(bvec.bv_page) + bvec.bv_offset;
		len = bvec.bv_len;

		if(bio_data_dir(bio) == READ)	/* 读数据 */
			memcpy(ptr, ramdisk.ramdiskbuf + offset, len);
		else if(bio_data_dir(bio) == WRITE)	/* 写数据 */
			memcpy(ramdisk.ramdiskbuf + offset, ptr, len);
		offset += len;
	}
	//set_bit(BIO_UPTODATE, &bio->bi_flags);
	bio_endio(bio);
done:
        return BLK_QC_T_NONE;
}


/*
 * @description	: 驱动出口函数
 * @param 		: 无
 * @return 		: 无
 */
static int __init ramdisk_init(void)
{
	int ret = 0;
	printk("ramdisk init\r\n");

	/* 1、申请用于ramdisk内存 */
	ramdisk.ramdiskbuf = kzalloc(RAMDISK_SIZE, GFP_KERNEL);
	if(ramdisk.ramdiskbuf == NULL) {
		ret = -EINVAL;
		goto ram_fail;
	}

	/* 2、初始化自旋锁 */
	spin_lock_init(&ramdisk.lock);

	/* 3、注册块设备 */
	ramdisk.major = register_blkdev(0, RAMDISK_NAME); /* 由系统自动分配主设备号 */
	if(ramdisk.major < 0) {
		goto register_blkdev_fail;
	}  
	printk("ramdisk major = %d\r\n", ramdisk.major);

	/* 4、分配并初始化gendisk */
	ramdisk.gendisk = alloc_disk(RADMISK_MINOR);
	if(!ramdisk.gendisk) {
		ret = -EINVAL;
		goto gendisk_alloc_fail;
	}

	/* 5、分配请求队列 */
	ramdisk.queue = blk_alloc_queue(GFP_KERNEL);
	if(!ramdisk.queue){
		ret = -EINVAL;
		goto blk_allo_fail;
	}

	/* 6、设置“制造请求”函数 */
	blk_queue_make_request(ramdisk.queue, ramdisk_make_request_fn);

	/* 7、添加(注册)disk */
	ramdisk.gendisk->major = ramdisk.major;		/* 主设备号 */
	ramdisk.gendisk->first_minor = 0;			/* 第一个次设备号(起始次设备号) */
	ramdisk.gendisk->fops = &ramdisk_fops; 		/* 操作函数 */
	ramdisk.gendisk->private_data = &ramdisk;	/* 私有数据 */
	ramdisk.gendisk->queue = ramdisk.queue;		/* 请求队列 */
	sprintf(ramdisk.gendisk->disk_name, RAMDISK_NAME); /* 名字 */
	set_capacity(ramdisk.gendisk, RAMDISK_SIZE/512);	/* 设备容量(单位为扇区) */
	add_disk(ramdisk.gendisk);

	return 0;

blk_allo_fail:
	put_disk(ramdisk.gendisk);
	//del_gendisk(ramdisk.gendisk);
gendisk_alloc_fail:
	unregister_blkdev(ramdisk.major, RAMDISK_NAME);
register_blkdev_fail:
	kfree(ramdisk.ramdiskbuf); /* 释放内存 */
ram_fail:
	return ret;
}

/*
 * @description	: 驱动出口函数
 * @param 		: 无
 * @return 		: 无
 */
static void __exit ramdisk_exit(void)
{
	printk("ramdisk exit\r\n");
	/* 释放gendisk */
	del_gendisk(ramdisk.gendisk);
	put_disk(ramdisk.gendisk);

	/* 清除请求队列 */
	blk_cleanup_queue(ramdisk.queue);

	/* 注销块设备 */
	unregister_blkdev(ramdisk.major, RAMDISK_NAME);

	/* 释放内存 */
	kfree(ramdisk.ramdiskbuf); 
}

module_init(ramdisk_init);
module_exit(ramdisk_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("kk");
