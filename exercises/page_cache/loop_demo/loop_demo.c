#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/blkdev.h>
#include <linux/slab.h>
#include <linux/genhd.h>
#include <linux/sched.h>

#include "loop_demo.h"

#define LOOP_DEMO_MAJOR	199
#define LOOP_DEMO_NAME	"loop-demo"

const unsigned int max_loop	= 1;

struct loop_demo_device* loop_dev = NULL;
struct gendisk**         disks	  = NULL;

static int loop_demo_open(struct inode* node,struct file* fp)
{
	struct loop_demo_device* lo = (struct loop_demo_device*)node->i_bdev->bd_disk->private_data;

	mutex_lock(&lo->lo_ctl_mutex);
	lo->lo_refcnt++;
	mutex_unlock(&lo->lo_ctl_mutex);
	return 0;
}

static int loop_demo_close(struct inode* node,struct file* fp)
{
	struct loop_demo_device* lo = (struct loop_demo_device*)node->i_bdev->bd_disk->private_data;

	mutex_lock(&lo->lo_ctl_mutex);
	lo->lo_refcnt --;
	mutex_unlock(&lo->lo_ctl_mutex);
	return 0;
}
static inline int is_loop_device(struct file* fp)
{
	struct inode* i = fp->f_mapping->host;
	return i && S_ISBLK(i->i_mode) && MAJOR(i->i_rdev) == LOOP_DEMO_MAJOR;
}

static loff_t get_loop_size(struct loop_demo_device*lo,struct file* fp)
{
	loff_t loopsize = i_size_read(fp->f_mapping->host);

	return loopsize >> 9;
}

static void loop_add_bio(struct loop_demo_device* lo,struct bio* bio)
{
	if(lo->lo_biotail)
	{
		lo->lo_biotail->bi_next = bio;
		lo->lo_biotail = bio;
	}
	else
	{
		lo->lo_bio = lo->lo_biotail = bio;
	}
}

static struct bio* loop_get_bio(struct loop_demo_device* lo)
{
	struct bio* bio = NULL;
	if((bio = lo->lo_bio))
	{
		if(bio == lo->lo_biotail)
			lo->lo_biotail = NULL;

		lo->lo_bio = bio->bi_next;
		bio->bi_next = NULL;
	}
	return bio;
}
static int transfer_none(struct loop_demo_device* lo,int cmd,
		struct page* raw_page,unsigned long raw_off,
		struct page* loop_page,unsigned long loop_off,
		int size,sector_t real_block)
{
	char* raw_buf = kmap_atomic(raw_page,KM_USER0) + raw_off;
	char* loop_buf = kmap_atomic(loop_page,KM_USER1) + loop_off;

	printk("[OS]=[%s:%d]=cmd:%d,size:%d,raw_off:%d,loop_off:%d\n",__func__,__LINE__,cmd,size,raw_off,loop_off);
	if(cmd == READ)
		memcpy(loop_buf,raw_buf,size);
	else
		memcpy(raw_buf,loop_buf,size);

	kunmap_atomic(raw_buf,KM_USER0);
	kunmap_atomic(loop_buf,KM_USER1);
	cond_resched();
	return 0;
}
struct lo_read_data 
{
	struct loop_demo_device* lo;
	struct page* page;
	unsigned long offset;
	int bsize;
};

static int lo_read_actor(read_descriptor_t* desc,struct page* page,
		unsigned long offset,unsigned long size)
{
	unsigned long count = desc->count;
	struct lo_read_data* p = desc->arg.data;
	struct loop_demo_device* lo = p->lo;
	sector_t IV = 0;
	IV = ((sector_t)page->index << (PAGE_CACHE_SHIFT - 9)) + (offset >> 9);

	if(size > count)
		size = count;
	int ret = transfer_none(lo,READ,page,offset,p->page,p->offset,size,IV);
	if(ret != 0)
	{
		size = 0;
		desc->error = -EINVAL;
	}
	flush_dcache_page(p->page);
	desc->count = count - size;
	desc->written += size;
	p->offset += size;
	return size;

}

#if 1
static int do_lo_read(struct loop_demo_device* lo,struct bio_vec* bvec,int bsize,loff_t pos)
{
	struct lo_read_data cookie = {0};
	struct file* file = NULL;
	int retval = 0;

	cookie.lo = lo;
	cookie.page = bvec->bv_page;
	cookie.offset = bvec->bv_offset;
	cookie.bsize = bsize;
	file = lo->lo_backing_file;
	retval = file->f_op->sendfile(file,&pos,bvec->bv_len,lo_read_actor,&cookie);;
	printk("[OS]=[%s:%d]=retval:%d\n",__func__,__LINE__,retval);
	return (retval < 0) ? retval : 0;
}
#else
static int do_lo_read(struct loop_demo_device* lo,struct bio_vec* bvec,int bsize,loff_t pos)
{
	int retval = 0;
	struct file* file = lo->lo_backing_file;
	retval = file->f_op->read(file,bvec->bv_page,bvec->bv_len,&pos);
	printk("[OS]=[%s:%d]==retval:%d\n",__func__,__LINE__,retval);
	return (retval != bvec->bv_len) ? -1 : 0;		
}
#endif


static int loop_do_read_handle(struct loop_demo_device* lo,struct bio* bio)
{
	struct bio_vec* bvec = NULL;
	int i = 0,ret = 0;
	loff_t pos = bio->bi_sector << 9;
	bio_for_each_segment(bvec,bio,i)
	{
		ret = do_lo_read(lo,bvec,lo->lo_blocksize,pos);
		if(ret < 0)
			break;
		pos += bvec->bv_len;
	}
	return ret;
}

static int do_lo_write(struct loop_demo_device* lo,struct bio_vec* bvec,int bsize,loff_t pos,struct page* page)
{
	struct file* fp = lo->lo_backing_file;
	struct address_space* mapping = fp->f_mapping;
	const struct address_space_operations* aops = mapping->a_ops;

	pgoff_t index = 0;
	unsigned long offset = 0,bv_offs = 0;
	int len = 0,ret = 0;

	mutex_lock(&mapping->host->i_mutex);
	index = pos >> PAGE_CACHE_SHIFT;
	offset = pos & ((pgoff_t)PAGE_CACHE_SIZE - 1);
	bv_offs = bvec->bv_offset;
	len = bvec->bv_len;
	while(len > 0)
	{
		sector_t IV;
		unsigned long size = 0;
		int transfer_result = 0;
		IV = ((sector_t)index << (PAGE_CACHE_SHIFT-9)) + (offset >> 9);
		size = PAGE_CACHE_SIZE - offset;

		if(size > len)
			size = len;

		page = grab_cache_page(mapping,index);
		if(unlikely(!page))
			goto fail;

		ret = aops->prepare_write(fp,page,offset,offset+size);
		printk("[OS]=[%s:%d]=ret:%d\n",__func__,__LINE__,ret);
		if(unlikely(ret))
		{
			if(ret == AOP_TRUNCATED_PAGE)
			{
				page_cache_release(page);
				continue;
			}
			goto unlock;
		}
		transfer_result = transfer_none(lo,WRITE,page,offset,bvec->bv_page,bv_offs,size,IV);
		printk("[OS]=[%s:%d]=transfer_result:%d\n",__func__,__LINE__,transfer_result);
		if(unlikely(transfer_result))
		{
			char* kaddr = NULL;
			kaddr = kmap_atomic(page,KM_USER0);
			memset(kaddr+offset,0,size);
			kunmap_atomic(kaddr,KM_USER0);
		}
		flush_dcache_page(page);

		ret = aops->commit_write(fp,page,offset,offset + size);
		printk("[OS]=[%s:%d]=ret:%d\n",__func__,__LINE__,ret);
		if(unlikely(ret))
		{
			if(ret == AOP_TRUNCATED_PAGE)
			{
				page_cache_release(page);
				continue;
			}
			goto unlock;
		}
		if(unlikely(transfer_result))
			goto unlock;

		bv_offs += size;
		len -= size;
		offset = 0;
		index ++;
		pos += size;
		unlock_page(page);
		page_cache_release(page);

	}
	ret = 0;

OUT:
	mutex_unlock(&mapping->host->i_mutex);
	return ret;
unlock:
	unlock_page(page);
	page_cache_release(page);
fail:
	ret = -1;
	goto OUT;
}

static int loop_do_write_handle(struct loop_demo_device* lo,struct bio* bio)
{
	struct bio_vec* bvec = NULL;
	struct page* page = NULL;
	int i = 0,ret = 0;
	loff_t pos = bio->bi_sector << 9;
	bio_for_each_segment(bvec,bio,i)
	{
		ret = do_lo_write(lo,bvec,lo->lo_blocksize,pos,page);
		if (ret < 0)
			break;

		pos += bvec->bv_len;
	}
	return ret;
}

static void loop_demo_handle_bio(struct loop_demo_device* lo,struct bio* bio)
{
//	printk("[OS]=[%s:%s:%d] bi_size:%d\n",__FILE__,__func__,__LINE__,bio->bi_size);

	/*
	 if programming does not platform this function ,the mkfs.ext2 or mount process will hunge up.
	 */
	int ret = 0;
	printk("[OS]=[%s:%d]=direct:%d,WRITE:%d,READ:%d\n",__func__,__LINE__,bio_rw(bio),WRITE,READ);
	if(bio_rw(bio) == WRITE)
		ret = loop_do_write_handle(lo,bio);
	else
		ret = loop_do_read_handle(lo,bio);
	//printk("ret = %d",ret);
	bio_endio(bio,bio->bi_size,ret);
}


static int loop_demo_make_request(request_queue_t* q,struct bio* old_bio)
{
	struct loop_demo_device* lo = q->queuedata;
	int rw = bio_rw(old_bio);
	if(rw == READA)
		rw = READ;

	BUG_ON(!lo || (rw != READ && rw != WRITE));
	spin_lock_irq(&lo->lo_lock);

	if(lo->lo_state != Lo_bound)
		goto OUT;

	if(unlikely(rw == WRITE && lo->lo_flags & LO_FLAGS_READ_ONLY))
		goto OUT;
	/*
	   add one  bio to bio list
	 */
	loop_add_bio(lo,old_bio);

	lo->lo_pending ++;

	spin_unlock_irq(&lo->lo_lock);
	complete(&lo->lo_bh_done);
	return 0;

OUT:
	if(lo->lo_pending == 0)
		complete(&lo->lo_bh_done);

	printk("[error]=[%s:%s:%d]\n",__FILE__,__func__,__LINE__);
	spin_unlock_irq(&lo->lo_lock);

	bio_io_error(old_bio,old_bio->bi_size);
}

static int loop_demo_thread_handle(void* data)
{
	struct loop_demo_device * lo = (struct loop_demo_device*)data;
	struct bio* bio = NULL;	
	current->flags |= PF_NOFREEZE;

	set_user_nice(current,-20);

	lo->lo_state	= Lo_bound;
	lo->lo_pending = 1;

	complete(&lo->lo_done);
	printk("[OS]=after complete [%s:%s:%d]\n",__FILE__,__func__,__LINE__);
	while(1)
	{
		int pending = 0;
		if(wait_for_completion_interruptible(&lo->lo_bh_done))
			continue;


		spin_lock_irq(&lo->lo_lock);
		if(unlikely(!lo->lo_pending))
		{
			spin_unlock_irq(&lo->lo_lock);
			break;
		}
		/*
		   get bio from bio list.
		 */
		bio = loop_get_bio(lo);
		lo->lo_pending --;
		pending = lo->lo_pending;
		
		spin_unlock_irq(&lo->lo_lock);

		BUG_ON(!bio);

		/*handle bio : read or write*/
		loop_demo_handle_bio(lo,bio);

		if(unlikely(!pending))
			break;
	}
	complete(&lo->lo_done); //why???
	return 0;
}

static int loop_demo_set_fd(struct loop_demo_device* lo,struct block_device* bdev,struct file* lo_file,unsigned long arg)
{
	if(lo->lo_state != Lo_unbound)
		return -EBUSY;
	struct file* fp = fget(arg);
	if(!fp)
		return -EBADF;

	struct address_space* mapping = fp->f_mapping;
	struct inode* node = mapping->host;

	int lo_flags = 0;
	if(!(fp->f_mode & FMODE_WRITE))
		lo_flags = LO_FLAGS_READ_ONLY;

	int err = -EINVAL;
	if(S_ISREG(node->i_mode) || S_ISBLK(node->i_mode))
	{
		const struct address_space_operations* aops = mapping->a_ops;
		if (!fp->f_op->sendfile)/*can`t read*/
			return err;

		if(aops->prepare_write && aops->commit_write)
			lo_flags |= LO_FLAGS_USE_AOPS;

		if(!(lo_flags & LO_FLAGS_USE_AOPS))
			lo_flags |= LO_FLAGS_READ_ONLY;
		
	}
	else
		return err;
	printk("[OS]=[%s:%d]=blocksize:%d\n",__func__,__LINE__,node->i_blksize);

	lo->lo_blocksize = node->i_blksize;
	lo->lo_device = bdev;
	lo->lo_backing_file = fp;
	lo->lo_flags = lo_flags;
	lo->lo_bio = lo->lo_biotail = NULL;

	blk_queue_make_request(lo->lo_queue,loop_demo_make_request);

	lo->lo_queue->queuedata = lo;

	set_capacity(disks[lo->lo_index],get_loop_size(lo,fp));
	set_blocksize(bdev,lo->lo_blocksize);

	err = kernel_thread(loop_demo_thread_handle,lo,CLONE_KERNEL);
	if(err < 0)
		return -1;

	printk("[OS]=[%s:%s:%d]--before wait_for_completion...\n",__FILE__,__func__,__LINE__);
	wait_for_completion(&lo->lo_done);
	printk("[OS]=[%s:%s:%d]--after wait_for_completion...\n",__FILE__,__func__,__LINE__);
		
	return 0;
}

static int loop_demo_clr_fd(struct loop_demo_device* lo,struct block_device* bdev,struct file* lo_file,unsigned long arg)
{
	return 0;
}

static int loop_demo_ioctl(struct inode* node,struct file* fp,unsigned int cmd,unsigned long arg)
{
	struct loop_demo_device* lo = (struct loop_demo_device*)node->i_bdev->bd_disk->private_data;
	int err = 0;
	mutex_lock(&lo->lo_ctl_mutex);
	switch(cmd)
	{
		case LOOP_SET_FD:
		{
			err = loop_demo_set_fd(lo,node->i_bdev,fp,arg);
			break;
		}
		case LOOP_CLR_FD:
		{
			err = loop_demo_clr_fd(lo,node->i_bdev,fp,arg);
			break;
		}
		case LOOP_CHANGE_FD:
		{
			break;
		}
		case LOOP_SET_STATUS64:
		{
			break;
		}
		case LOOP_GET_STATUS64:
		{
			break;
		}
		default:
			mutex_unlock(&lo->lo_ctl_mutex);
			printk("[Error] [%s:%d] cmd:0x%x does not support!\n ",__func__,__LINE__,cmd);
			return -1;
	}
	mutex_unlock(&lo->lo_ctl_mutex);
	return err;
}

static struct block_device_operations blk_ops = {
	.owner		= THIS_MODULE,
	.open 		= loop_demo_open,
	.release	= loop_demo_close,
	.ioctl		= loop_demo_ioctl,
};

static int __init loop_demo_module_init(void)
{
	if(register_blkdev(LOOP_DEMO_MAJOR,LOOP_DEMO_NAME))
		return -1;
	loop_dev = kmalloc(sizeof(struct loop_demo_device) * max_loop,GFP_KERNEL);
	if(!loop_dev)
		goto REGISTER_ERROR;

	disks = kmalloc(sizeof(struct gendisk*) * max_loop,GFP_KERNEL);
	if(!disks)
		goto MEM_OUT1; 

	unsigned int i = 0;
	for(i = 0; i < max_loop; i++)
	{
		disks[i] = alloc_disk(1);
		if(!disks[i])
			goto MEM_OUT3;
	}

	for(i = 0; i < max_loop; i++)
	{
		struct loop_demo_device* lo = &loop_dev[i];
		struct gendisk* gd = disks[i];

		memset(lo,0,sizeof(struct loop_demo_device));
		lo->lo_queue = blk_alloc_queue(GFP_KERNEL);
		if(!lo->lo_queue)
			goto MEM_OUT4;

		lo->lo_refcnt = 0;
		lo->lo_state = Lo_unbound;
		lo->lo_index	= i;
		mutex_init(&lo->lo_ctl_mutex);

		init_completion(&lo->lo_done);
		init_completion(&lo->lo_bh_done);

		spin_lock_init(&lo->lo_lock);

		gd->major = LOOP_DEMO_MAJOR;
		gd->first_minor = i;
		gd->fops = &blk_ops;
		sprintf(gd->disk_name,"loop-demo%d",i);

		gd->queue = lo->lo_queue;
		gd->private_data = lo;
	}

	for(i = 0; i < max_loop; i++)
		add_disk(disks[i]);

	return 0;
MEM_OUT4:
	while(--i)
		blk_cleanup_queue(loop_dev[i].lo_queue);
	i = max_loop;

MEM_OUT3:
	while(--i)
		put_disk(disks[i]);

MEM_OUT2:
	kfree(disks);
MEM_OUT1:
	kfree(loop_dev);

REGISTER_ERROR:
	unregister_blkdev(LOOP_DEMO_MAJOR,LOOP_DEMO_NAME);

	return -1;
}

static void __exit loop_demo_module_exit(void)
{
	unsigned int i = 0;

	for(i = 0; i < max_loop; i ++)
	{
		del_gendisk(disks[i]);
		blk_cleanup_queue(loop_dev[i].lo_queue);
		put_disk(disks[i]);
	}

	unregister_blkdev(LOOP_DEMO_MAJOR,LOOP_DEMO_NAME);
	kfree(disks);

	kfree(loop_dev);

}

module_init(loop_demo_module_init);

module_exit(loop_demo_module_exit);

MODULE_LICENSE("GPL");

