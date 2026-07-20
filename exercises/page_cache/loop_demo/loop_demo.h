#ifndef _LOOP_DEMO_H_
#define _LOOP_DEMO_H_

#include <linux/spinlock.h>
#include <linux/fs.h>
#include <linux/mutex.h>
#include <linux/blkdev.h>


#define LOOP_NAME_SIZE 64

typedef enum _loop_state_ 
{
	Lo_unbound,
	Lo_bound,
	Lo_rundown
}loop_state;

struct loop_demo_device
{
	loop_state					lo_state;
	int							lo_refcnt;
	int 						lo_flags;
	char						lo_file_name[LOOP_NAME_SIZE];
	struct completion			lo_done;
	struct completion			lo_bh_done;
	struct mutex				lo_ctl_mutex;
	int							lo_pending;

	unsigned int				lo_index;
	unsigned long 				lo_blocksize;
	struct file* 				lo_backing_file;
	struct block_device* 		lo_device;

	spinlock_t					lo_lock;
	struct bio* 				lo_bio;
	struct bio*					lo_biotail;

	request_queue_t*			lo_queue;
};

enum 
{
		LO_FLAGS_READ_ONLY = 1 << 0,
		LO_FLAGS_USE_AOPS = 1 << 1,
		LO_FLAGS_USE_DERICT = 1 << 2,
};


enum
{
 	LOOP_SET_FD		= 0X4C00,
	LOOP_CLR_FD	,
	LOOP_SET_STATUS	,
	LOOP_GET_STATUS ,
	LOOP_SET_STATUS64 ,
	LOOP_GET_STATUS64 ,
	LOOP_CHANGE_FD	,
};


#endif

