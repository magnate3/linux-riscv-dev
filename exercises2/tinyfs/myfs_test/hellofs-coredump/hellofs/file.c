#include <linux/sched/signal.h>
#include<asm-generic/resource.h>
#include "khellofs.h"

ssize_t hellofs_read(struct file *filp, char __user *buf, size_t len,
                     loff_t *ppos) {
    struct super_block *sb;
    struct inode *inode;
    struct hellofs_inode *hellofs_inode;
    struct buffer_head *bh;
    char *buffer;
    int nbytes;

    inode = filp->f_path.dentry->d_inode;
    sb = inode->i_sb;
    hellofs_inode = HELLOFS_INODE(inode);
    
    if (*ppos >= hellofs_inode->file_size) {
        return 0;
    }

    bh = sb_bread(sb, hellofs_inode->data_block_no);
    if (!bh) {
        printk(KERN_ERR "Failed to read data block %llu\n",
               hellofs_inode->data_block_no);
        return 0;
    }

    buffer = (char *)bh->b_data + *ppos;
    nbytes = min((size_t)(hellofs_inode->file_size - *ppos), len);

    if (copy_to_user(buf, buffer, nbytes)) {
        brelse(bh);
        printk(KERN_ERR
               "Error copying file content to userspace buffer\n");
        return -EFAULT;
    }

    brelse(bh);
    *ppos += nbytes;
    return nbytes;
}

inline int test_generic_write_checks(struct file *file, loff_t *pos, size_t *count, int isblk)
{
	struct inode *inode = file->f_mapping->host;
	//unsigned long limit = current->signal->rlim[RLIMIT_STACK].rlim_cur;
	unsigned long limit = current->signal->rlim[RLIMIT_FSIZE].rlim_cur;

        if (unlikely(*pos < 0))
                return -EINVAL;

	if (!isblk) {
		/* FIXME: this is for backwards compatibility with 2.4 */
		if (file->f_flags & O_APPEND)
                        *pos = i_size_read(inode);

		if (limit != RLIM_INFINITY) {
			if (*pos >= limit) {
				send_sig(SIGXFSZ, current, 0);
				return -EFBIG;
			}
			if (*count > limit - (typeof(limit))*pos) {
				*count = limit - (typeof(limit))*pos;
			}
		}
	}

	/*
 * 	 * LFS rule
 * 	 	 */
	if (unlikely(*pos + *count > MAX_NON_LFS &&
				!(file->f_flags & O_LARGEFILE))) {
		if (*pos >= MAX_NON_LFS) {
			return -EFBIG;
		}
		if (*count > MAX_NON_LFS - (unsigned long)*pos) {
			*count = MAX_NON_LFS - (unsigned long)*pos;
		}
	}

	/*
 * 	 * Are we about to exceed the fs block limit ?
 * 	 	 *
 * 	 	 	 * If we have written data it becomes a short write.  If we have
 * 	 	 	 	 * exceeded without writing data we send a signal and return EFBIG.
 * 	 	 	 	 	 * Linus frestrict idea will clean these up nicely..
 * 	 	 	 	 	 	 */
	if (likely(!isblk)) {
		if (unlikely(*pos >= inode->i_sb->s_maxbytes)) {
			if (*count || *pos > inode->i_sb->s_maxbytes) {
				return -EFBIG;
			}
			/* zero-length writes at ->s_maxbytes are OK */
		}

		if (unlikely(*pos + *count > inode->i_sb->s_maxbytes))
			*count = inode->i_sb->s_maxbytes - *pos;
	} else {
#ifdef CONFIG_BLOCK
		loff_t isize;
		if (bdev_read_only(I_BDEV(inode)))
			return -EPERM;
		isize = i_size_read(inode);
		if (*pos >= isize) {
			if (*count || *pos > isize)
				return -ENOSPC;
		}

		if (*pos + *count > isize)
			*count = isize - *pos;
#else
		return -EPERM;
#endif
	}
	return 0;
}
/* TODO We didn't use address_space/pagecache here.
   If we hook file_operations.write = do_sync_write,
   and file_operations.aio_write = generic_file_aio_write,
   we will use write to pagecache instead. */
ssize_t hellofs_write(struct file *filp, const char __user *buf, size_t len,
                      loff_t *ppos) {
    struct super_block *sb;
    struct inode *inode;
    struct hellofs_inode *hellofs_inode;
    struct buffer_head *bh;
    struct hellofs_superblock *hellofs_sb;
    char *buffer;
    int ret;

    inode = filp->f_path.dentry->d_inode;
    sb = inode->i_sb;
    hellofs_inode = HELLOFS_INODE(inode);
    hellofs_sb = HELLOFS_SB(sb);

    ret = test_generic_write_checks(filp, ppos, &len, 0);
    if (ret) {
        return ret;
    }

    bh = sb_bread(sb, hellofs_inode->data_block_no);
    if (!bh) {
        printk(KERN_ERR "Failed to read data block %llu\n",
               hellofs_inode->data_block_no);
        return 0;
    }

    buffer = (char *)bh->b_data + *ppos;
    if (copy_from_user(buffer, buf, len)) {
        brelse(bh);
        printk(KERN_ERR
               "Error copying file content from userspace buffer "
               "to kernel space\n");
        return -EFAULT;
    }
    *ppos += len;

    mark_buffer_dirty(bh);
    sync_dirty_buffer(bh);
    brelse(bh);

    hellofs_inode->file_size = max((size_t)(hellofs_inode->file_size),
                                   (size_t)(*ppos));
    hellofs_save_hellofs_inode(sb, hellofs_inode);

    /* TODO We didn't update file size here. To be frank I don't know how. */

    return len;
}
