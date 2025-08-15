#ifndef __SUPPER_H__
#define __SUPPER_H__

#include "header.h"

// 每个文件系统需要一个MAGIC number
#define AUFS_MAGIC 0x64668735

// 用于填充aufs的super_block
int aufs_fill_super(struct super_block* sb, void *data, int silent);

// 创建aufs文件系统的对应的根目录的dentry
struct dentry* aufs_get_sb(struct file_system_type* fs_type, int flags, const char* dev_name, void* data);

int aufs_fill_super(struct super_block* sb, void* data, int silent)
{
    static struct tree_descr debug_files[] = {{""}};
 
    return simple_fill_super(sb, AUFS_MAGIC, debug_files);
}

struct dentry* aufs_get_sb(struct file_system_type* fs_type,
        int flags, const char* dev_name, void* data)
{
    return mount_single(fs_type, flags, data, aufs_fill_super);
}

#endif /* __SUPPER_H__ */
