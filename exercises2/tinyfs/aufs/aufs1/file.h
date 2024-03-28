#ifndef __FILE_H__
#define __FILE_H__

#include "header.h"
#include "node.h"
#include "dentry.h"

int enabled = 1;

// 在aufs文件系统中创建文件
struct dentry* aufs_create_file(const char *name, mode_t mode,
            struct dentry* parent, void *data,
            struct file_operations* fops);

// 在aufs文件系统中创建一个文件夹
struct dentry* aufs_create_dir(const char* name, struct dentry* parent);

// 对应于打开的aufs文件的读取方法
ssize_t aufs_file_read(struct file* fle, char __user *buf, size_t nbytes, loff_t *ppos);

// 对应于打开的aufs文件的写入方法
ssize_t aufs_file_write(struct file* file, const char* __user buffer, size_t count, loff_t* ppos);

struct dentry* aufs_create_file(const char *name, mode_t mode,
            struct dentry* parent, void *data,
            struct file_operations* fops)
{
    struct dentry* dentry = NULL;
    int error = 0;
 
    printk("aufs: creating file \'%s\'", name);
     
    error = aufs_create_by_name(name, mode, parent, &dentry);
    if (error) {
        dentry = NULL;
        goto exit;
    }
 
    if (dentry->d_inode) {
        if (data)
            dentry->d_inode->i_private = data;
        if (fops)
            dentry->d_inode->i_fop = fops;
    }

exit:
    return dentry;
}

struct dentry* aufs_create_dir(const char* name, struct dentry* parent)
{
    return aufs_create_file(name, S_IFDIR | S_IRWXU | S_IRUGO, parent, NULL, NULL);
}

ssize_t aufs_file_read(struct file* fle, char __user *buf, size_t nbytes, loff_t *ppos)
{
    char *s = enabled ? "aufs read enabled\\n" : "aufs read disabled\\n";
    dump_stack();
    return simple_read_from_buffer(buf, nbytes, ppos, s, strlen(s));
}

ssize_t aufs_file_write(struct file* file, const char* __user buffer, size_t count, loff_t* ppos)
{
    int res = *buffer - '0';
 
    if (res)
        enabled = 1;
    else
        enabled = 0;
 
    return count;
}

#endif /* __FILE_H__ */
