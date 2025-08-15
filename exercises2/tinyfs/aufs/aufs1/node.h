#ifndef __NODE_H__
#define __NODE_H__

#include "header.h"

// 根据创建的aufs文件系统的 super_block创建具体的inode结构体
struct inode* aufs_get_inode(struct super_block* sb, int mode, dev_t dev);

// 把创建的inode和dentry结构体连接起来
int aufs_mknod(struct inode* dir, struct dentry* dentry, int mode, dev_t dev);

int aufs_mkdir(struct inode* dir, struct dentry* dentry, int mode);

int aufs_create(struct inode* dir, struct dentry* dentry, int mode);

struct inode* aufs_get_inode(struct super_block* sb, int mode, dev_t dev)
{
    struct inode* inode = new_inode(sb);
 
    if (inode) {
        inode->i_mode = mode;
        inode->i_uid = current_fsuid();
        inode->i_gid = current_fsgid();
        inode->i_blocks = 0;
        inode->i_atime = inode->i_mtime = inode->i_ctime;
        switch (mode & S_IFMT) {
            default:
                init_special_inode(inode, mode, dev);
                break;
            case S_IFREG:
                printk("create a file \\n");
                break;
            case S_IFDIR:
                inode->i_op = &simple_dir_inode_operations;
                inode->i_fop = &simple_dir_operations;
                printk("creat a dir file \\n");
                 
                inode->__i_nlink++;
                break;
        }
    }
 
    return inode;
}

int aufs_mknod(struct inode* dir, struct dentry* dentry, int mode, dev_t dev)
{
    struct inode*  inode;
    int error = -EPERM;
 
    if (dentry->d_inode)
        return -EEXIST;
    inode = aufs_get_inode(dir->i_sb, mode, dev);
    if (inode) {
        d_instantiate(dentry, inode);
        dget(dentry);
        error = 0;
    }
     
    return error;
}
 
int aufs_mkdir(struct inode* dir, struct dentry* dentry, int mode)
{
    int res;
 
    res = aufs_mknod(dir, dentry, mode | S_IFDIR, 0);
    if (!res) {
        dir->__i_nlink++;
    }
 
    return res;
}
 
int aufs_create(struct inode* dir, struct dentry* dentry, int mode)
{
    return aufs_mknod(dir, dentry, mode | S_IFREG, 0);
}

#endif /* __NODE_H__ */
