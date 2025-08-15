#ifndef __DENTRY_H__
#define __DENTRY_H__

#include "header.h"

extern struct vfsmount* aufs_mount;

// 根据父dentry、mode、name创建子dentry
int aufs_create_by_name(const char* name, mode_t mode, struct dentry* parent, struct dentry** dentry);

int aufs_create_by_name(const char* name, mode_t mode, struct dentry* parent, struct dentry** dentry)
{
    int error = 0;
 
    if (!parent) {
        if (aufs_mount && aufs_mount->mnt_sb) {
            parent = aufs_mount->mnt_sb->s_root;
        }
    }
 
    if (!parent) {
        printk("Ah! can not find a parent!\\n");
        return -EFAULT;
    }
 
    *dentry = NULL;
    *dentry = lookup_one_len(name, parent, strlen(name));
    if (!IS_ERR(dentry)) {
        if ((mode & S_IFMT) == S_IFDIR)
            error = aufs_mkdir(parent->d_inode, *dentry, mode);
        else
            error = aufs_create(parent->d_inode, *dentry, mode);
    } else
        error = PTR_ERR(dentry);
 
    return error;
}

#endif /* __DENTRY_H__ */
