#include "header.h"
#include "node.h"
#include "dentry.h"
#include "supper.h"
#include "file.h"

// aufs文件系统的挂载点
struct vfsmount* aufs_mount;

// 对应具体打开文件的文件操作方式
static struct file_operations aufs_file_operations = {
    .read = aufs_file_read,
    .write = aufs_file_write,
};

// 初始化aufs文件系统的 file_system_type结构，每个文件系统对应一个这样的结构体，主要用于提供具体文件系统的的信息，以及操作的方法
static struct file_system_type aufs_type = {
    .name = "aufs",
    .mount = aufs_get_sb,
    .kill_sb = kill_litter_super,
};

// 创建aufs文件系统，同时创建对应的文件夹和文件
static int __init aufs_init(void)
{
    int ret = 0;
    struct dentry* pslot = NULL;
     
    ret = register_filesystem(&aufs_type);
    if (ret) {
        printk(KERN_ERR "aufs: cannot register file system\\n");
        return ret;
    }
 
    aufs_mount = kern_mount(&aufs_type);
    if (IS_ERR(aufs_mount)) {
        printk(KERN_ERR "aufs: cannot mount file system\\n");
        unregister_filesystem(&aufs_type);
        return ret;
    }
 
    pslot = aufs_create_dir("woman_star", NULL); // 创建woman_star文件系统，返回所创建文件夹的dentry
    aufs_create_file("lbb", S_IFREG | S_IRUGO, pslot, NULL, &aufs_file_operations);// 在对应的文件夹下，创建具体的文件
    aufs_create_file("fbb", S_IFREG | S_IRUGO, pslot, NULL, &aufs_file_operations);
    aufs_create_file("lj1", S_IFREG | S_IRUGO, pslot, NULL, &aufs_file_operations);
 
    pslot = aufs_create_dir("man_star", NULL);
    aufs_create_file("ldh", S_IFREG | S_IRUGO, pslot, NULL, &aufs_file_operations);
    aufs_create_file("lcw", S_IFREG | S_IRUGO, pslot, NULL, &aufs_file_operations);
    aufs_create_file("jw",  S_IFREG | S_IRUGO, pslot, NULL, &aufs_file_operations);
 
    return ret;
}

// 卸载aufs文件系统
static void __exit aufs_exit(void)
{
    kern_unmount(aufs_mount);
    unregister_filesystem(&aufs_type);
    aufs_mount = NULL;
}
 
module_init(aufs_init);
module_exit(aufs_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("This is a simple aufs fs module");
MODULE_VERSION("Ver 0.1");
