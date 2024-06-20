

# .get_unmapped_area = ramfs_mmu_get_unmapped_area

```
static unsigned long
ramfs_mmu_get_unmapped_area(struct file *file, unsigned long addr, unsigned long len,
                            unsigned long pgoff, unsigned long flags)
{
    return current->mm->get_unmapped_area(file, addr, len, pgoff, flags);
}
```

```
const struct file_operations ramfs_file_ops = {
    .read_iter = generic_file_read_iter,
    .write_iter = generic_file_write_iter,
    .mmap = generic_file_mmap,
    .fsync = noop_fsync,
    .splice_read = generic_file_splice_read,
    .splice_write = iter_file_splice_write,
    .llseek = generic_file_llseek,
    .get_unmapped_area = ramfs_mmu_get_unmapped_area,
};
```

# insmod  ramfs.ko 

```
[root@centos7 ramfs-loadable-module]# insmod  ramfs.ko 
[root@centos7 ramfs-loadable-module]#  mount -t myramfs none  /mnt/ram
[root@centos7 ramfs-loadable-module]# touch /mnt/ram/ram.txt
[root@centos7 ramfs-loadable-module]# echo 'hello world' > /mnt/ram/ram.txt
[root@centos7 ramfs-loadable-module]#  cat  /mnt/ram/ram.txt
hello world
[root@centos7 ramfs-loadable-module]# 
```

```
[root@centos7 simple]# gcc test-unmap-area1.c  -o test-unmap-area1
[root@centos7 simple]# ./test-unmap-area1 

Write/Read test ...
```

```
[347456.985551] [<ffff000008089e14>] dump_backtrace+0x0/0x23c
[347456.991012] [<ffff00000808a074>] show_stack+0x24/0x2c
[347456.996127] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[347457.001243] [<ffff000000f3002c>] ramfs_mmu_get_unmapped_area+0x2c/0x5c [ramfs]
[347457.008519] [<ffff000008254a2c>] get_unmapped_area.part.37+0x60/0xd0
[347457.014929] [<ffff000008257f74>] do_mmap+0x12c/0x34c
[347457.019958] [<ffff00000823467c>] vm_mmap_pgoff+0xf0/0x124
[347457.025418] [<ffff000008255a7c>] SyS_mmap_pgoff+0xc0/0x23c
[347457.030965] [<ffff000008089548>] sys_mmap+0x54/0x68
```

# truncate
truncate 通过simple_setattr实现   
```
const struct inode_operations ramfs_file_inode_ops = {
    .setattr = simple_setattr,
    .getattr = simple_getattr,
};
```