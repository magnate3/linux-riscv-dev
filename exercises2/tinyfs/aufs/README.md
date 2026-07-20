
# run 

```
[root@centos7 aufs]# ls
aufs.c  aufs.ko  aufs.mod.c  aufs.mod.o  aufs.o  dentry.h  file.h  header.h  Makefile  modules.order  Module.symvers  node.h  supper.h
[root@centos7 aufs]# mkdir test_dir
[root@centos7 aufs]# mount -t aufs none test_dir/
mount: unknown filesystem type 'aufs'
[root@centos7 aufs]# insmod  aufs.ko 
[root@centos7 aufs]# mount -t aufs none test_dir/
[root@centos7 aufs]# ls test_dir/
man_star  woman_star
[root@centos7 aufs]# cat test_dir/man_star/jw  | more
aufs read enabled\n
[root@centos7 aufs]# 
```

```
[797028.104143] Call trace:
[797028.106665] [<ffff000008089e14>] dump_backtrace+0x0/0x23c
[797028.112126] [<ffff00000808a074>] show_stack+0x24/0x2c
[797028.117240] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[797028.122355] [<ffff000001630128>] aufs_file_read+0x38/0x88 [aufs]
[797028.128420] [<ffff0000082b22a4>] __vfs_read+0x58/0x178
[797028.133622] [<ffff0000082b2454>] vfs_read+0x90/0x14c
[797028.138649] [<ffff0000082b2b70>] SyS_read+0x60/0xc0
```

```
ssize_t simple_read_from_buffer(void __user *to, size_t count, loff_t *ppos,
                                const void *from, size_t available)
{
        loff_t pos = *ppos;
        size_t ret;

        if (pos < 0)
                return -EINVAL;
        if (pos >= available || !count)
                return 0;
        if (count > available - pos)
                count = available - pos;
        ret = copy_to_user(to, from + pos, count);
        if (ret == count)
                return -EFAULT;
        count -= ret;
        *ppos = pos + count;
        return count;
}
EXPORT_SYMBOL(simple_read_from_buffer);
```
# myfs

```
[root@centos7 myfs2]# insmod myfs.ko
[root@centos7 mnt]# mount -t myfs none /mnt/myfs
[root@centos7 mnt]# ls /mnt/myfs/
[root@centos7 mnt]# cd /mnt/myfs/
[root@centos7 myfs]# pwd
/mnt/myfs
[root@centos7 myfs]# touch myfile
[root@centos7 myfs]#  echo "message" > myfile 
[root@centos7 myfs]# cat myfile 
message
[root@centos7 myfs]# cd ~/programming/kernel/myfs/myfs2
[root@centos7 myfs2]# umount /mnt/myfs
[root@centos7 myfs2]# 
```

# linux_file_system_course
+ 介绍   
linux环境下的文件磁盘系统编写教程。最近在配置网络文件系统nfs，然后又发现了国产分布式文件系统fastdfs，于是十分好奇这个东西到底是什么原理，linux的文件系统是怎么搞出来的，可以不可以用Python语言配合C语言搞一个linux的磁盘文件系统呢，那种可以mount的，于是一查，还真有教程，感觉比较靠谱，这里转载下，也做收藏。   

+ 说明   
教程PPT原地址： https://download.samba.org/pub/samba/cifs-cvs/ols2006-fs-tutorial-smf.pdf   
代码原地址： http://svn.samba.org/samba/ftp/cifs-cvs/samplefs.tar.gz    

# 参考

[vvsfs](https://github.com/anacrolix/archive/blob/9dadfe81fbac8300c7544ca185234dffad618eb3/university/vvsfs/vvsfs.c)

[myfs](https://github.com/linux-kernel-labs/linux/blob/2c934b4a0456b99c8938cc3c75579b38454b3004/tools/labs/templates/filesystems/myfs/myfs.c)   

[dummyfs](https://github.com/tim-day-387/dummyfs/blob/bd06b7ccb6a01d10580e0db2f3177145aaf3db7e/tests/sanity.sh)   

[aufs-如何自己编写一个文件系统](https://ggaaooppeenngg.github.io/zh-CN/2016/01/04/aufs-%E5%A6%82%E4%BD%95%E8%87%AA%E5%B7%B1%E7%BC%96%E5%86%99%E4%B8%80%E4%B8%AA%E6%96%87%E4%BB%B6%E7%B3%BB%E7%BB%9F/)