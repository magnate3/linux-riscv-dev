

#  new_sync_read
```
static const struct file_operations myfs_file_operations = {
        /* TODO 6/4: Fill file operations structure. */
        .read           = new_sync_read,
        .read_iter      = generic_file_read_iter,
        .write_iter     = generic_file_write_iter,
        .mmap           = test_generic_file_mmap,
        //.mmap           = generic_file_mmap,
        .llseek         = generic_file_llseek,
        //.fsync          = generic_file_fsync,
};

static ssize_t new_sync_read(struct file *filp, char __user *buf, size_t len, loff_t *ppos)
{
        pr_info("%s \n",__func__);
        struct iovec iov = { .iov_base = buf, .iov_len = len };
        struct kiocb kiocb;
        struct iov_iter iter;
        ssize_t ret;

        init_sync_kiocb(&kiocb, filp);
        kiocb.ki_pos = *ppos;
        iov_iter_init(&iter, READ, &iov, 1, len);

        ret = call_read_iter(filp, &kiocb, &iter);
        BUG_ON(ret == -EIOCBQUEUED);
        *ppos = kiocb.ki_pos;
        return ret;
}
```

```
static inline ssize_t call_read_iter(struct file *file, struct kiocb *kio,
                                     struct iov_iter *iter)
{
        return file->f_op->read_iter(kio, iter);
}
```


call_write_iter 和 call_read_iter 执行读写      
not 测试  
```
ssize_t new_sync_read(struct file *filp, char __user *buf, size_t len, loff_t *ppos)
{
       struct iovec iov = { .iov_base = buf, .iov_len = len };
       struct kiocb kiocb;
       struct iov_iter iter;
       ssize_t ret;

       init_sync_kiocb(&kiocb, filp);
       kiocb.ki_pos = *ppos;
       kiocb.ki_nbytes = len;
       iov_iter_init(&iter, READ, &iov, 1, len);

       ret = filp->f_op->read_iter(&kiocb, &iter); // <- delegate back to file_operations.read_iter
       if (-EIOCBQUEUED == ret)
               ret = wait_on_sync_kiocb(&kiocb);
       *ppos = kiocb.ki_pos;
       return ret;
}
```



# run 
```
[root@centos7 myfs2]# insmod  myfs.ko 
[root@centos7 myfs2]# mount -t myfs none /mnt/myfs
[root@centos7 myfs2]# touch /mnt/myfs/myfile
[root@centos7 myfs2]# echo 'hellowrld' >  /mnt/myfs/myfile
[root@centos7 myfs2]# cat /mnt/myfs/myfile
hellowrld
[root@centos7 myfs2]# dmesg | tail -n 5
[53403.096612] myfs: no symbol version for module_layout
[53403.101645] myfs: loading out-of-tree module taints kernel.
[53403.107254] myfs: module verification failed: signature and/or required key missing - tainting kernel
[53408.916807] root inode has 2 link(s)
[53423.974074] new_sync_read 
[53423.976786] new_sync_read 
[root@centos7 myfs2]# 
```