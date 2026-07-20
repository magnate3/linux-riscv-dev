

#  dma_buf_release

```
static void exporter_release(struct dma_buf *dmabuf)
{
        pr_info("exporter_release: releasing dma_buf \n");
        dump_stack();
        kfree(dmabuf->priv);
        dmabuf_exported = NULL;
}
```
调用  close(dma_buf_fd)
```
[17945.763989] Call trace:
[17945.766434] [<ffff000008089e14>] dump_backtrace+0x0/0x23c
[17945.771809] [<ffff00000808a074>] show_stack+0x24/0x2c
[17945.776839] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[17945.781869] [<ffff0000010201fc>] exporter_release+0x24/0x50 [exporter_fd_test]
[17945.789060] [<ffff0000085acc8c>] dma_buf_release+0x64/0x1a0
[17945.794610] [<ffff0000082b3a68>] __fput+0xa8/0x1cc
[17945.799379] [<ffff0000082b3c04>] ____fput+0x20/0x2c
[17945.804238] [<ffff0000080f6364>] task_work_run+0xcc/0xf8
[17945.809525] [<ffff0000080894d0>] do_notify_resume+0x104/0x128
```

# test


```
[root@centos7 04]# ./mmap_dmabuf 
ion alloc success: size = 65536, dmabuf_fd = 4
dma buf fd 4 
read from dmabuf mmap: hello world!
[root@centos7 04]# ./mmap_dmabuf 
ion alloc success: size = 65536, dmabuf_fd = 4
dma buf fd 4 
read from dmabuf mmap: hello world!
[root@centos7 04]# 
```