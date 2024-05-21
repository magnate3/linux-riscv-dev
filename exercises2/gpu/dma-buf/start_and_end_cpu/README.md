
#  exporter_begin_cpu_access

sync.flags = DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START;   
```
[22214.876794] [<ffff000008089e14>] dump_backtrace+0x0/0x23c
[22214.882168] [<ffff00000808a074>] show_stack+0x24/0x2c
[22214.887198] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[22214.892229] [<ffff000000f207b0>] exporter_begin_cpu_access+0x20/0x80 [exporter_page]
[22214.899940] [<ffff0000085ad888>] dma_buf_begin_cpu_access+0x38/0x80
[22214.906177] [<ffff0000085ad9d8>] dma_buf_ioctl+0x108/0x138
[22214.911642] [<ffff0000082c6e18>] do_vfs_ioctl+0xcc/0x8fc
[22214.916931] [<ffff0000082c76d8>] SyS_ioctl+0x90/0xa4
```

```

        sync.flags = DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START;
        ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync);
```


# dma_buf_end_cpu_access

sync.flags = DMA_BUF_SYNC_READ | DMA_BUF_SYNC_END;   


```
[22215.030745] Call trace:
[22215.033183] [<ffff000008089e14>] dump_backtrace+0x0/0x23c
[22215.038557] [<ffff00000808a074>] show_stack+0x24/0x2c
[22215.043584] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[22215.048613] [<ffff000000f20730>] exporter_end_cpu_access+0x20/0x80 [exporter_page]
[22215.056147] [<ffff0000085ac8b0>] dma_buf_end_cpu_access+0x38/0x5c
[22215.062212] [<ffff0000085ad9ac>] dma_buf_ioctl+0xdc/0x138
[22215.067585] [<ffff0000082c6e18>] do_vfs_ioctl+0xcc/0x8fc
[22215.072865] [<ffff0000082c76d8>] SyS_ioctl+0x90/0xa4
```


```
sync.flags = DMA_BUF_SYNC_READ | DMA_BUF_SYNC_END;
ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync);
```