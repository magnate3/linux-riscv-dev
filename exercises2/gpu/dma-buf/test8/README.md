

#  Module exporter_test is in use
1） 不加载 importer-test.ko ，不运行dmabuf-test/dmabuf_sync   
2）  insmod  exporter-test.ko 然后 rmmod  exporter-test.ko 

```
[root@centos7 08]# insmod  exporter-test.ko 
[root@centos7 08]# rmmod  exporter-test.ko 
rmmod: ERROR: Module exporter_test is in use
[root@centos7 08]# 
```

这是因为*dma_buf_export 调用了try_module_get，需要调用dma_buf_release -->  module_put
{

```
static void dma_buf_release(struct dentry *dentry)
{
        struct dma_buf *dmabuf;

        dmabuf = dentry->d_fsdata;
        if (unlikely(!dmabuf))
                return;

        BUG_ON(dmabuf->vmapping_counter);

        /*
         * If you hit this BUG() it could mean:
         * * There's a file reference imbalance in dma_buf_poll / dma_buf_poll_cb or somewhere else
         * * dmabuf->cb_in/out.active are non-0 despite no pending fence callback
         */
        BUG_ON(dmabuf->cb_in.active || dmabuf->cb_out.active);

        dma_buf_stats_teardown(dmabuf);
        dmabuf->ops->release(dmabuf);

        if (dmabuf->resv == (struct dma_resv *)&dmabuf[1])
                dma_resv_fini(dmabuf->resv);

        WARN_ON(!list_empty(&dmabuf->attachments));
        module_put(dmabuf->owner);
        kfree(dmabuf->name);
        kfree(dmabuf);
}
```

#   struct dma_buf_ops->realease

struct dma_buf_ops->realease called by  dma_buf_put    

dma_buf_put -->    fput(dmabuf->file) --> ……  -->dma_buf_release --> exporter_release  
```
[ 1834.078872] [<ffff000000c204d0>] exporter_release+0x20/0x70 [exporter_test]
[ 1834.085803] [<ffff0000085acc8c>] dma_buf_release+0x64/0x1a0
[ 1834.091352] [<ffff0000082b3a68>] __fput+0xa8/0x1cc  （   file->f_op->release(inode, file)）
[ 1834.096121] [<ffff0000082b3c04>] ____fput+0x20/0x2c
[ 1834.100981] [<ffff0000080f6364>] task_work_run+0xcc/0xf8
[ 1834.106273] [<ffff0000080d9c04>] do_exit+0x2ec/0xa94
[ 1834.111214] [<ffff0000080da43c>] do_group_exit+0x40/0xd8
[ 1834.116502] [<ffff0000080da4f4>] __wake_up_parent+0x0/0x40
```

#   map_dma_buf  and  unmap_dma_buf
```
 
   struct sg_table *dma_buf_map_attachment(struct dma_buf_attachment *attach,
                       enum dma_data_direction direction)
   {
       struct sg_table *sg_table = ERR_PTR(-EINVAL);
   
       might_sleep();
   
       if (WARN_ON(!attach || !attach->dmabuf))
           return ERR_PTR(-EINVAL);
   
       sg_table = attach->dmabuf->ops->map_dma_buf(attach, direction);
   
       return sg_table;
   }
   EXPORT_SYMBOL_GPL(dma_buf_map_attachment);
   
   void dma_buf_unmap_attachment(struct dma_buf_attachment *attach,
                   struct sg_table *sg_table,
                   enum dma_data_direction direction)
   {
       if (WARN_ON(!attach || !attach->dmabuf || !sg_table))
           return;
   
       attach->dmabuf->ops->unmap_dma_buf(attach, sg_table,
                           direction);
   }
   EXPORT_SYMBOL_GPL(dma_buf_unmap_attachment);
```

# test

```
[root@centos7 08]# 
[root@centos7 08]# insmod  exporter-test.ko 
[root@centos7 08]# insmod  importer-test.ko 
[root@centos7 08]# ./dmabuf-test/dmabuf_sync 
read from dmabuf mmap: hello world!
[root@centos7 08]# dmesg | tail -n 4
[  490.919471] exporter_test: no symbol version for module_layout
[  490.925281] exporter_test: loading out-of-tree module taints kernel.
[  490.931699] exporter_test: module verification failed: signature and/or required key missing - tainting kernel
[  500.957483] ************ dmabuf release
[root@centos7 08]# 
```