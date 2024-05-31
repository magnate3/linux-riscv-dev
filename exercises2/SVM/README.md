
# Qemu-in-guest-SVM-demo

[Qemu-in-guest-SVM-demo](https://github.com/BullSequana/Qemu-in-guest-SVM-demo)   


# SVM-demo

```
grep SVM config-5.13.0-39-generic 
# CONFIG_DRM_NOUVEAU_SVM is not set
CONFIG_INTEL_IDXD_SVM=y
CONFIG_INTEL_IOMMU_SVM=y
```

# CONFIG_IOMMU_SVA


#  iommu_dma_alloc_iova
```
 backtrace:
    [<000000001b204ddf>] kmem_cache_alloc+0x1b0/0x350
    [<00000000d9ef2e50>] alloc_iova+0x3c/0x168
    [<00000000ea30f99d>] alloc_iova_fast+0x7c/0x2d8
    [<00000000b8bb2f1f>] iommu_dma_alloc_iova.isra.0+0x12c/0x138
    [<000000002f1a43b5>] __iommu_dma_map+0x8c/0xf8
    [<00000000ecde7899>] iommu_dma_map_page+0x98/0xf8
    [<0000000082004e59>] otx2_alloc_rbuf+0xf4/0x158
    [<000000002b107f6b>] otx2_rq_aura_pool_init+0x110/0x270
    [<00000000c3d563c7>] otx2_open+0x15c/0x734
    [<00000000a2f5f3a8>] otx2_dev_open+0x3c/0x68
    [<00000000456a98b5>] otx2_set_ringparam+0x1ac/0x1d4
    [<00000000f2fbb819>] dev_ethtool+0xb84/0x2028
    [<0000000069b67c5a>] dev_ioctl+0x248/0x3a0
    [<00000000af38663a>] sock_ioctl+0x280/0x638
    [<000000002582384c>] do_vfs_ioctl+0x8b0/0xa80
    [<000000004e1a2c02>] ksys_ioctl+0x84/0xb8
```

# references

[Linux x86-64 IOMMU详解（六）——Intel IOMMU参与下的DMA Coherent Mapping流程](https://blog.csdn.net/qq_34719392/article/details/117699839)   

