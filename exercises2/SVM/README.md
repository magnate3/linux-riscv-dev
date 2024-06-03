
# Qemu-in-guest-SVM-demo

[Qemu-in-guest-SVM-demo](https://github.com/BullSequana/Qemu-in-guest-SVM-demo)   




# SVM-demo

```
grep SVM config-5.13.0-39-generic 
# CONFIG_DRM_NOUVEAU_SVM is not set
CONFIG_INTEL_IDXD_SVM=y
CONFIG_INTEL_IOMMU_SVM=y
```

# CONFIG_IOMMU_SVA  IOMMU_DOMAIN_SVA

```
#define IOMMU_DOMAIN_BLOCKED    (0U)
#define IOMMU_DOMAIN_IDENTITY   (__IOMMU_DOMAIN_PT)
#define IOMMU_DOMAIN_UNMANAGED  (__IOMMU_DOMAIN_PAGING)
#define IOMMU_DOMAIN_DMA        (__IOMMU_DOMAIN_PAGING |        \
                                 __IOMMU_DOMAIN_DMA_API)
#define IOMMU_DOMAIN_DMA_FQ     (__IOMMU_DOMAIN_PAGING |        \
                                 __IOMMU_DOMAIN_DMA_API |       \
                                 __IOMMU_DOMAIN_DMA_FQ)
#define IOMMU_DOMAIN_SVA        (__IOMMU_DOMAIN_SVA)
```


```
	case IOMMU_DOMAIN_SVA:
		return intel_svm_domain_alloc();
```

# page fault

> ##  iommu_sva_handle_iopf -->  handle_mm_fault
```
/*
 * I/O page fault handler for SVA
 */
enum iommu_page_response_code
iommu_sva_handle_iopf(struct iommu_fault *fault, void *data)
{
	vm_fault_t ret;
	struct vm_area_struct *vma;
	struct mm_struct *mm = data;
	unsigned int access_flags = 0;
	unsigned int fault_flags = FAULT_FLAG_REMOTE;
	struct iommu_fault_page_request *prm = &fault->prm;
	enum iommu_page_response_code status = IOMMU_PAGE_RESP_INVALID;

	if (!(prm->flags & IOMMU_FAULT_PAGE_REQUEST_PASID_VALID))
		return status;

	if (!mmget_not_zero(mm))
		return status;

	mmap_read_lock(mm);

	vma = vma_lookup(mm, prm->addr);
	if (!vma)
		/* Unmapped area */
		goto out_put_mm;

	if (prm->perm & IOMMU_FAULT_PERM_READ)
		access_flags |= VM_READ;

	if (prm->perm & IOMMU_FAULT_PERM_WRITE) {
		access_flags |= VM_WRITE;
		fault_flags |= FAULT_FLAG_WRITE;
	}

	if (prm->perm & IOMMU_FAULT_PERM_EXEC) {
		access_flags |= VM_EXEC;
		fault_flags |= FAULT_FLAG_INSTRUCTION;
	}

	if (!(prm->perm & IOMMU_FAULT_PERM_PRIV))
		fault_flags |= FAULT_FLAG_USER;

	if (access_flags & ~vma->vm_flags)
		/* Access fault */
		goto out_put_mm;

	ret = handle_mm_fault(vma, prm->addr, fault_flags, NULL);
	status = ret & VM_FAULT_ERROR ? IOMMU_PAGE_RESP_INVALID :
		IOMMU_PAGE_RESP_SUCCESS;

out_put_mm:
	mmap_read_unlock(mm);
	mmput(mm);

	return status;
}
```

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

