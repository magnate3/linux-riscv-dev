
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
Use of SVA requires IOMMU support in the platform. IOMMU is also required to support the PCIe features ATS and PRI. ATS allows devices to cache translations for virtual addresses. The IOMMU driver uses the mmu_notifier() support to keep the device TLB cache and the CPU cache in sync. When an ATS lookup fails for a virtual address, the device should use the **PRI** in order to request the virtual address to be paged into the CPU page tables. The device must use ATS again in order the fetch the translation before use.

设想SVA的场景中，先malloc得到va, 然把这个va传给设备，配置设备DMA去访问该地址空间，这时内核并没有为va分配实际的
物理内存，所以设备一侧的访问流程必然需要进行类似的缺页请求。支持设备侧缺页
请求的硬件设备就是SMMU，其中对于PCI设备，还需要ATS、PRI硬件特性支持。
平台设备需要SMMU stall mode支持(使用event queue)。PCI设备和平台设备都需要
PASID特性的支持。   

SMMU内部使用command queue，event queue，pri queue做基本的事件管理。当有相应
硬件事件发生时，硬件把相应的描述符写入event queue或者pri queue, 然后上报中断。
软件使用command queue下发相应的命令操作硬件   

> ## pagefault 处理

```
[root@centos7 linux-6.3]# cat /proc/interrupts |grep -i  arm-smmu-v3-priq |awk '{print $1,$(NF-2),$(NF-1),$NF}'
21: 100354 Edge arm-smmu-v3-priq
24: 102402 Edge arm-smmu-v3-priq
27: 104450 Edge arm-smmu-v3-priq
30: 106498 Edge arm-smmu-v3-priq
33: 108546 Edge arm-smmu-v3-priq
36: 110594 Edge arm-smmu-v3-priq
39: 112642 Edge arm-smmu-v3-priq
42: 114690 Edge arm-smmu-v3-priq
[root@centos7 linux-6.3]# 
```

```Text
devm_request_threaded_irq(..., arm_smmu_priq_thread, ...)
arm_smmu_priq_thread
  +-> arm_smmu_handle_ppr
    +-> iommu_report_device_fault
      +-> iommu_fault_param->handler
        +-> iommu_queue_iopf /* 初始化参见上面第2部分 */
          +-> iopf_group = kzalloc
          +-> list_add(faults list in group, fault)
          +-> INIT_WORK(&group->work, iopf_handle_group)
          +-> queue_work(iopf_param->queue->wq, &group->work)
          这段代码创建缺页的group，并把当前的缺页请求挂入group里的链表，然后
          创建一个任务，并调度这个任务运行

          在工作队列线程中:
          +-> iopf_handle_group
            +-> iopf_handle_single
              +-> handle_mm_fault
                  这里会最终申请内存并建立页表

    +-> arm_smmu_page_response
        软件执行完缺页流程后，软件控制SMMU向设备回响应。
```

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

[再议 IOMMU](https://zhuanlan.zhihu.com/p/610416847?utm_id=0)   
