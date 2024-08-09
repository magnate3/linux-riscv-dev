

#  IO page fault ( IOPF)

为了保持高内存利用率，需要使能I/O缺页错误（I/O Page Fault, IOPF）以按需灵活分配内存。
IOPF实现方案：ATS+PRI标准   


	Shared Virtual Addressing (SVA) allows the processor and device to use the same virtual addresses avoiding the need for software to translate virtual addresses to physical addresses. SVA is what PCIe calls Shared Virtual Memory (SVM).   

	In addition to the convenience of using application virtual addresses by the device, it also doesn’t require pinning pages for DMA. PCIe Address Translation Services (ATS) along with Page Request Interface (PRI) allow devices to function much the same way as the CPU handling application page-faults. For more information please refer to the PCIe specification Chapter 10: ATS Specification.   

	Use of SVA requires IOMMU support in the platform. IOMMU is also required to support the PCIe features ATS and PRI. ATS allows devices to cache translations for virtual addresses. The IOMMU driver uses the mmu_notifier() support to keep the device TLB cache and the CPU cache in sync. When an ATS lookup fails for a virtual address, the device should use the PRI in order to request the virtual address to be paged into the CPU page tables. The device must use ATS again in order the fetch the translation before use.    



```
static int svm_demo_enable_sva(void)
{
	ret = pci_enable_ats(device.dev, PAGE_SHIFT);
	ret = iommu_dev_enable_feature(&device.dev->dev, IOMMU_DEV_FEAT_IOPF);
	ret = iommu_dev_enable_feature(&device.dev->dev, IOMMU_DEV_FEAT_SVA);
	return 0;
}
```


```
int arm_smmu_master_enable_sva(struct arm_smmu_master *master)
{
        int ret;

        mutex_lock(&sva_lock);
        ret = arm_smmu_master_sva_enable_iopf(master);
        if (!ret)
                master->sva_enabled = true;
        mutex_unlock(&sva_lock);

        return ret;
}

```

> ## sva  iopf
domain->type = IOMMU_DOMAIN_SVA;    
```

struct iommu_domain *iommu_sva_domain_alloc(struct device *dev,
                                            struct mm_struct *mm)
{
        const struct iommu_ops *ops = dev_iommu_ops(dev);
        struct iommu_domain *domain;

        domain = ops->domain_alloc(IOMMU_DOMAIN_SVA);
        if (!domain)
                return NULL;

        domain->type = IOMMU_DOMAIN_SVA;
        mmgrab(mm);
        domain->mm = mm;
        domain->iopf_handler = iommu_sva_handle_iopf;
        domain->fault_data = mm;

        return domain;
}
```

```
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

        vma = find_extend_vma(mm, prm->addr);
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
}
```

## pri 中断处理


+ 注册 中断 callback (iommu_queue_iopf ) 
iommu_queue_iopf
```
        +-> iommu_sva_enable
          +-> iommu_register_device_fault_handler(dev, iommu_queue_iopf, dev)
              /* 动态部分将执行iommu_queue_iopf */
              把iommu_queue_iopf赋值给iommu_fault_param里的handler                  
        +-> iopf_queue_add_device(struct iopf_queue, dev)
            把相应的iopf_queue赋值给iopf_device_param里的iopf_queue, 这里有
            pri对应的iopf_queue或者是stall mode对应的iopf_queue。初始化
            iopf_device_param里的wait queue
```

+ 中断触发   
 

当一个PRI或者是一个stall event上报后, 软件会在缺页流程里建立页表，然后控制
SMMU给设备返送reponse信息。我们可以从SMMU PRI queue或者是event queue的中断
处理流程入手跟踪: e.g.PRI中断流程       
iommu_report_device_fault    
```
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


# /dev/devmm_svm    
smmu svm vma  IOPF
```
int devmm_insert_pages_range_to_vma(struct devmm_svm_process *svm_process, unsigned long va,
    u64 page_num, struct page **inpages)
{
    struct vm_area_struct *vma = svm_process->vma;
    phys_addr_t offset;
    int ret;
    u64 i;

    vma->vm_page_prot = devmm_make_pgprot(0);
    for (i = 0; i < page_num; i++) {
        offset = (i << PAGE_SHIFT);
        ret = remap_pfn_range(vma, va + offset, page_to_pfn(inpages[i]), PAGE_SIZE, vma->vm_page_prot);
        if (ret) {
            devmm_drv_err("vm_insert_page() failed,ret=%d. va=0x%lx, i=%llu, page_num=%llu.\n",
                ret, va, i, page_num);
            /* will not return fail ,so free page here */
            devmm_free_pages(1, &inpages[i], svm_process);
        } else {
            devmm_pin_page(inpages[i]);
        }
        devmm_svm_stat_page_inc(PAGE_SIZE);
        devmm_svm_stat_pg_map_inc();
    }

    return 0;
}
```