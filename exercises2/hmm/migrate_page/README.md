# 匿名页MAP_ANONYMOUS
```
/* Reserve a range of addresses. */
        buffer->ptr = mmap(NULL, size,
                           PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS,
                           buffer->fd, 0);
```



#  mknod
```
root@ubuntux86:# cat /proc/devices | grep HMM_DMIRROR
510 HMM_DMIRROR
root@ubuntux86:# 

root@ubuntux86:# mknod /dev/hmm_dmirror0  c  510 0
root@ubuntux86:# mknod /dev/hmm_dmirror1  c  510 1
```

# insmod  test_hmm.ko 

```
root@ubuntux86:# insmod  test_hmm.ko 
root@ubuntux86:# ./user 
run over 
```

```
[ 5451.042985] memmap_init_zone_device initialised 65536 pages in 0ms
[ 5451.042992] added new 256 MB chunk (total 1 chunks, 256 MB) PFNs [0x3ff0000 0x4000000)
[ 5451.044528] memmap_init_zone_device initialised 65536 pages in 0ms
[ 5451.044530] added new 256 MB chunk (total 1 chunks, 256 MB) PFNs [0x3fe0000 0x3ff0000)
[ 5451.045160] HMM test module loaded. This is only for testing HMM.
[ 5548.870953] cmp page begin 
[ 5548.870960] g_addr 140271137603584 , page start 18446613406139936768, page end 18446613406139940864
[ 5548.870971] src and dts page are equal 
[ 5548.870973] buf is hello world 
[ 5548.871016] cmp page begin 
[ 5548.871017] g_addr 140271137607680 , page start 18446613405101101056, page end 18446613405101105152
[ 5548.871027] src and dts page are equal 
[ 5548.871028] buf is good bye 
```

##  HMM_DMIRROR_MIGRATE
+ 1） dmirror_migrate_alloc_and_copy --> copy_highpage  
```
        for (addr = start; addr < end; addr = next) {
                vma = find_vma(mm, addr);
                if (!vma || addr < vma->vm_start ||
                    !(vma->vm_flags & VM_READ)) {
                        ret = -EINVAL;
                        goto out;
                }
                next = min(end, addr + (ARRAY_SIZE(src_pfns) << PAGE_SHIFT));
                if (next > vma->vm_end)
                        next = vma->vm_end;

                args.vma = vma;
                args.src = src_pfns;
                args.dst = dst_pfns;
                args.start = addr;
                args.end = next;
                args.pgmap_owner = dmirror->mdevice;
                args.flags = MIGRATE_VMA_SELECT_SYSTEM;
                ret = migrate_vma_setup(&args);
                if (ret)
                        goto out;

                dmirror_migrate_alloc_and_copy(&args, dmirror);
                migrate_vma_pages(&args);
                dmirror_migrate_finalize_and_map(&args, dmirror);
                migrate_vma_finalize(&args);
        }
```

## dmirror_devmem_free调用

```
[ 7463.088859]  <TASK>
[ 7463.088861]  dump_stack+0x7d/0x9c
[ 7463.088870]  dmirror_devmem_free+0x1a/0x71 [test_hmm]
[ 7463.088878]  free_devmap_managed_page+0x59/0x60
[ 7463.088885]  put_devmap_managed_page+0x53/0x60
[ 7463.088893]  put_page+0x42/0x50
[ 7463.088902]  zap_pte_range.isra.0+0x4bf/0x7c0
[ 7463.088913]  unmap_page_range+0x40b/0x640
[ 7463.088924]  unmap_single_vma+0x7f/0xf0
[ 7463.088934]  unmap_vmas+0x79/0xf0
[ 7463.088944]  unmap_region+0xbf/0x120
[ 7463.088952]  __do_munmap+0x26f/0x500
[ 7463.088960]  __vm_munmap+0x7f/0x130
[ 7463.088967]  __x64_sys_munmap+0x2d/0x40
[ 7463.088974]  do_syscall_64+0x61/0xb0
[ 7463.088978]  ? irqentry_exit+0x19/0x30
[ 7463.088985]  ? exc_page_fault+0x8f/0x170
[ 7463.088992]  ? asm_exc_page_fault+0x8/0x30
[ 7463.089001]  entry_SYSCALL_64_after_hwframe+0x44/0xae
```



# 缺页异常之swap缺页异常
```Text
1.swap分区的来由
当系统内存不足时，首先回收page cache页面，仍然不足时，继续回收匿名页面，但是匿名页面没有对应文件，因此建立一个swap文件，来临时存储匿名页面，这时匿名页面可以被回收掉，当再次读取匿名页内容时，触发缺页中断，从swap文件读取恢复。

2.swap缺页异常触发条件
pte表项为不为空, 且pte页表项的PRESENT没有置位

3.应用场景
系统内存不足， 匿名页/ipc共享内存页/tmpfs页被换出，再次访问时发生swap缺页异常
```
handle_mm_fault

调用vmf->page->pgmap->ops->migrate_to_ram   
```
else if (is_device_private_entry(entry)) {
            vmf->page = device_private_entry_to_page(entry);
            ret = vmf->page->pgmap->ops->migrate_to_ram(vmf);
        } 
```

```Text
INIT_WORK(&fault->work, do_fault)
do_fault --> handle_mm_fault
```

# ZONE_DEVICE 

```Text
HMM新定义一个名为ZONE_DEVICE的zone类型，外设内存被标记为ZONE_DEVICE，系统内存可以迁移到这个zone中。
从CPU角度看，就想把系统内存swapping到ZONE_DEVICE中，当CPU需要访问这些内存时会触发一个缺页中断，然后再把这些内存从外设中迁移回到系统内存。
```
memremap_pages --> pagemap_range 添加到ZONE_DEVICE   

```
zone = &NODE_DATA(nid)->node_zones[ZONE_DEVICE];
                move_pfn_range_to_zone(zone, PHYS_PFN(range->start),
                                PHYS_PFN(range_len(range)), params->altmap,
                                MIGRATE_MOVABLE);
```

# nouveau_dmem_migrate_to_ram

+ 1)   nouveau_dmem_copy_one --> nvc0b5_migrate_copy(操作硬件)   

```
static vm_fault_t nouveau_dmem_migrate_to_ram(struct vm_fault *vmf)
{
	struct nouveau_drm *drm = page_to_drm(vmf->page);
	struct nouveau_dmem *dmem = drm->dmem;
	struct nouveau_fence *fence;
	struct nouveau_svmm *svmm;
	struct page *spage, *dpage;
	unsigned long src = 0, dst = 0;
	dma_addr_t dma_addr = 0;
	vm_fault_t ret = 0;
	struct migrate_vma args = {
		.vma		= vmf->vma,
		.start		= vmf->address,
		.end		= vmf->address + PAGE_SIZE,
		.src		= &src,
		.dst		= &dst,
		.pgmap_owner	= drm->dev,
		.fault_page	= vmf->page,
		.flags		= MIGRATE_VMA_SELECT_DEVICE_PRIVATE,
	};

	/*
	 * FIXME what we really want is to find some heuristic to migrate more
	 * than just one page on CPU fault. When such fault happens it is very
	 * likely that more surrounding page will CPU fault too.
	 */
	if (migrate_vma_setup(&args) < 0)
		return VM_FAULT_SIGBUS;
	if (!args.cpages)
		return 0;

	spage = migrate_pfn_to_page(src);
	if (!spage || !(src & MIGRATE_PFN_MIGRATE))
		goto done;

	dpage = alloc_page_vma(GFP_HIGHUSER, vmf->vma, vmf->address);
	if (!dpage)
		goto done;

	dst = migrate_pfn(page_to_pfn(dpage));

	svmm = spage->zone_device_data;
	mutex_lock(&svmm->mutex);
	nouveau_svmm_invalidate(svmm, args.start, args.end);
	ret = nouveau_dmem_copy_one(drm, spage, dpage, &dma_addr);
	mutex_unlock(&svmm->mutex);
	if (ret) {
		ret = VM_FAULT_SIGBUS;
		goto done;
	}

	if (!nouveau_fence_new(&fence))
		nouveau_fence_emit(fence, dmem->migrate.chan);
	migrate_vma_pages(&args);
	nouveau_dmem_fence_done(&fence);
	dma_unmap_page(drm->dev->dev, dma_addr, PAGE_SIZE, DMA_BIDIRECTIONAL);
done:
	migrate_vma_finalize(&args);
	return ret;
}

static const struct dev_pagemap_ops nouveau_dmem_pagemap_ops = {
	.page_free		= nouveau_dmem_page_free,
	.migrate_to_ram		= nouveau_dmem_migrate_to_ram,
};

```

#  migrate_vma_pages

# migrate_pfn
```

static inline unsigned long migrate_pfn(unsigned long pfn)
{
        return (pfn << MIGRATE_PFN_SHIFT) | MIGRATE_PFN_VALID;
}
```

```
chunk = kzalloc(sizeof(*chunk), GFP_KERNEL);
chunk->drm = drm;
chunk->pagemap.type = MEMORY_DEVICE_PRIVATE;
chunk->pagemap.range.start = res->start;
chunk->pagemap.range.end = res->end;
chunk->pagemap.nr_range = 1;
chunk->pagemap.ops = &nouveau_dmem_pagemap_ops;
chunk->pagemap.owner = drm->dev;
 ptr = memremap_pages(&chunk->pagemap, numa_node_id());
```