
#  dma_map_single/dma_unmap_single的使用
设备驱动里一般调用dma_map_single()/dma_unmap_single()处理cache。调用dma_map_single函数时需要指定DMA的方向，DMA_TO_DEVICE或者DMA_FROM_DEVICE。Linux会根据direction的值invalidate或者clean cache。   
drivers\net\ethernet\cadence\macb_main.c的函数macb_tx_map()里，调用dma_map_single()刷新cache，macb_tx_interrupt()的macb_tx_unmap()再调用dma_unmap_single()。  
代码简化后如下：   
```C
macb_tx_map( )
{
.......
	mapping = dma_map_single(&bp->pdev->dev,
			 skb->data + offset,
			 size, DMA_TO_DEVICE);
	.......		
}
			 
macb_tx_unmap( )
{
	.......				
	 
	dma_unmap_single(&bp->pdev->dev, tx_skb->mapping,
			 tx_skb->size, DMA_TO_DEVICE);
	.......		
}
	 
					 					
gem_rx( )
{
.......
	dma_unmap_single(&bp->pdev->dev, addr,
		bp->rx_buffer_size, DMA_FROM_DEVICE);
	.......		
}					
					
gem_rx_refill()
{
.......
	/* now fill corresponding descriptor entry */
	paddr = dma_map_single(&bp->pdev->dev, skb->data,
					bp->rx_buffer_size,
					DMA_FROM_DEVICE);
	.......		
}
```

# dma_map_single/dma_unmap_single的定义
dma_map_single()和dma_unmap_single（）都在include\linux\dma-mapping.h里定义。如果没有特殊情况，会调用dma_direct_map_page（）、dma_direct_unmap_page（）。   
 arm64的特殊情况包括iommu和Xen虚拟机。    iommu和Xen虚拟机都需要提供dma_map_ops，于是使用其中的map、unmap函数。iommu的dma_map_ops是drivers\iommu\Dma-iommu.c中定义的iommu_dma_ops。 iommu的dma_map_ops是drivers/xen/swiotlb-xen.c中定义的xen_swiotlb_dma_ops。   
 
 ```C
 #define dma_map_single(d, a, s, r) dma_map_single_attrs(d, a, s, r, 0)
#define dma_unmap_single(d, a, s, r) dma_unmap_single_attrs(d, a, s, r, 0)

static inline dma_addr_t dma_map_single_attrs(struct device *dev, void *ptr,
		size_t size, enum dma_data_direction dir, unsigned long attrs)
{
	debug_dma_map_single(dev, ptr, size);
	return dma_map_page_attrs(dev, virt_to_page(ptr), offset_in_page(ptr),
			size, dir, attrs);
}

static inline void dma_unmap_single_attrs(struct device *dev, dma_addr_t addr,
		size_t size, enum dma_data_direction dir, unsigned long attrs)
{
	return dma_unmap_page_attrs(dev, addr, size, dir, attrs);
}


static inline dma_addr_t dma_map_page_attrs(struct device *dev,
		struct page *page, size_t offset, size_t size,
		enum dma_data_direction dir, unsigned long attrs)
{
	const struct dma_map_ops *ops = get_dma_ops(dev);
	dma_addr_t addr;

	BUG_ON(!valid_dma_direction(dir));
	if (dma_is_direct(ops))
		addr = dma_direct_map_page(dev, page, offset, size, dir, attrs);
	else
		addr = ops->map_page(dev, page, offset, size, dir, attrs);
	debug_dma_map_page(dev, page, offset, size, dir, addr);

	return addr;
}

static inline void dma_unmap_page_attrs(struct device *dev, dma_addr_t addr,
		size_t size, enum dma_data_direction dir, unsigned long attrs)
{
	const struct dma_map_ops *ops = get_dma_ops(dev);

	BUG_ON(!valid_dma_direction(dir));
	if (dma_is_direct(ops))
		dma_direct_unmap_page(dev, addr, size, dir, attrs);
	else if (ops->unmap_page)
		ops->unmap_page(dev, addr, size, dir, attrs);
	debug_dma_unmap_page(dev, addr, size, dir);
}
 ```
#  dma_direct_map_page/dma_direct_unmap_page的定义
dma_direct_map_page()、dma_direct_unmap_page()在kernel\dma\direct.c中定义。    
```C
dma_addr_t dma_direct_map_page(struct device *dev, struct page *page,
		unsigned long offset, size_t size, enum dma_data_direction dir,
		unsigned long attrs)
{
	phys_addr_t phys = page_to_phys(page) + offset;
	dma_addr_t dma_addr = phys_to_dma(dev, phys);

	if (unlikely(!dma_direct_possible(dev, dma_addr, size)) &&
	    !swiotlb_map(dev, &phys, &dma_addr, size, dir, attrs)) {
		report_addr(dev, dma_addr, size);
		return DMA_MAPPING_ERROR;
	}

	if (!dev_is_dma_coherent(dev) && !(attrs & DMA_ATTR_SKIP_CPU_SYNC))
		arch_sync_dma_for_device(dev, phys, size, dir);
	return dma_addr;
}
EXPORT_SYMBOL(dma_direct_map_page);


void dma_direct_unmap_page(struct device *dev, dma_addr_t addr,
		size_t size, enum dma_data_direction dir, unsigned long attrs)
{
	phys_addr_t phys = dma_to_phys(dev, addr);

	if (!(attrs & DMA_ATTR_SKIP_CPU_SYNC))
		dma_direct_sync_single_for_cpu(dev, addr, size, dir);

	if (unlikely(is_swiotlb_buffer(phys)))
		swiotlb_tbl_unmap_single(dev, phys, size, size, dir, attrs);
}
EXPORT_SYMBOL(dma_direct_unmap_page);


void dma_direct_sync_single_for_cpu(struct device *dev,
        dma_addr_t addr, size_t size, enum dma_data_direction dir)
{
    phys_addr_t paddr = dma_to_phys(dev, addr);
 
    if (!dev_is_dma_coherent(dev)) {
        arch_sync_dma_for_cpu(dev, paddr, size, dir);
        arch_sync_dma_for_cpu_all(dev);
    }
 
    if (unlikely(is_swiotlb_buffer(paddr)))
        swiotlb_tbl_sync_single(dev, paddr, size, dir, SYNC_FOR_CPU);
}
```
一路跟踪，dma_map_single()会最终调用到arch_sync_dma_for_device()， dma_unmap_single()会最终调用到arch_sync_dma_for_cpu()， 和arch_sync_dma_for_cpu_all()。 而arch_sync_dma_for_cpu_all()对Arm64是空函数。   

#  arch_sync_dma_for_device/arch_sync_dma_for_cpu的定义

arch_sync_dma_for_device/arch_sync_dma_for_cpu的定义在文件arch\arm64\mm\dma-mapping.c中。   
```C
void arch_sync_dma_for_device(struct device *dev, phys_addr_t paddr,
		size_t size, enum dma_data_direction dir)
{
	__dma_map_area(phys_to_virt(paddr), size, dir);
}


void arch_sync_dma_for_cpu(struct device *dev, phys_addr_t paddr,
		size_t size, enum dma_data_direction dir)
{
	__dma_unmap_area(phys_to_virt(paddr), size, dir);
}
```

#  __dma_map_area/__dma_unmap_area的定义
__dma_map_area/__dma_unmap_area的定义在文件arch\arm64\mm\cache.S中。   

所有汇编实现，也在文件arch\arm64\mm\cache.S中。   

```C
*
 *	__dma_map_area(start, size, dir)
 *	- start	- kernel virtual start address
 *	- size	- size of region
 *	- dir	- DMA direction
 */
ENTRY(__dma_map_area)
	cmp	w2, #DMA_FROM_DEVICE
	b.eq	__dma_inv_area
	b	__dma_clean_area
ENDPIPROC(__dma_map_area)

/*
 *	__dma_unmap_area(start, size, dir)
 *	- start	- kernel virtual start address
 *	- size	- size of region
 *	- dir	- DMA direction
 */
ENTRY(__dma_unmap_area)
	cmp	w2, #DMA_TO_DEVICE
	b.ne	__dma_inv_area
	ret
ENDPIPROC(__dma_unmap_area
```
可以看到，map系列函数调用的__dma_map_area，方向如果是DMA_FROM_DEVICE，执行__dma_inv_area； 否则执行 __dma_clean_area。
 unmap系列函数调用的__dma_unmap_area，方向如果不是DMA_TO_DEVICE，执行__dma_inv_area； 否则执行__dma_clean_area。
    