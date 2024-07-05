


#  nvidia_p2p_dma_map_pages
```
long __afhba_gpumem_lock(
		struct AFHBA_DEV *adev, const char* name,
		unsigned long iova, uint64_t addr, uint64_t size, unsigned prot)
{
	struct nvidia_p2p_dma_mapping *nv_dma_map = 0;
	size_t pin_size = 0ULL;

	dev_info(pdev(adev), "Original %s HostBuffer physical address is %lx.\n", name, iova);
	dev_info(pdev(adev), "Virtual %s GPU address is  %llx.\n", name, addr);

	if (gpu_pin(adev, name, &nv_dma_map, addr, size, &pin_size)){
		dev_err(pdev(adev), "%s(): Error in gpu_pin()", __FUNCTION__);
	   	return -EFAULT;
	}

	//  Enable iommu DMA remapping -> AFHBA404 card can only address 32-bit memory
	if (afhba_iommu_map(adev, iova, nv_dma_map->dma_addresses[0], pin_size, prot)){
		dev_err(pdev(adev), "iommu_map failed -- aborting.\n");
		return -EFAULT;
	}else{
		dev_info(pdev(adev), "iommu_map success %s iova %lx..%llx points to %llx",
				name, iova, iova+size, nv_dma_map->dma_addresses[0]);
	}
	return 0;
}
```

调用afhba_iommu_map（nv_dma_map->dma_addresses）

nvidia_p2p_dma_map_pages(adev->pci_dev, entry->page_table, nv_dma_map)    
```
/*------------------------------------------------------------------------------
-  gpu_pin:
-    Function to pin the specified gpu virtual address and get the physical address
-    Currently unused, this is done manually for channel A in afhba_gpumem_unlock
-    Should be correctly implemented to generalize the gpu memory pinning for
-      multiple channel usage.
-    TODO: get gpumem->table_list allocated correctly as a static struct?
------------------------------------------------------------------------------*/
int gpu_pin(struct AFHBA_DEV *adev, const char* name,
	struct nvidia_p2p_dma_mapping ** nv_dma_map,
	uint64_t addr, uint64_t size, size_t *ppin_size){
	// gpu_pin function is currently unused, this is done manually inside afhba_gpumem_lock
	// should be separated out to generalize the gpu memory pinning
	int error = 0;
	size_t pin_size = 0ULL;
	struct gpumem_t *entry = (struct gpumem_t*)kzalloc(sizeof(struct gpumem_t), GFP_KERNEL);
	if(!entry) {
		dev_err(pdev(adev), "%s(): Error allocate memory to mapping struct\n", __FUNCTION__);
		return -ENOMEM;
	}
	INIT_LIST_HEAD(&entry->list);
	strncpy(entry->name, name, sizeof(entry->name)-1);
	entry->virt_start = (addr & GPU_BOUND_MASK);
	pin_size = addr + size - entry->virt_start;
	if(!pin_size) {
		printk(KERN_ERR"%s(): Error invalid memory size!\n", __FUNCTION__);
		error = -EINVAL;
		goto do_free_mem;
	}else{
		*ppin_size = pin_size;
	}

	dev_info(pdev(adev), "%s %s addr=%llx, size=%llx, virt_start=%llx, pin_size=%lx",
			__FUNCTION__, name, addr, size, entry->virt_start, pin_size);

	error = nvidia_p2p_get_pages(0, 0, entry->virt_start, pin_size, &entry->page_table, free_nvp_callback, entry);
	if(error != 0) {
		dev_err(pdev(adev), "%s(): Error in nvidia_p2p_get_pages()\n", __FUNCTION__);
		error = -EINVAL;
		goto do_unlock_pages;
	}

	dev_info(pdev(adev),"%s %s Pinned GPU memory, physical address is %llx",
			__FUNCTION__, name, entry->page_table->pages[0]->physical_address);

	error = nvidia_p2p_dma_map_pages(adev->pci_dev, entry->page_table, nv_dma_map);

	if(error){
		dev_err(pdev(adev), "%s(): Error %d in nvidia_p2p_dma_map_pages()\n", __FUNCTION__,error);
		error = -EFAULT;
		goto do_unmap_dma;
	}

	dev_info(pdev(adev),"%s %s nvidia_dma_mapping: npages= %d\n",
			__FUNCTION__, name, (*nv_dma_map)->entries);

	list_add_tail(&entry->list, &adev->gpumem.table_list);
	return 0;

do_unmap_dma:
	nvidia_p2p_dma_unmap_pages(adev->pci_dev, entry->page_table, *nv_dma_map);
do_unlock_pages:
	nvidia_p2p_put_pages(0, 0, entry->virt_start, entry->page_table);
do_free_mem:
	kfree(entry);
	return (long) error;
}
```