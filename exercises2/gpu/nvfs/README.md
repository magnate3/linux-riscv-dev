# write read

[gpu-direct](https://joyxu.github.io/2022/06/06/gpu-direct/)    
其中cudaMalloc/cuFileBufRegister会从GPU内存分配，并调用nvidia-fs.ko做映射，得到一个va和gpu pa/dma、cpu pa的映射，
后面再调用cuFileRead/cuFileWrite的时候把这个va传递给虚拟文件系统VFS，并通过kernel的call_write_iter/call_read_iter
函数进行文件读写，之后底层sas控制器或者nvme控制器的驱动通过dma_map相关函数把这个va又转换成具体的pa，把内容读或者
写到这块地址中，具体流程如下：
```
nvfs_open
 nvfs_blk_register_dma_ops
  register nvfs_dma_rw_ops 
```
NVFS_IOCTL_READ、NVFS_IOCTL_MAP 、NVFS_IOCTL_WRITE        
```
cuFileRead/cuFileWrite
 nvfs_ioctl
  nvfs_start_io_op
   nvfs_direct_io
    call_write_iter/call_read_iter
     blk_mq_ops.queue_rq
      nvme_queue_rq
       nvme_map_data
        dma_map_bvec
     call nvfs register dma callback
```
> ## cuFileBufRegister(NVFS_IOCTL_MAP) and nvidia_p2p_get_pages   
The cuFileBufRegister function makes the pages that underlie a range of GPU virtual
memory accessible to a third-party device. This process is completed by pinning the GPU
device memory in the BAR space, which is an expensive operation and can take up to a
few milliseconds.    
cuFileBufRegister invokes nvidia_p2p_get_pages NVIDIA driver function to pin
GPU device memory in the BAR space. This information is obtained by running $ perf
top -g and getting the call graph of cuFileBufRegister    


## nvfs register dma callback

 
```
static int nvme_rdma_nvfs_map_data(struct ib_device *ibdev, struct request *rq, bool *is_nvfs_io, int* count)
{
		// associates bio pages to scatterlist
		*count = nvfs_ops->nvfs_blk_rq_map_sg(rq->q, rq , req->data_sgl.sg_table.sgl);
	 

		*count = nvfs_ops->nvfs_dma_map_sg_attrs(ibdev->dma_device,
				req->data_sgl.sg_table.sgl,
				req->data_sgl.nents,
				dma_dir,
				DMA_ATTR_NO_WARN);
 

```
 
#  struct file_operations nvfs_dev_fops


```
struct file_operations nvfs_dev_fops = {
	.compat_ioctl = nvfs_ioctl,
	.unlocked_ioctl = nvfs_ioctl,
	.open = nvfs_open,
	.release = nvfs_close,
        .mmap = nvfs_mgroup_mmap,
        .owner = THIS_MODULE,
};
```
#  const struct vm_operations_struct nvfs_mmap_ops

```
static const struct vm_operations_struct nvfs_mmap_ops = {
	.open = nvfs_vma_open,
#ifdef HAVE_VM_OPS_SPLIT
	.split = nvfs_vma_split,
#else
	.may_split = nvfs_vma_split,
#endif
	.mremap = nvfs_vma_mremap,
	.close = nvfs_vma_close,
        .fault = nvfs_vma_fault,
        .pfn_mkwrite = nvfs_pfn_mkwrite,
        .page_mkwrite = nvfs_page_mkwrite,
};
```
+ nvfs_mgroup->nvfs_ppages[j] = alloc_page(GFP_USER|__GFP_ZERO);没有采用nvidia_p2p_get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages, 
            (void (*)(void*)) force_release_gpu_memory, map)        
```
static int nvfs_mgroup_mmap_internal(struct file *filp, struct vm_area_struct *vma)
{
for (i = 0; i < nvfs_blocks_count; i++) {
		j = i / nvfs_block_count_per_page;
		if (nvfs_mgroup->nvfs_ppages[j] == NULL) {
	                nvfs_mgroup->nvfs_ppages[j] = alloc_page(GFP_USER|__GFP_ZERO);
	                if (nvfs_mgroup->nvfs_ppages[j]) {
	                        nvfs_mgroup->nvfs_ppages[j]->index = (base_index * NVFS_MAX_SHADOW_PAGES) + j;
#ifdef CONFIG_FAULT_INJECTION
				if (nvfs_fault_trigger(&nvfs_vm_insert_page_error)) {
					ret = -EFAULT;
				}
				else
#endif
				{	
					// This will take a page reference which is released in mgroup_put
                        		ret = vm_insert_page(vma, vma->vm_start + j * PAGE_SIZE,
						nvfs_mgroup->nvfs_ppages[j]);
				}

	                        nvfs_dbg("vm_insert_page : %d pages: %lx mapping: %p, "
					  "index: %lx (%lx - %lx) ret: %d  \n",
                	                        j, (unsigned long)nvfs_mgroup->nvfs_ppages[j],
						nvfs_mgroup->nvfs_ppages[j]->mapping,
						nvfs_mgroup->nvfs_ppages[j]->index,
        	                                vma->vm_start + (j * PAGE_SIZE) ,
						vma->vm_start + (j + 1) * PAGE_SIZE,
						ret);
        	                if (ret) {
                	                nvfs_mgroup->nvfs_blocks_count = (j+1) * nvfs_block_count_per_page;
                        	        nvfs_mgroup_put(nvfs_mgroup);
					ret = -ENOMEM;
        				goto error;
                        	}
                	} else {
                        	nvfs_mgroup->nvfs_blocks_count = j * nvfs_block_count_per_page;
	                        nvfs_mgroup_put(nvfs_mgroup);
				ret = -ENOMEM;
        			goto error;
                	}
		}
                //fill the nvfs metadata header
                nvfs_mgroup->nvfs_metadata[i].nvfs_start_magic = NVFS_START_MAGIC;
                nvfs_mgroup->nvfs_metadata[i].nvfs_state = NVFS_IO_ALLOC;
                nvfs_mgroup->nvfs_metadata[i].page = nvfs_mgroup->nvfs_ppages[j];
        }
}
```

#  nvidia_p2p_get_pages
NVFS_IOCTL_MAP --> nvfs_map -->
nvfs_map_gpu_info  -->  nvfs_pin_gpu_pages

```
static int nvfs_pin_gpu_pages(nvfs_ioctl_map_t *input_param,
		struct nvfs_gpu_args *gpu_info)
{
	ret = nvfs_nvidia_p2p_get_pages(0, 0, gpu_virt_start, rounded_size,
			       &gpu_info->page_table,
                               nvfs_get_pages_free_callback, nvfs_mgroup);
}
```

# references

[gpu-direct](https://joyxu.github.io/2022/06/06/gpu-direct/)       