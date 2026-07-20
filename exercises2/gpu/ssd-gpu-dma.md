

# gpu内存


> ## NVM_MAP_DEVICE_MEMORY
 create_mapping(&md, _MAP_TYPE_CUDA, fd, devptr, size) --> map_memory-->
 ioctl(md->ioctl_fd,  NVM_MAP_DEVICE_MEMORY, &request) -->
map_device_memory  -->   map_gpu_memory


```
  err = nvidia_p2p_get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages, 
            (void (*)(void*)) force_release_gpu_memory, map);


  err = nvidia_p2p_dma_map_pages(map->pdev, gd->pages, &gd->mappings);
```



```
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice)
{
    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size);

    int status = nvm_dma_map_device(&dma, ctrl, (void *)NVM_PAGE_ALIGN((uintptr_t)devicePtr, 1UL << 16), NVM_ADDR_MASK(size, 1UL << 16));
 

    return DmaPtr(dma, [bufferPtr](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(bufferPtr);
    });
}
```

getDeviceMemory采用cudaMalloc，没有采用cudaMallocManaged   

```
static void getDeviceMemory(int device, void*& bufferPtr, void*& devicePtr, size_t size)
{
    bufferPtr = nullptr;
    devicePtr = nullptr;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }

    devicePtr = attrs.devicePointer;
}

```

# nvfs

```
   $ echo -n "Hello, GDS World!" > test.txt
   
   $ ./strrev_gds.co test.txt 
   sys_len : 17
   !dlroW SDG ,olleH
   See also test.txt
   
   $ cat test.txt 
   !dlroW SDG ,olleH
   
   $ ./strrev_gds.co test.txt 
   sys_len : 17
   Hello, GDS World!
   See also test.txt
   
   $ cat test.txt 
   Hello, GDS World!
```

```
int main(int argc, char *argv[])
{
	int fd;
	int ret;
	int *sys_len;
	int *gpu_len;
	char *system_buf;
	char *gpumem_buf;
	system_buf = (char*)malloc(KB(4));
	sys_len = (int*)malloc(KB(1));
	cudaMalloc(&gpumem_buf, KB(4));
	cudaMalloc(&gpu_len, KB(1));
        off_t file_offset = 0;
        off_t mem_offset = 0;
	CUfileDescr_t cf_desc; 
	CUfileHandle_t cf_handle;

	cuFileDriverOpen();
	fd = open(argv[1], O_RDWR | O_DIRECT);

	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	cuFileHandleRegister(&cf_handle, &cf_desc);
	cuFileBufRegister((char*)gpumem_buf, KB(4), 0);

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, KB(4), file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d", ret); 
	}

	/*
	hello<<<1,1>>>(gpumem_buf);
	*/
	strrev<<<1,1>>>(gpumem_buf, gpu_len);

	cudaMemcpy(sys_len, gpu_len, KB(1), cudaMemcpyDeviceToHost);
	printf("sys_len : %d\n", *sys_len); 
	ret = cuFileWrite(cf_handle, (char*)gpumem_buf, *sys_len, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileWrite failed : %d", ret); 
	}

	cudaMemcpy(system_buf, gpumem_buf, KB(4), cudaMemcpyDeviceToHost);
	printf("%s\n", system_buf);
	printf("See also %s\n", argv[1]);

	cuFileBufDeregister((char*)gpumem_buf);

	cudaFree(gpumem_buf);
	cudaFree(gpu_len);
	free(system_buf);
	free(sys_len);

	close(fd);
	cuFileDriverClose();
}
```

## NVFS_IOCTL_READ


```
        case NVFS_IOCTL_READ:
        case NVFS_IOCTL_WRITE:
```

```
long nvfs_io_start_op(nvfs_io_t* nvfsio)
                if (f->f_op->read_iter && f->f_op->write_iter) {
                        nvfs_get_ops();
                        ret = nvfs_direct_io(op, f,
                                        nvfsio->cpuvaddr,
                                        bytes_issued,
                                        fd_offset,
                                        nvfsio);
                }
```
call_write_iter 和 call_read_iter 执行读写   
```
//TODO: If the config is not present fallback to vfs_read/vfs_write
#ifdef HAVE_CALL_READ_WRITE_ITER
        if(op == WRITE) {
                set_write_flag(&nvfsio->common);
                file_start_write(filp);

                ret = nvfs_io_ret(&nvfsio->common,
                                call_write_iter(filp, &nvfsio->common, &iter));
                if (S_ISREG(file_inode(filp)->i_mode))
                        __sb_writers_release(file_inode(filp)->i_sb,
                                SB_FREEZE_WRITE);
        } else {
                ret = nvfs_io_ret(&nvfsio->common,
                                call_read_iter(filp, &nvfsio->common, &iter));
        }
#endif
```

## nvfsio->cpuvaddr


```
                        ret = nvfs_direct_io(op, f,
                                        nvfsio->cpuvaddr,
                                        bytes_issued,
                                        fd_offset,
                                        nvfsio)
```

##  struct vm_operations_struct
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

## "nvidia-fs"字符设备  struct file_operations 

不同于/dev/nvidia-uvm    
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

## nvfs_dma_map_sg_attrs 

nvidia-fs.ko对外声明了nvfs_dma_map_sg_attrs   


### nvme_rdma_nvfs_map_data 

drivers/nvme/host/rdma.c   
```
static int nvme_rdma_dma_map_req(struct ib_device *ibdev, struct request *rq,
		int *count, int *pi_count)
{
	struct nvme_rdma_request *req = blk_mq_rq_to_pdu(rq);
	int ret;

	req->data_sgl.sg_table.sgl = (struct scatterlist *)(req + 1);
	ret = sg_alloc_table_chained(&req->data_sgl.sg_table,
			blk_rq_nr_phys_segments(rq), req->data_sgl.sg_table.sgl,
			NVME_INLINE_SG_CNT);
	if (ret)
		return -ENOMEM;

#ifdef CONFIG_NVFS
        {
        bool is_nvfs_io = false;
        ret = nvme_rdma_nvfs_map_data(ibdev, rq, &is_nvfs_io, count);
        if (is_nvfs_io) {
	        if (ret)
	               goto out_free_table;
                return 0;
	}
        }
#endif
}
```


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

***nvfs_nvme_dma_rw_ops***

```
	{
		1,
		0,
		NVFS_PROC_MOD_NVME_RDMA_KEY,
		0,
		"nvme_rdma_v1_register_nvfs_dma_ops",
		0,
		"nvme_rdma_v1_unregister_nvfs_dma_ops",
		0,
		&nvfs_nvme_dma_rw_ops
	},
```


```
#ifdef NVFS_ENABLE_KERN_RDMA_SUPPORT
#define SET_DEFAULT_OPS                                         \
	.ft_bmap                        = NVIDIA_FS_SET_FT_ALL, \
	.nvfs_blk_rq_map_sg             = nvfs_blk_rq_map_sg,   \
	.nvfs_dma_map_sg_attrs          = nvfs_dma_map_sg_attrs,        \
	.nvfs_dma_unmap_sg              = nvfs_dma_unmap_sg,    \
	.nvfs_is_gpu_page               = nvfs_is_gpu_page,     \
	.nvfs_gpu_index                 = nvfs_gpu_index,               \
	.nvfs_device_priority           = nvfs_device_priority, \
	.nvfs_get_gpu_sglist_rdma_info  = nvfs_get_gpu_sglist_rdma_info,
#else
#define SET_DEFAULT_OPS                                         \
	.ft_bmap                        = NVIDIA_FS_SET_FT_ALL, \
	.nvfs_blk_rq_map_sg             = nvfs_blk_rq_map_sg,   \
	.nvfs_dma_map_sg_attrs          = nvfs_dma_map_sg_attrs,        \
	.nvfs_dma_unmap_sg              = nvfs_dma_unmap_sg,    \
	.nvfs_is_gpu_page               = nvfs_is_gpu_page,     \
	.nvfs_gpu_index                 = nvfs_gpu_index,               \
	.nvfs_device_priority           = nvfs_device_priority,
#endif


struct nvfs_dma_rw_ops nvfs_dev_dma_rw_ops = {
	SET_DEFAULT_OPS
};
```