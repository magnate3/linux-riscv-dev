

#  收发缓冲区  ibv_reg_dmabuf_mr 


这部分缓冲区会根据是否支持 gdr 选择分配在内存或者是显存上。如果分配到显存上面，会使用 gdr ，收发直接到显存上。gdr 优先使用 dmabuf 方式，否则使用 peermem 方式。

```
for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        int dmabuf_fd;
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
#endif
      {
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
      }
    }
  }
```
 

虽然 RDMA 这个概念已经很早了，但是 GDR 的提出不过是最近十年左右的事情——早期并没有大量数据直接传输到 GPU 上的需求。因此，GDR 这一方案也并没有一个明确的标准。GDR 与普通 RDMA 的区别在于，计算机访问显存和内存的方式不一样，早期的 libverbs 接口中，并不存在ibv_reg_dmabuf_mr，只存在ibv_reg_mr这一通用的接口。因此cudaMalloc分配出的假显存被注册时，IB core 是不知道这是一片映射得到的内存，在尝试去 pin 这块内存时，内核会报错，具体的原因可以参见[pin_user_pages](https://www.kernel.org/doc/html/v5.7/core-api/pin_user_pages.html)这一函数的设计文档，里面有比较详细的介绍。

nvidia-peermem的这一方案，是 Mallenox 公司推出的。它没有改变 libverbs 的接口，而是修改了自身的驱动，对于不能 pin 住的内存，它会轮询额外加载的peer_memory_client模块，让这些模块尝试翻译并 pin 住这些内存区域。对于 N 卡来说，这一模块就是nvidia-peermem。在忽略 GPU 页表切换时，其实这一模块就是完成了一个地址翻译的工作，数据传输本质上还是依赖于显卡驱动的。

ibv_reg_dmabuf_mr是 OpenFabric 提出的方案，为了反对 Mallenox 公司对 GDR 方案的垄断：nvidia-peermem需要绑定 Mallenox 的网卡。这一方案利用 Kernel 中设计的 dmabuf，将其作为中间值，从而让 RNIC 能够拿到翻译后的物理地址。ibv_reg_dmabuf_mr就是为了这一方案而新引入 libverbs 中的一个接口。这一方案的实现是更加优雅的，并且在注册 mr 时不需要轮询，在注册时的性能“理论上”会好一些。ibv_reg_dmabuf_mr这一接口的具体实现，直至 Linux Kernel 5.12 才被加入内核中，因此对软件栈的要求是比较高的。

如果使用的是 Mallenox 网卡，两种方案并没有本质上的区别，因为数据传输都是由显卡驱动完成的。

# mlx5_ib_reg_user_mr_dmabuf

```
.reg_user_mr_dmabuf = mlx5_ib_reg_user_mr_dmabuf
```



#  dma_buf_map_attachment
dma-buf 提供给 DMA 硬件访问的 API 主要就两个：

dma_buf_attach()
dma_buf_map_attachment()
这两个接口调用有严格的先后顺序，必须先 attach，再 map attachment，因为后者的参数是由前者提供的，所以通常这两个接口形影不离。
 
```
int ib_umem_dmabuf_map_pages(struct ib_umem_dmabuf *umem_dmabuf)
{
	struct sg_table *sgt;
	struct scatterlist *sg;
	unsigned long start, end, cur = 0;
	unsigned int nmap = 0;
	long ret;
	int i;

	dma_resv_assert_held(umem_dmabuf->attach->dmabuf->resv);

	if (umem_dmabuf->sgt)
		goto wait_fence;

	sgt = dma_buf_map_attachment(umem_dmabuf->attach,
				     DMA_BIDIRECTIONAL);
	if (IS_ERR(sgt))
		return PTR_ERR(sgt);

	/* modify the sg list in-place to match umem address and length */

	start = ALIGN_DOWN(umem_dmabuf->umem.address, PAGE_SIZE);
	end = ALIGN(umem_dmabuf->umem.address + umem_dmabuf->umem.length,
		    PAGE_SIZE);
	for_each_sgtable_dma_sg(sgt, sg, i) {
		if (start < cur + sg_dma_len(sg) && cur < end)
			nmap++;
		if (cur <= start && start < cur + sg_dma_len(sg)) {
			unsigned long offset = start - cur;

			umem_dmabuf->first_sg = sg;
			umem_dmabuf->first_sg_offset = offset;
			sg_dma_address(sg) += offset;
			sg_dma_len(sg) -= offset;
			cur += offset;
		}
		if (cur < end && end <= cur + sg_dma_len(sg)) {
			unsigned long trim = cur + sg_dma_len(sg) - end;

			umem_dmabuf->last_sg = sg;
			umem_dmabuf->last_sg_trim = trim;
			sg_dma_len(sg) -= trim;
			break;
		}
		cur += sg_dma_len(sg);
	}

	umem_dmabuf->umem.sgt_append.sgt.sgl = umem_dmabuf->first_sg;
	umem_dmabuf->umem.sgt_append.sgt.nents = nmap;
	umem_dmabuf->sgt = sgt;

wait_fence:
	/*
	 * Although the sg list is valid now, the content of the pages
	 * may be not up-to-date. Wait for the exporter to finish
	 * the migration.
	 */
	ret = dma_resv_wait_timeout(umem_dmabuf->attach->dmabuf->resv,
				     DMA_RESV_USAGE_KERNEL,
				     false, MAX_SCHEDULE_TIMEOUT);
	if (ret < 0)
		return ret;
	if (ret == 0)
		return -ETIMEDOUT;
	return 0;
}
EXPORT_SYMBOL(ib_umem_dmabuf_map_pages);
```

# references

[RDMA（三）- 从DMA-BUF 到GDR](https://zhuanlan.zhihu.com/p/685361884)   
[RDMA（二）- 从rdma 看 CPU 架构和瓶颈](https://zhuanlan.zhihu.com/p/676931271)   