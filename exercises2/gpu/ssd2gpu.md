
# gpu 地址和dma


```
nvidia_p2p_page_table_t *page_table = mgmem->page_table;
base_addr = page_table->pages[base + i]->physical_address;
iod->sg[i].dma_address = base_addr + offset;
```


```
static int
submit_ssd2gpu_memcpy(strom_dma_state *dstate)
{
	strom_dma_task	   *dtask = dstate->dtask;
	mapped_gpu_memory  *mgmem = dtask->mgmem;
	nvidia_p2p_page_table_t *page_table = mgmem->page_table;
	struct nvme_ns	   *nvme_ns = dstate->nvme_ns;
	struct nvme_dev	   *nvme_dev = nvme_ns->dev;
	struct nvme_iod	   *iod;
	size_t				offset;
	size_t				total_nbytes;
	dma_addr_t			base_addr;
	int					length;
	int					i, base;
	int					retval;

	total_nbytes = (dstate->nr_blocks << dstate->blocksz_shift);
	if (!total_nbytes || total_nbytes > STROM_DMA_SSD2GPU_MAXLEN)
		return -EINVAL;
	if (dstate->dest_offset < mgmem->map_offset ||
		dstate->dest_offset + total_nbytes > (mgmem->map_offset +
											  mgmem->map_length))
		return -ERANGE;

	iod = nvme_alloc_iod(total_nbytes,
						 mgmem,
						 nvme_dev,
						 GFP_KERNEL);
	if (!iod)
		return -ENOMEM;

	base = (dstate->dest_offset >> mgmem->gpu_page_shift);
	offset = (dstate->dest_offset & (mgmem->gpu_page_sz - 1));
	prDebug("base=%d offset=%zu dest_offset=%zu total_nbytes=%zu",
			base, offset, dstate->dest_offset, total_nbytes);

	for (i=0; i < page_table->entries; i++)
	{
		if (!total_nbytes)
			break;

		base_addr = page_table->pages[base + i]->physical_address;
		length = Min(total_nbytes, mgmem->gpu_page_sz - offset);
		iod->sg[i].page_link = 0;
		iod->sg[i].dma_address = base_addr + offset;
		iod->sg[i].length = length;
		iod->sg[i].dma_length = length;
		iod->sg[i].offset = 0;

		offset = 0;
		total_nbytes -= length;
	}

	if (total_nbytes)
	{
		__nvme_free_iod(nvme_dev, iod);
		return -EINVAL;
	}
	sg_mark_end(&iod->sg[i]);
	iod->nents = i;

	retval = nvme_submit_async_read_cmd(dstate, iod);
	if (retval)
		__nvme_free_iod(nvme_dev, iod);

	/* clear the state */
	dstate->nr_blocks = 0;
	dstate->src_block = 0;
	dstate->dest_offset = ~0UL;

	return retval;
}
```