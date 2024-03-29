

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