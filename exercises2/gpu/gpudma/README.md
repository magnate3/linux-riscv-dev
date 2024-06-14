

# main.cpp


```
   size_t size = 0x100000;
    CUdeviceptr dptr = 0;
    unsigned int flag = 1;
    unsigned char *h_odata = NULL;
    h_odata = (unsigned char *)malloc(size);

    CUresult status = cuMemAlloc(&dptr, size);
	// TODO: add kernel driver interaction...
    lock.addr = dptr;
    lock.size = size;
    res = ioctl(fd, IOCTL_GPUMEM_LOCK, &lock);
```

+ 1 malloc(size)   

+ 2 cuMemAlloc(&dptr, size),没有采用cudaMallocManaged     

+ 3 ioctl(fd, IOCTL_GPUMEM_LOCK, &lock)   

+ 4  ioctl(fd, IOCTL_GPUMEM_STATE, state)   

+ 5 cpu虚拟地址：     
    mmap(0, state->page_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (off_t)state->pages[i])      

# 内核  

##  IOCTL_GPUMEM_LOCK --> ioctl_mem_lock

```

    CUresult status = cuMemAlloc(&dptr, size);
    if(wasError(status)) {
        goto do_free_context;
    }

    fprintf(stderr, "Allocate memory address: 0x%llx\n",  (unsigned long long)dptr);

    status = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, dptr);
    // TODO: add kernel driver interaction...
    lock.addr = dptr;
    lock.size = size;
    res = ioctl(fd, IOCTL_GPUMEM_LOCK, &lock);
```
entry->virt_start = (param.addr & GPU_BOUND_MASK);  GPU虚拟地址     
```
   INIT_LIST_HEAD(&entry->list);
    entry->handle = entry;

    entry->virt_start = (param.addr & GPU_BOUND_MASK);
    pin_size = (param.addr + param.size - entry->virt_start);
    if(!pin_size) {
        printk(KERN_ERR"%s(): Error invalid memory size!\n", __FUNCTION__);
        error = -EINVAL;
        goto do_free_mem;
    }

    error = nvidia_p2p_get_pages(0, 0, entry->virt_start, pin_size, &entry->page_table, free_nvp_callback, entry);
    param.page_count = entry->page_table->entries;
    param.handle = entry;
```
***entry->page_table 是gpu分配的内存***

## IOCTL_GPUMEM_STATE 获取gpu分配的内存的物理地址  

```
int ioctl_mem_state(struct gpumem *drv, unsigned long arg)
{
    
    list_for_each_safe(pos, n, &drv->table_list) {

        entry = list_entry(pos, struct gpumem_t, list);
        if(entry) {
            if(entry->handle == header.handle) {
         
                for(i=0; i<entry->page_table->entries; i++) {
                    struct nvidia_p2p_page *nvp = entry->page_table->pages[i];
                    if(nvp) {
                        param->pages[i] = nvp->physical_address;
                        param->page_count++;
                        printk(KERN_ERR"%s(): %02d - 0x%llx\n", __FUNCTION__, i, param->pages[i]);
                    }
                }
               

            } 
        }
    }

}
```

param->pages[i] = nvp->physical_address;这个物理地址给mmap使用   