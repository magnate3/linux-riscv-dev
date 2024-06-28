

# 获取nvidia_p2p_get_pages的cpuaddr虚拟地址   
参考  libdonard/libdonard/pinpool.c    
```
void *pinpool_mmap(struct pin_buf *p)
{
    int ret = ioctl(devfd, DONARD_IOCTL_SELECT_MMAP_MEMORY, p->handle);
    if (ret) {
        errno = -ret;
        return NULL;
    }

    void *addr = mmap(NULL, p->bufsize, PROT_READ | PROT_WRITE, MAP_SHARED, devfd, 0);
    if (addr == MAP_FAILED)
        return NULL;
    return addr;
}
static int pin(struct pin_buf_priv *pb)
{
    struct donard_gpu_mem gpumem = {
        .address = (__u64) pb->pub.address,
        .size = pb->pub.bufsize,
        .p2pToken = pb->tokens.p2pToken,
        .vaSpaceToken = pb->tokens.vaSpaceToken,
    };

    int ret = ioctl(devfd, DONARD_IOCTL_PIN_GPU_MEMORY, &gpumem);

    pb->pub.handle = gpumem.handle;

    return ret;
}

```
+ 1 用户态调用DONARD_IOCTL_PIN_GPU_MEMORY，内核态调用nvidia_p2p_get_pages分配GPU 内存   
+ 2 用户态调用mmap 分配cpuaddr虚拟地址   

+ 3 内核态mmap 通过remap_pfn_range 建立cpuaddr虚拟地址  和nvidia_p2p_get_pages的gpu physical_address的映射     

```
static int donard_pinbuf_mmap(struct file *filp, struct vm_area_struct *vma)
{
    uint32_t page_size;
    unsigned long addr = vma->vm_start;
    struct nvidia_p2p_page_table *pages;
    int i;
    int ret;

    if (mmap_page_handle == NULL)
        return -EINVAL;

    pages = mmap_page_handle->page_table;

    switch(pages->page_size) {
    case NVIDIA_P2P_PAGE_SIZE_4KB:   page_size =   4*1024; break;
    case NVIDIA_P2P_PAGE_SIZE_64KB:  page_size =  64*1024; break;
    case NVIDIA_P2P_PAGE_SIZE_128KB: page_size = 128*1024; break;
    default:
        return -EIO;
    }


    for (i = 0; i < pages->entries; i++) {
        if (addr+page_size > vma->vm_end) break;

        if ((ret = remap_pfn_range(vma, addr,
                                   pages->pages[i]->physical_address >> PAGE_SHIFT,
                                   page_size, vma->vm_page_prot))) {

            printk("donard: remap %d failed: %d\n", i, ret);
            return -EAGAIN;
        }
        addr += page_size;
    }

    return 0;
}

```

