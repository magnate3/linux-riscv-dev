
# test   

```
[root@centos7 zap_vma_ptes-remap_pfn_range]# insmod  hybridmem_test.ko 
[root@centos7 zap_vma_ptes-remap_pfn_range]# cat /proc/devices | grep kex
241 kex
259 blkext
[root@centos7 zap_vma_ptes-remap_pfn_range]#  mknod /dev/devKex c 241 0
[root@centos7 zap_vma_ptes-remap_pfn_range]# chmod a+rw   /dev/devKex
 

```

```
[  576.556609] [<ffff000008250d1c>] vm_insert_page+0x23c/0x240
[  576.562159] [<ffff000002aa00b4>] _mmap+0xb4/0x188 [hybridmem_test]
[  576.568312] [<ffff000008257c74>] mmap_region+0x348/0x51c
[  576.573598] [<ffff000008258138>] do_mmap+0x2f0/0x34c
[  576.578542] [<ffff00000823467c>] vm_mmap_pgoff+0xf0/0x124
[  576.583916] [<ffff000008255a7c>] SyS_mmap_pgoff+0xc0/0x23c
[  576.589380] [<ffff000008089548>] sys_mmap+0x54/0x68
```

coredump的原因是BUG_ON(vma->vm_flags & VM_PFNMAP)，设置了 BUG_ON(vma->vm_flags & VM_PFNMAP)   
```
int vm_insert_page(struct vm_area_struct *vma, unsigned long addr,
                        struct page *page)
{
        if (addr < vma->vm_start || addr >= vma->vm_end)
                return -EFAULT;
        if (!page_count(page))
                return -EINVAL;
        if (!(vma->vm_flags & VM_MIXEDMAP)) {
                BUG_ON(down_read_trylock(&vma->vm_mm->mmap_sem));
                BUG_ON(vma->vm_flags & VM_PFNMAP);
                vma->vm_flags |= VM_MIXEDMAP;
        } 
        return insert_page(vma, addr, page, vma->vm_page_prot);
}       
EXPORT_SYMBOL(vm_insert_page);
```


## test1

+ user 不进行读写

```
    address = mmap (NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (address == MAP_FAILED)
    {
        perror ("mmap operation failed");
        goto err2;
    }
    //test_write_read(address);
    munmap(address,total);
```

+ kernel 先remap_pfn_range后 zap_vma_ptes

```
  for(i=0; i < pages; i++) {
    page = alloc_page(GFP_KERNEL); // TODO IO RESERVE?
    if (!page) {
      // TODO free previous pages
      printk(KERN_DEBUG "alloc_page failed\n");
      goto error;
    }
    addr = vma->vm_start+i*PAGE_SIZE;
#if 0
    if (vm_insert_page(vma,addr,page) < 0) {
      // TODO free previous pages
      printk(KERN_DEBUG "vm_insert_page failed\n");
      goto error;
    }
#else
    if (remap_pfn_range(vma, addr, page_to_pfn(page), PAGE_SIZE, (vma->vm_page_prot)) < 0) {
      printk(KERN_DEBUG "remap_pfn_range failed\n");
      goto error;
    }
#endif
    printk(KERN_DEBUG "inserted page %d at %p\n",i,(void*)addr);
    // TODO __free_page now, should be ok, since vm_insert_page incremented its
    // refcount. that way, upon munmap, refcount hits zer0, pages get freed
    __free_page(page);
    //address_space(vma);
  }
  printk(KERN_DEBUG "completed inserting %lu pages\n", pages);

  rc = 0;
//  return rc;
 error:
    for (addr = vma->vm_start, j = 0 ; addr < vma->vm_end && j < i ; addr += PAGE_SIZE, j++) {
            pr_info(" zap_vma_ptes addr 0x%lx \n",addr);
            zap_vma_ptes(vma, addr, PAGE_SIZE);
            //zap_vma_ptes(vma, vma->vm_start, addr - vma->vm_start);
    }
  return rc;
}
```

```
[ 1029.247563] vma->vm_end 281472929169408 vm_start 281472928907264 len 262144 pages 4 vm_pgoff 0
[ 1029.247567] inserted page 0 at 0000ffff85f10000
[ 1029.247568] inserted page 1 at 0000ffff85f20000
[ 1029.247569] inserted page 2 at 0000ffff85f30000
[ 1029.247570] inserted page 3 at 0000ffff85f40000
[ 1029.247571] completed inserting 4 pages
[ 1029.247571]  zap_vma_ptes addr 0xffff85f10000 
[ 1029.252008]  zap_vma_ptes addr 0xffff85f20000 
[ 1029.256433]  zap_vma_ptes addr 0xffff85f30000 
[ 1029.260862]  zap_vma_ptes addr 0xffff85f40000 
```

> ## test2

+ user  进行读写

```
    address = mmap (NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (address == MAP_FAILED)
    {
        perror ("mmap operation failed");
        goto err2;
    }
    test_write_read(address);
    munmap(address,total);
```
+ kernel 先remap_pfn_range,然后不进行 zap_vma_ptes

```
  for(i=0; i < pages; i++) {
    page = alloc_page(GFP_KERNEL); // TODO IO RESERVE?
    if (!page) {
      // TODO free previous pages
      printk(KERN_DEBUG "alloc_page failed\n");
      goto error;
    }
    addr = vma->vm_start+i*PAGE_SIZE;
#if 0
    if (vm_insert_page(vma,addr,page) < 0) {
      // TODO free previous pages
      printk(KERN_DEBUG "vm_insert_page failed\n");
      goto error;
    }
#else
    if (remap_pfn_range(vma, addr, page_to_pfn(page), PAGE_SIZE, (vma->vm_page_prot)) < 0) {
      printk(KERN_DEBUG "remap_pfn_range failed\n");
      goto error;
    }
#endif
    printk(KERN_DEBUG "inserted page %d at %p\n",i,(void*)addr);
    // TODO __free_page now, should be ok, since vm_insert_page incremented its
    // refcount. that way, upon munmap, refcount hits zer0, pages get freed
    __free_page(page);
    //address_space(vma);
  }
  printk(KERN_DEBUG "completed inserting %lu pages\n", pages);

  rc = 0;
  return rc;
 error:
    for (addr = vma->vm_start, j = 0 ; addr < vma->vm_end && j < i ; addr += PAGE_SIZE, j++) {
            pr_info(" zap_vma_ptes addr 0x%lx \n",addr);
            zap_vma_ptes(vma, addr, PAGE_SIZE);
            //zap_vma_ptes(vma, vma->vm_start, addr - vma->vm_start);
    }
  return rc;
}
```

```
[root@centos7 zap_vma_ptes-remap_pfn_range]# ./mmap_test 

Write/Read test ...
[root@centos7 zap_vma_ptes-remap_pfn_range]# 
```

```
[ 1223.878967] vma->vm_end 281472980484096 vm_start 281472980221952 len 262144 pages 4 vm_pgoff 0
[ 1223.878974] inserted page 0 at 0000ffff89000000
[ 1223.878977] inserted page 1 at 0000ffff89010000
[ 1223.878980] inserted page 2 at 0000ffff89020000
[ 1223.878983] inserted page 3 at 0000ffff89030000
[ 1223.878985] completed inserting 4 pages
```

+ 昇腾hdcdrv_mem   
[昇腾hdcdrv_mem.c](https://github.com/apulis/Apulis-AI-Platform/blob/2cf1fbb50e08b477940f5f336b1b897a49608b72/src/ClusterBootstrap/test_npu/driver/kernel/hdc_host/hdcdrv_mem.c#L994)   





```
void hdcdrv_zap_vma_ptes(struct hdcdrv_fast_mem *f_mem, struct vm_area_struct *vma, int phy_addr_num)
{
    int i;
    u32 len;
    u32 offset = 0;

    for (i = 0; i < phy_addr_num; i++) {
        len = f_mem->mem[i].len;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 18, 0)
        zap_vma_ptes(vma, f_mem->user_va + offset, f_mem->mem[i].len);
#else
        if (zap_vma_ptes(vma, f_mem->user_va + offset, f_mem->mem[i].len) != 0) {
            hdcdrv_err("va 0x%llx zap_vma_ptes failed.\n", f_mem->user_va + offset);
        }
#endif
        offset += len;
    }
}

int hdcdrv_remap_va(struct hdcdrv_fast_mem *f_mem)
{
    int i, ret;
    unsigned int len;
    unsigned int offset = 0;
    struct vm_area_struct *vma = NULL;

    if (f_mem->page_type == HDCDRV_PAGE_TYPE_HUGE) {
        return HDCDRV_OK;
    }

    down_read(&current->mm->mmap_sem);

    vma = find_vma(current->mm, f_mem->user_va);
    if (vma == NULL) {
        up_read(&current->mm->mmap_sem);
        hdcdrv_err("devid %d find vma fail va 0x%llx.\n", f_mem->devid, f_mem->user_va);
        return HDCDRV_FIND_VMA_FAIL;
    }

    ret = hdcdrv_check_va(vma, f_mem);
    if (ret != HDCDRV_OK) {
        up_read(&current->mm->mmap_sem);
        return ret;
    }

    if (hdcdrv_get_running_env() == HDCDRV_RUNNING_ENV_ARM_3559) {
        vma->vm_flags |= VM_IO | VM_SHARED;
        /*lint -e446 */
        vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
        /*lint +e446 */
    }

    for (i = 0; i < f_mem->phy_addr_num; i++) {
        len = f_mem->mem[i].len;
        if (len > 0) {
            /*lint -e648 */
            ret = remap_pfn_range(vma, f_mem->user_va + offset, page_to_pfn(f_mem->mem[i].page), len,
                                  (vma->vm_page_prot));
            /*lint +e648 */
        }
        offset += len;
        if (ret) {
            break;
        }
    }
    if (i == f_mem->phy_addr_num) {
        up_read(&current->mm->mmap_sem);
        return HDCDRV_OK;
    }

    hdcdrv_zap_vma_ptes(f_mem, vma, i);

    up_read(&current->mm->mmap_sem);

    hdcdrv_err("dev %d vma start %lx, end %lx, addr %llx, len %x remap va failed.\n", f_mem->devid, vma->vm_start,
        vma->vm_end, f_mem->user_va, f_mem->alloc_len);

    return HDCDRV_DMA_MPA_FAIL;
}
```  


```
static int
trans_mmap(struct file *f, struct vm_area_struct *vma)
{
    struct trans_channel *c = f->private_data;
    struct page *pg = NULL;
    unsigned long addr, sz, pages;
    int i;
    BUG_ON(!c);

    printl("trans_mmap, sz %d\n", vma->vm_end - vma->vm_start);
    sz = vma->vm_end - vma->vm_start;
    pages = sz/PAGE_SIZE;
    if (sz > TRANS_MAX_MAPPING) return -EINVAL;

    for (addr = vma->vm_start, i = 0 ;
            addr < vma->vm_end ;
            addr += PAGE_SIZE, i++) {
        pg = virt_to_page(&c->mem[PAGE_SIZE*i]);
        BUG_ON(!pg);

        if (vm_insert_page(vma, addr, pg)) {
            zap_vma_ptes(vma, vma->vm_start, addr - vma->vm_start);
            goto err;
        }
        //BUG_ON(pg != follow_page(vma, addr, 0));
    }
    vma->vm_flags |= (VM_RESERVED | VM_INSERTPAGE);
    vma->vm_ops = &trans_vmops;

    BUG_ON(vma->vm_private_data);
    vma->vm_private_data = c;
    c->size = sz;

    return 0;
err:
    return -EAGAIN;
}
```