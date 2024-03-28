

#  内核  gpumem_mmap

```
int remap_pfn_mmap(struct file *file, struct vm_area_struct *vma)
{
    size_t size = vma->vm_end - vma->vm_start;
            if (!(vma->vm_flags & VM_MAYSHARE))
                        return -EINVAL;

            vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

            if (remap_pfn_range(vma, vma->vm_start, vma->vm_pgoff, size, vma->vm_page_prot)) {
                pr_err("%s(): error in remap_page_range.\n", __func__ );
                return -EAGAIN;
            }

            return 0;
}
```

参考   
[gpudma/module/gpumemdrv.c](https://github.com/karakozov/gpudma/blob/master/module/gpumemdrv.c)   
```
int gpumem_mmap(struct file *file, struct vm_area_struct *vma)
{
    size_t size = vma->vm_end - vma->vm_start;

    if (!(vma->vm_flags & VM_MAYSHARE))
        return -EINVAL;

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

    if (remap_pfn_range(vma,
                        vma->vm_start,
                        vma->vm_pgoff,
                        size,
                        vma->vm_page_prot)) {
        pr_err("%s(): error in remap_page_range.\n", __func__ );
        return -EAGAIN;
    }

    return 0;
}
```

# 用户程序 



```
 ret = posix_memalign((void **)&ptr, pagesize, pagesize);
 memcpy(ptr, str, strlen(str) +1);
 phyaddr = mem_virt2phy(ptr);
 addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, phyaddr);
```
通过mmap传递物理地址    


# 运行


```
root@ubuntux86:# ./mmap_test 
virt addr 0x5627ec8ae000, phy addr of ptr  0x15ae8c000 
addr: 0x7fc9ab8fa000 
buf is: hello krishna 

Write/Read test ...
0x66616365
root@ubuntux86:# 
```