

# page index (offset >>  PAGE_SHIFT)
```
static int mmap_fault(struct vm_fault *vmf)
{       
        struct page *page;
        struct mmap_info *info;
        unsigned long offset;
        struct vm_area_struct *vma = vmf->vma;
        info = (struct mmap_info *)vma->vm_private_data;
        if (!info->data) {
                printk("No data\n");
                return 0;
        }       
        offset = ((unsigned long)vmf->address) - ((unsigned long)vmf->vma->vm_start);
        offset = offset >>  PAGE_SHIFT;
        if (offset > (1 << NR_PAGES_ORDER)) {
            printk(KERN_ERR "Invalid address deference, offset = %lu \n",
           offset);
          return 0;
        }
        printk(KERN_ERR "page index offset = %lu \n",offset);
        page = virt_to_page(info->data) + offset;
    
        get_page(page);
        vmf->page = page;
    
        return 0;
}
```

# test
```
[root@centos7 mmap-kernel-transfer-data]# ./test 
Initial message: Hello from kernel this is file: mmap-test
Changed message: Hello from *user* this is file: mmap-test

Write/Read test ...
[root@centos7 mmap-kernel-transfer-data]# dmesg | tail -n 10
[59689.756162] page index offset = 1 
[59689.759549] page index offset = 2 
[59689.762954] page index offset = 3 
[59764.248952] mmap-example: Module exit correctly
[59770.778749] sample char device init
[59770.778891] mmap-example: mmap-test registered with major 240
[59772.481462] page index offset = 0 
[59772.484870] page index offset = 1 
[59772.488257] page index offset = 2 
[59772.491661] page index offset = 3 
```
发生了4次page fault   