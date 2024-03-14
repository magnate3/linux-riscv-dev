
# dpdk get_user_pages_remote
```
static inline phys_addr_t iova_to_phys(struct task_struct *tsk,
				       unsigned long iova)
{
	phys_addr_t offset, phys_addr;
	struct page *page = NULL;
	long ret;

	offset = iova & (PAGE_SIZE - 1);

	/* Read one page struct info */
#ifdef HAVE_TSK_IN_GUP
	ret = get_user_pages_remote(tsk, tsk->mm, iova, 1,
				    FOLL_TOUCH, &page, NULL, NULL);
#else
	ret = get_user_pages_remote(tsk->mm, iova, 1,
				    FOLL_TOUCH, &page, NULL, NULL);
#endif
	if (ret < 0)
		return 0;

	phys_addr = page_to_phys(page) | offset;
	put_page(page);

	return phys_addr;
}
```

# insmod  get_user_pages_remote_test.ko 

```

[root@centos7 mem_map]# insmod  get_user_pages_remote_test.ko 
[root@centos7 mem_map]# dmesg | tail -n 10
[80617.267779] vma_pages(mm->pages) = 3
[80617.271348] address of pages is: 0xffff000262e2f898
[80617.276203] address of pages1 is: 0x0
[80617.279867] ret = 3
[80617.281959] page_count(pages)) = 5
[80617.285345] address of pages is: 0xffff000262e2f898
[80617.290207] ret = 3
[80617.292299] page_count(pages1)) = 6
[80617.295770] address of pages1 is: 0xffff7fe00ffde100
[80617.300716] pfn: 0x3ff784, zone name :Normal 
[root@centos7 mem_map]# 
```

```
./arch/arm64/include/asm/pgtable.h:36:28:  
 #define vmemmap   ((struct page *)VMEMMAP_START - (memstart_addr >> PAGE_SHIFT))
                            ^
```

# insmod  vm_file_test.ko 
```
[root@centos7 mem_map]# insmod  vm_file_test.ko 
[root@centos7 mem_map]#  cat /proc/kglow
[root@centos7 mem_map]# dmesg | tail -n 10
[78451.034155] page_count(pages1)) = 4
[78451.037633] address of pages1 is: 0xffff7fe00ffde100
[79998.823379] vm_file->f_path.dentry->d_iname:  mmap_test2 
[79998.828757] vm_file->f_path.dentry->d_iname:  mmap_test2 
[79998.834144] vm_file->f_path.dentry->d_iname:  libc-2.17.so 
[79998.839697] vm_file->f_path.dentry->d_iname:  libc-2.17.so 
[79998.845244] vm_file->f_path.dentry->d_iname:  libc-2.17.so 
[79998.850795] vm_file->f_path.dentry->d_iname:  ld-2.17.so 
[79998.856168] vm_file->f_path.dentry->d_iname:  ld-2.17.so 
[79998.861550] vm_file->f_path.dentry->d_iname:  ld-2.17.so 
```