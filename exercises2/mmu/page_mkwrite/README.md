
mmap syscall
这里以SLES11 SP2 3.0.80内核，ext4文件系统为例。ext4文件系统file_operations=ext4_file_operations，对应的mmap函数为ext4_file_mmap.
#   mmap syscall

mm/mmap.c mmap_pgoff---->do_mmap_pgoff---->mmap_region---->file->f_op->mmap---->ext4_file_mmap---->vma->vm_ops = &ext4_file_mmap.   

设置:    
vm_ops.fault=filemap_fault;    
vm_ops.page_mkwrite=ext4_page_mkwrite   

 
> ##write for first time
第一次写相应的页面的使用，由于页面还没有到内存中，所有会触发缺页异常  

do_page_fault-->handle_mm_fault-->handle_pte_offset   

因为vma->vm_ops不为空，所以进入 do_linear_fault  

 

do_linear_fault---->__do_fault---->vma->vm_ops->fault---->filemap_fault       

这里等待从磁盘读取页面到pagecache。处理完成后，由于使用VM_SHARED模式mmap，所以会进入vma->vm_ops->page_mkwrite     

 

vma->vm_ops->page_mkwrite---->ext4_page_mkwrite     

这里会lock_page()和wait_on_page_writeback()，如果恰好页面被其他进程锁定或者正在写回，那么会block，由于是第一次读取页面，所以一般不会在这里block。  


```
do_page_fault
          handle_pte_fault
            do_wp_page
			   -->  vma->vm_ops->page_mkwrite   
			
			
```  

#   page_mkwrite and fault





```
	static const struct vm_operations_struct vmw_vm_ops = {
		.pfn_mkwrite = vmw_bo_vm_mkwrite,
		.page_mkwrite = vmw_bo_vm_mkwrite,
		.fault = vmw_bo_vm_fault,
		.open = ttm_bo_vm_open,
		.close = ttm_bo_vm_close,
	};
```