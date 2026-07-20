
```Text   
fault:		    yes		can return with page locked   
page_mkwrite:	yes		can return with page locked   
	->fault() is called when a previously not present pte is about
to be faulted in. The filesystem must find and return the page associated
with the passed in "pgoff" in the vm_fault structure. If it is possible that
the page may be truncated and/or invalidated, then the filesystem must lock
the page, then ensure it is not already truncated (the page lock will block
subsequent truncate), and then return with VM_FAULT_LOCKED, and the page
locked. The VM will unlock the page.

	->page_mkwrite() is called when a previously read-only pte is
about to become writeable. The filesystem again must ensure that there are
no truncate/invalidate races, and then return with the page locked. If
the page has been truncated, the filesystem should not look up a new page
like the ->fault() handler, but simply return with VM_FAULT_NOPAGE, which
will cause the VM to retry the fault.

	->access() is called when get_user_pages() fails in
acces_process_vm(), typically used to debug a process through
/proc/pid/mem or ptrace.  This function is needed only for
VM_IO | VM_PFNMAP VMAs.

```

```
} else if (unlikely((vma->vm_flags & (VM_WRITE|VM_SHARED)) ==
					(VM_WRITE|VM_SHARED))) {
		/*
		 * Only catch write-faults on shared writable pages,
		 * read-only shared pages can get COWed by
		 * get_user_pages(.write=1, .force=1).
		 */
		if (vma->vm_ops && vma->vm_ops->page_mkwrite) {
			struct vm_fault vmf;
			int tmp;

			vmf.virtual_address = (void __user *)(address &
								PAGE_MASK);
			vmf.pgoff = old_page->index;
			vmf.flags = FAULT_FLAG_WRITE|FAULT_FLAG_MKWRITE;
			vmf.page = old_page;

			/*
			 * Notify the address space that the page is about to
			 * become writable so that it can prohibit this or wait
			 * for the page to get into an appropriate state.
			 *
			 * We do this without the lock held, so that it can
			 * sleep if it needs to.
			 */
			page_cache_get(old_page);
			pte_unmap_unlock(page_table, ptl);

			tmp = vma->vm_ops->page_mkwrite(vma, &vmf);
			if (unlikely(tmp &
					(VM_FAULT_ERROR | VM_FAULT_NOPAGE))) {
				ret = tmp;
				goto unwritable_page;
			}
			if (unlikely(!(tmp & VM_FAULT_LOCKED))) {
				lock_page(old_page);
				if (!old_page->mapping) {
					ret = 0; /* retry the fault */
					unlock_page(old_page);
					goto unwritable_page;
				}
			} else
				VM_BUG_ON(!PageLocked(old_page));

```
# test

```
static int op_page_mkwrite(struct vm_fault *vmf)
{
     pr_info("%s \n",__func__);
     //return mmap_fault(vmf);
     //return VM_FAULT_NOPAGE;
     return VM_FAULT_LOCKED;
}
struct vm_operations_struct mmap_vm_ops = {
        .open = mmap_open,
        .close = mmap_close,
        .fault = mmap_fault,
#if 1
        .page_mkwrite = op_page_mkwrite,
#endif
};
```
的返回值不VM_FAULT_LOCKED，内核会hang    


+ kernel 先执行 mmap_fault 后执行 op_page_mkwrite   
```
[21950.346169] sample char device init
[21950.346289] mmap-example: mmap-test registered with major 241
[21974.809204] page index offset = 0 
[21974.812592] op_page_mkwrite 
[21974.815470] page index offset = 1 
[21974.818856] op_page_mkwrite 
[21974.821723] page index offset = 2 
[21974.825114] op_page_mkwrite 
[21974.827982] page index offset = 3 
[21974.831367] op_page_mkwrite 
```


```
/*
 * Notify the address space that the page is about to become writable so that
 * it can prohibit this or wait for the page to get into an appropriate state.
 *
 * We do this without the lock held, so that it can sleep if it needs to.
 */
static vm_fault_t do_page_mkwrite(struct vm_fault *vmf)
{
	vm_fault_t ret;
	struct page *page = vmf->page;
	unsigned int old_flags = vmf->flags;

	vmf->flags = FAULT_FLAG_WRITE|FAULT_FLAG_MKWRITE;

	if (vmf->vma->vm_file &&
	    IS_SWAPFILE(vmf->vma->vm_file->f_mapping->host))
		return VM_FAULT_SIGBUS;

	ret = vmf->vma->vm_ops->page_mkwrite(vmf);
	/* Restore original flags so that caller is not surprised */
	vmf->flags = old_flags;
	if (unlikely(ret & (VM_FAULT_ERROR | VM_FAULT_NOPAGE)))
		return ret;
	if (unlikely(!(ret & VM_FAULT_LOCKED))) {
		lock_page(page);
		if (!page->mapping) {
			unlock_page(page);
			return 0; /* retry */
		}
		ret |= VM_FAULT_LOCKED;
	} else
		VM_BUG_ON_PAGE(!PageLocked(page), page);
	return ret;
}

```

# mr_vm_mkwrite

```
/* Notify when pages are promoted from WriteProtect to Read/Write */
static int mr_vm_mkwrite(struct vm_fault *vmf)
{
        struct vm_area_struct *vma = vmf->vma;
        pr_info("mkwrite: vma=%p, file=%p, as=%p, va=%p\n", vma, vma->vm_file,
                        vma->vm_file ? vma->vm_file->f_mapping : NULL,
                        (void*)vmf->address);
        return VM_FAULT_LOCKED;
}
```

#   mmap syscall

mm/mmap.c mmap_pgoff---->do_mmap_pgoff---->mmap_region---->file->f_op->mmap---->ext4_file_mmap---->vma->vm_ops = &ext4_file_mmap.   

设置:    
vm_ops.fault=filemap_fault;    
vm_ops.page_mkwrite=ext4_page_mkwrite   

 
> ##write for first time

```
#10 [ffff95a25df2fbf8] file_update_time at ffffffff9d268080
#11 [ffff95a25df2fc38] ext4_page_mkwrite at ffffffffc020bf0d  [ext4]
#12 [ffff95a25df2fc90] do_page_mkwrite at ffffffff9d1ebeba
#13 [ffff95a25df2fd10] do_wp_page at ffffffff9d1ef977
#14 [ffff95a25df2fdb8] handle_mm_fault at ffffffff9d1f3cd2
#15 [ffff95a25df2fe80] __do_page_fault at ffffffff9d788653
#16 [ffff95a25df2fef0] trace_do_page_fault at ffffffff9d788a26
#17 [ffff95a25df2ff30] do_async_page_fault at ffffffff9d787fa2
#18 [ffff95a25df2ff50] async_page_fault at ffffffff9d7847a8
```



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
# 页写保护处理 do_wp_page
linux0.12采取了load-on-demand和copy-on-write的策略，当调用系统调用fork出一个子进程时，子进程和父亲进程会共享物理内存，并且子进程和父进程的页表项会被置为只读，当子进程或者父进程的任何一方企图对自己的进程空间线性地址进行写操作时，CPU会触发写保护中断操作，do_wp_page这时候会被调用。其主要作用就是新开辟一页内存，以便进程空间线性地址进行写操作可以继续执行   


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


```
static vm_fault_t sel_mmap_policy_fault(struct vm_fault *vmf)
{
	struct policy_load_memory *plm = vmf->vma->vm_file->private_data;
	unsigned long offset;
	struct page *page;

	if (vmf->flags & (FAULT_FLAG_MKWRITE | FAULT_FLAG_WRITE))
		return VM_FAULT_SIGBUS;

	offset = vmf->pgoff << PAGE_SHIFT;
	if (offset >= roundup(plm->len, PAGE_SIZE))
		return VM_FAULT_SIGBUS;

	page = vmalloc_to_page(plm->data + offset);
	get_page(page);

	vmf->page = page;

	return 0;
}

static const struct vm_operations_struct sel_mmap_policy_ops = {
	.fault = sel_mmap_policy_fault,
	.page_mkwrite = sel_mmap_policy_fault,
};
``` 

```
int filemap_page_mkwrite(struct vm_fault *vmf)
{
	struct page *page = vmf->page;
	struct inode *inode = file_inode(vmf->vma->vm_file);
	int ret = VM_FAULT_LOCKED;

	sb_start_pagefault(inode->i_sb);
	file_update_time(vmf->vma->vm_file);
	lock_page(page);
	if (page->mapping != inode->i_mapping) {
		unlock_page(page);
		ret = VM_FAULT_NOPAGE;
		goto out;
	}
	/*
	 * We mark the page dirty already here so that when freeze is in
	 * progress, we are guaranteed that writeback during freezing will
	 * see the dirty page and writeprotect it again.
	 */
	set_page_dirty(page);
	wait_for_stable_page(page);
out:
	sb_end_pagefault(inode->i_sb);
	return ret;
}
EXPORT_SYMBOL(filemap_page_mkwrite);
```

# ceph


```
/*
 * vm ops
 */

/*
 * Reuse write_begin here for simplicity.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 30)
static int ceph_page_mkwrite(struct vm_area_struct *vma, struct vm_fault *vmf)
#else
static int ceph_page_mkwrite(struct vm_area_struct *vma, struct page *page)
#endif
{
	struct inode *inode = vma->vm_file->f_dentry->d_inode;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 30)
	struct page *page = vmf->page;
#endif
	struct ceph_mds_client *mdsc = &ceph_inode_to_client(inode)->mdsc;
	loff_t off = page->index << PAGE_CACHE_SHIFT;
	loff_t size, len;
	int ret;

	size = i_size_read(inode);
	if (off + PAGE_CACHE_SIZE <= size)
		len = PAGE_CACHE_SIZE;
	else
		len = size & ~PAGE_CACHE_MASK;

	dout("page_mkwrite %p %llu~%llu page %p idx %lu\n", inode,
	     off, len, page, page->index);

	lock_page(page);

	ret = VM_FAULT_NOPAGE;
	if ((off > size) ||
	    (page->mapping != inode->i_mapping))
		goto out;

	ret = ceph_update_writeable_page(vma->vm_file, off, len, page);
	if (ret == 0) {
		/* success.  we'll keep the page locked. */
		set_page_dirty(page);
		up_read(&mdsc->snap_rwsem);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 30)
		ret = VM_FAULT_LOCKED;
#else
		unlock_page(page);
#endif
	} else {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 30)
		if (ret == -ENOMEM)
			ret = VM_FAULT_OOM;
		else
			ret = VM_FAULT_SIGBUS;
#endif
	}
out:
	dout("page_mkwrite %p %llu~%llu = %d\n", inode, off, len, ret);
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 30)
	if (ret != VM_FAULT_LOCKED)
		unlock_page(page);
#endif
	return ret;
}

static struct vm_operations_struct ceph_vmops = {
	.fault		= filemap_fault,
	.page_mkwrite	= ceph_page_mkwrite,
};
```
ceph_page_mkwrite(struct vm_area_struct *vma, struct vm_fault *vmf)    当页面从readonly状态变迁到writeable状态时该函数被调用     

|__调用ceph_update_writeable_page()函数来设置vmf->page页为writeable    

|__调用set_page_diry()函数设置该物理内存页为dirty的   
```
int ceph_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct address_space *mapping = file->f_mapping;

	if (!mapping->a_ops->readpage)
		return -ENOEXEC;
	file_accessed(file);
	vma->vm_ops = &ceph_vmops;
	vma->vm_flags |= VM_CAN_NONLINEAR;
	return 0;
}
```