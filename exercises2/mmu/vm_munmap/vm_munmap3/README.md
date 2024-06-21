
+ user    
申请start_addr = 0xffff95010000的虚拟地址
```
   ctrl.user_addr = 0xffff95010000;
   if(ioctl(fd, IOCTL_ALLOC_VMA, &ctrl) < 0){
           printf("Testcase Failed\n");
           goto err2;
       }
```


+ kernel  
vm_mmap(file, user_addr, total  申请start_addr = 0xffff95010000的虚拟地址   
```
user_addr = addr.user_addr;
        if(0 != user_addr) {
             vm_area = find_vma(current->mm, user_addr);
             if(NULL == vm_area)
             {
                  pr_info("vma addr 0x%lx  not exist \n", user_addr);
             }
             else
             {
                  pr_info("vma addr 0x%lx  exist \n", user_addr);
             }
        }
        user_addr = vm_mmap(file, user_addr, total,
                            PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_ANONYMOUS | MAP_PRIVATE, 0);
```  

#  find_vma



```
        if(0 != user_addr) {
             vm_area = find_vma(current->mm, user_addr);
             if(NULL == vm_area)
             {
                  pr_info("vma addr 0x%lx  not exist \n", user_addr);
             }
             else
             {
                  pr_info("vma addr 0x%lx  exist \n", user_addr);
             }
        }
```

```
[root@centos7 vm_munmap2]# ./test-mmap 
alloced virt addr : 0xffff95010000 
Changed message: Hello from *user* this is file: mmap-test

Write/Read test ...
[root@centos7 vm_munmap2]# 
```

kernel msg:   
用户空间 0xffff95010000地址没有使用， 但是find_vma返回不是NULL       
```
[841111.574452] mmap-example: mmap-test registered with major 241
[842992.391234] vma addr 0xffff95010000  exist 
[842992.395556] page index offset = 0 
[842992.399066] page index offset = 1 
[842992.402575] page index offset = 2 
[842992.406067] page index offset = 3 
```

#  do_mmap
```
unsigned long do_mmap(struct file *file, unsigned long addr,
            unsigned long len, unsigned long prot,
            unsigned long flags, vm_flags_t vm_flags,
            unsigned long pgoff, unsigned long *populate,
            struct list_head *uf)
{
    struct mm_struct *mm = current->mm;
    // 在进程虚拟内存空间中寻找一块未映射的虚拟内存范围
    // 这段虚拟内存范围后续将会用于 mmap 内存映射
    addr = get_unmapped_area(file, addr, len, pgoff, flags);

   
    // 这里就是 mmap 内存映射的核心
    addr = mmap_region(file, addr, len, vm_flags, pgoff, uf);

    // 当 mmap 设置了 MAP_POPULATE 或者 MAP_LOCKED 标志
    // 那么在映射完之后，需要立马为这块虚拟内存分配物理内存页，后续访问就不会发生缺页了
    if (!IS_ERR_VALUE(addr) &&
        ((vm_flags & VM_LOCKED) ||
         (flags & (MAP_POPULATE | MAP_NONBLOCK)) == MAP_POPULATE))
        // 设置需要分配的物理内存大小
        *populate = len;
    return addr;
}
```

#  do_munmap  存在的vma    


```
/*
 * detach and kill segment if marked destroyed.
 * The work is done in shm_close.
 */
SYSCALL_DEFINE1(shmdt, char __user *, shmaddr)
{
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma;
	unsigned long addr = (unsigned long)shmaddr;
	int retval = -EINVAL;
#ifdef CONFIG_MMU
	loff_t size = 0;
	struct vm_area_struct *next;
#endif

	if (addr & ~PAGE_MASK)
		return retval;

	down_write(&mm->mmap_sem);

	/*
	 * This function tries to be smart and unmap shm segments that
	 * were modified by partial mlock or munmap calls:
	 * - It first determines the size of the shm segment that should be
	 *   unmapped: It searches for a vma that is backed by shm and that
	 *   started at address shmaddr. It records it's size and then unmaps
	 *   it.
	 * - Then it unmaps all shm vmas that started at shmaddr and that
	 *   are within the initially determined size.
	 * Errors from do_munmap are ignored: the function only fails if
	 * it's called with invalid parameters or if it's called to unmap
	 * a part of a vma. Both calls in this function are for full vmas,
	 * the parameters are directly copied from the vma itself and always
	 * valid - therefore do_munmap cannot fail. (famous last words?)
	 */
	/*
	 * If it had been mremap()'d, the starting address would not
	 * match the usual checks anyway. So assume all vma's are
	 * above the starting address given.
	 */
	vma = find_vma(mm, addr);

#ifdef CONFIG_MMU
	while (vma) {
		next = vma->vm_next;

		/*
		 * Check if the starting address would match, i.e. it's
		 * a fragment created by mprotect() and/or munmap(), or it
		 * otherwise it starts at this address with no hassles.
		 */
		if ((vma->vm_ops == &shm_vm_ops) &&
			(vma->vm_start - addr)/PAGE_SIZE == vma->vm_pgoff) {


			size = vma->vm_file->f_path.dentry->d_inode->i_size;
			do_munmap(mm, vma->vm_start, vma->vm_end - vma->vm_start);
			/*
			 * We discovered the size of the shm segment, so
			 * break out of here and fall through to the next
			 * loop that uses the size information to stop
			 * searching for matching vma's.
			 */
			retval = 0;
			vma = next;
			break;
		}
		vma = next;
	}

	/*
	 * We need look no further than the maximum address a fragment
	 * could possibly have landed at. Also cast things to loff_t to
	 * prevent overflows and make comparisons vs. equal-width types.
	 */
	size = PAGE_ALIGN(size);
	while (vma && (loff_t)(vma->vm_end - addr) <= size) {
		next = vma->vm_next;

		/* finding a matching vma now does not alter retval */
		if ((vma->vm_ops == &shm_vm_ops) &&
			(vma->vm_start - addr)/PAGE_SIZE == vma->vm_pgoff)

			do_munmap(mm, vma->vm_start, vma->vm_end - vma->vm_start);
		vma = next;
	}

#else /* CONFIG_MMU */
	/* under NOMMU conditions, the exact address to be destroyed must be
	 * given */
	retval = -EINVAL;
	if (vma->vm_start == addr && vma->vm_ops == &shm_vm_ops) {
		do_munmap(mm, vma->vm_start, vma->vm_end - vma->vm_start);
		retval = 0;
	}

#endif

	up_write(&mm->mmap_sem);
	return retval;
}
```

#  get_unmapped_area

vm_mmap -->  vm_mmap_pgoff  --> do_mmap -->  get_unmapped_area   
```
[839776.297906] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[839776.303022] [<ffff0000016b0624>] sgx_get_unmapped_area+0x2c/0x60 [mmap_example]
[839776.310385] [<ffff000008254a2c>] get_unmapped_area.part.37+0x60/0xd0
[839776.316794] [<ffff000008257f74>] do_mmap+0x12c/0x34c
[839776.321823] [<ffff00000823467c>] vm_mmap_pgoff+0xf0/0x124
[839776.327283] [<ffff00000823471c>] vm_mmap+0x6c/0x80
[839776.332131] [<ffff0000016b046c>] device_ioctl+0x14c/0x298 [mmap_example]
[839776.338888] [<ffff0000082c6e18>] do_vfs_ioctl+0xcc/0x8fc
[839776.344263] [<ffff0000082c76d8>] SyS_ioctl+0x90/0xa4
```

```Text    
查找一个空闲的地址区间：get_unmapped_area
参数：
len，指定区间的长度，
addr，非空的addr指定必须从哪个地址开始查找。
返回值：
如查找成功，返回这个新区间的起始地址；否则，返回错误码-ENOMEM。

如addr不等于NULL，就检查所指定的地址是否在用户态空间并与页边界对齐。函数根据线性地址区间是否应用于文件内存映射或匿名内存映射，调两个方法（get_unmapped_area文件操作和内存描述符的get_unmapped_area）中的一个。

前一种情况下，函数执行get_unmapped_area文件操作。第二种情况下，函数执行内存描述符的get_unmapped_area。根据进程的线性区类型，由函数arch_get_unmapped_area或arch_get_unmapped_area_topdown实现get_unmapped_area。

通过系统调用map，每个进程可获得两种不同形式的线区：
一种从线性地址0x40000000开始并向高端地址增长，
另一种正好从用户态堆栈开始并向低端地址增长。

在分配从低端地址向高端地址移动的线性区时使用arch_get_unmapped_area。
if(len > TASK_SIEZ)
	return -ENOMEM;
addr = (addr + 0xfff) & 0xfffff000;
if(addr & addr + len <= TASK_SIZE)
{
	vma = find_vma(current->mm, addr);
	if(!vma || addr + len <= vma->vm_start)
		return addr;
}

start_addr = addr = mm->free_area_cache;
for(vma = find_vma(current->mm, addr); ; vma = vma->vm_next)
{
	if(addr + len > TASK_SIZE)
	{
		if(start_addr == (TASK_SIZE/3 + 0xfff) & 0xfffff000)
			return -ENOMEM;
		start_addr = addr = (TASK_SIZE/3 + 0xfff) & 0xfffff000;// 这是允许的最低起始线性地址
		vma = find_vma(current->mm, addr);
	}
	
	if(!vma || addr + len <= vma->vm_start)
	{
		mm->free_area_cache = addr + len;
		return addr;// 返回线性地址是满足分配要求线性区（尚未分配）的起始地址
	}
	
	addr = vma->vm_end;
}

```


```Text
函数先检查区间的长度是否在用户态下线性地址区间的限长TASK_SIZE之内。
如addr不为0，函数就试图从addr开始分配区间。为安全，函数把addr值调整为4KB倍数。
如addr等于0或前面的搜索失败，arch_get_unmapped_area就扫描用户态线性地址空间以查找一个可包含新区的足够大的线性地址范围。但任何已有的线性区都不包括这个地址范围。

为提高搜索速度，让搜索从最近被分配的线性区后面的线性地址开始，把内存描述符的字段mm->free_area_cache初始化为用户态线性地址空间的三分之一，并在以后创建新线性区时对它更新。如找不到一个合适的线性地址范围，就从用户态线性地址空间的三分之一的开始处重新开始搜索。其实，用户态线性地址空间的三分之一是为有预定义起始线性地址的线性区（典型的是可执行文件的正文段，数据段，bss段）而保留的。

函数调find_vma以确定搜索起点后第一个线性区终点的位置。三种情况：
(1).如所请求的区间大于正待扫描的线性地址空间部分（addr+len>TASK_SIZE），函数就从用户态地址空间的三分之一处重新开始搜索，如已完成第二次搜索，就返回-ENOMEM。
(2).刚扫描过的线性区后面的空闲区没足够的大小，vma != NULL && vma->vm_start < addr + len此时，继续考虑下一个线性区。
(3).如以上两情况都没发生，则找到一个足够大的空闲区。函数返回addr。


```


# 添加vma   


##  mmap_region

[从内核世界透视 mmap 内存映射的本质（源码实现篇）](https://ost.51cto.com/posts/27288)   

vm_mmap  -->  vm_mmap_pgoff -->   do_mmap -->   mmap_region-->   op_mmap   
```
[841030.648739] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[841030.653855] [<ffff0000017e0200>] op_mmap+0x20/0x68 [mmap_example]
[841030.660008] [<ffff000008257c74>] mmap_region+0x348/0x51c
[841030.665380] [<ffff000008258138>] do_mmap+0x2f0/0x34c
[841030.670410] [<ffff00000823467c>] vm_mmap_pgoff+0xf0/0x124
[841030.675870] [<ffff00000823471c>] vm_mmap+0x6c/0x80
[841030.680726] [<ffff0000017e04c4>] device_ioctl+0x14c/0x298 [mmap_example]
[841030.687484] [<ffff0000082c6e18>] do_vfs_ioctl+0xcc/0x8fc
[841030.692856] [<ffff0000082c76d8>] SyS_ioctl+0x90/0xa4
```

```
unsigned long
mmap_region(struct file *file, unsigned long addr,
            unsigned long len, unsigned long flags,
            unsigned int vm_flags, unsigned long pgoff,
            int accountable)
{
    struct mm_struct *mm = current->mm;
    struct vm_area_struct *vma, *prev;
    int correct_wcount = 0;
    int error;
    ...

    // 1. 申请一个虚拟内存区管理结构(vma)
    vma = kmem_cache_zalloc(vm_area_cachep, GFP_KERNEL);
    ...

    // 2. 设置vma结构各个字段的值
    vma->vm_mm = mm;
    vma->vm_start = addr;
    vma->vm_end = addr + len;
    vma->vm_flags = vm_flags;
    vma->vm_page_prot = protection_map[vm_flags & (VM_READ|VM_WRITE|VM_EXEC|VM_SHARED)];
    vma->vm_pgoff = pgoff;

    if (file) {
        ...
        vma->vm_file = file;

        /* 3. 此处是内存映射的关键点，调用文件对象的 mmap() 回调函数来设置vma结构的 fault() 回调函数。
         *    vma对象的 fault() 回调函数的作用是：
         *        - 当访问的虚拟内存没有映射到物理内存时，
         *        - 将会调用 fault() 回调函数对虚拟内存地址映射到物理内存地址。
         */
        error = file->f_op->mmap(file, vma);
        ...
    }
    ...

    // 4. 把 vma 结构连接到进程虚拟内存区的链表和红黑树中。
    vma_link(mm, vma, prev, rb_link, rb_parent);
    ...

    return addr;
}

```

mmap_region() 函数主要完成以下 4 件事情：    


+ 1 申请一个 vm_area_struct 结构（vma），内核使用 vma 来管理进程的虚拟内存地址。   


+ 2 设置 vma 结构各个字段的值。    


+ 3  通过调用文件对象的 mmap() 回调函数来设置vma结构的 fault() 回调函数，一般文件对象的 mmap() 回调函数为：generic_file_mmap()。   


+ 4  把新创建的 vma 结构连接到进程的虚拟内存区链表和红黑树中。     

 
##   add_vma_to_mm
```
/*
 * add a VMA into a process's mm_struct in the appropriate place in the list
 * and tree and add to the address space's page tree also if not an anonymous
 * page
 * - should be called with mm->mmap_lock held writelocked
 */
static void add_vma_to_mm(struct mm_struct *mm, struct vm_area_struct *vma)
{
	struct vm_area_struct *pvma, *prev;
	struct address_space *mapping;
	struct rb_node **p, *parent, *rb_prev;

	BUG_ON(!vma->vm_region);

	mm->map_count++;
	vma->vm_mm = mm;

	/* add the VMA to the mapping */
	if (vma->vm_file) {
		mapping = vma->vm_file->f_mapping;

		i_mmap_lock_write(mapping);
		flush_dcache_mmap_lock(mapping);
		vma_interval_tree_insert(vma, &mapping->i_mmap);
		flush_dcache_mmap_unlock(mapping);
		i_mmap_unlock_write(mapping);
	}

	/* add the VMA to the tree */
	parent = rb_prev = NULL;
	p = &mm->mm_rb.rb_node;
	while (*p) {
		parent = *p;
		pvma = rb_entry(parent, struct vm_area_struct, vm_rb);

		/* sort by: start addr, end addr, VMA struct addr in that order
		 * (the latter is necessary as we may get identical VMAs) */
		if (vma->vm_start < pvma->vm_start)
			p = &(*p)->rb_left;
		else if (vma->vm_start > pvma->vm_start) {
			rb_prev = parent;
			p = &(*p)->rb_right;
		} else if (vma->vm_end < pvma->vm_end)
			p = &(*p)->rb_left;
		else if (vma->vm_end > pvma->vm_end) {
			rb_prev = parent;
			p = &(*p)->rb_right;
		} else if (vma < pvma)
			p = &(*p)->rb_left;
		else if (vma > pvma) {
			rb_prev = parent;
			p = &(*p)->rb_right;
		} else
			BUG();
	}

	rb_link_node(&vma->vm_rb, parent, p);
	rb_insert_color(&vma->vm_rb, &mm->mm_rb);

	/* add VMA to the VMA list also */
	prev = NULL;
	if (rb_prev)
		prev = rb_entry(rb_prev, struct vm_area_struct, vm_rb);

	__vma_link_list(mm, vma, prev);
}
```
 
 