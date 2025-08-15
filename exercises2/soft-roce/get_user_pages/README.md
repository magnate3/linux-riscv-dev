

# 获取虚拟地址的物理地址

```
/*
 * omap_vout_uservirt_to_phys: This inline function is used to convert user
 * space virtual address to physical address.
 */
static u32 omap_vout_uservirt_to_phys(u32 virtp)
{
	unsigned long physp = 0;
	struct vm_area_struct *vma;
	struct mm_struct *mm = current->mm;

	vma = find_vma(mm, virtp);
	/* For kernel direct-mapped memory, take the easy way */
	if (virtp >= PAGE_OFFSET) {
		physp = virt_to_phys((void *) virtp);
	} else if (vma && (vma->vm_flags & VM_IO) && vma->vm_pgoff) {
		/* this will catch, kernel-allocated, mmaped-to-usermode
		   addresses */
		physp = (vma->vm_pgoff << PAGE_SHIFT) + (virtp - vma->vm_start);
	} else {
		/* otherwise, use get_user_pages() for general userland pages */
		int res, nr_pages = 1;
		struct page *pages;
		down_read(&current->mm->mmap_sem);

		res = get_user_pages(current, current->mm, virtp, nr_pages, 1,
				0, &pages, NULL);
		up_read(&current->mm->mmap_sem);

		if (res == nr_pages) {
			physp =  __pa(page_address(&pages[0]) +
					(virtp & ~PAGE_MASK));
		} else {
			printk(KERN_WARNING VOUT_NAME
					"get_user_pages failed\n");
			return 0;
		}
	}

	return physp;
}
```

#   get_user_pages_fast

// vhost/vhost.c

```

/* TODO: This is really inefficient.  We need something like get_user()
 * (instruction directly accesses the data, with an exception table entry
 * returning -EFAULT). See Documentation/x86/exception-tables.txt.
 */
static int set_bit_to_user(int nr, void __user *addr)
{
        unsigned long log = (unsigned long)addr;
        struct page *page;
        void *base;
        int bit = nr + (log % PAGE_SIZE) * 8;
        int r;

        r = get_user_pages_fast(log, 1, 1, &page);
        if (r < 0)
                return r;
        BUG_ON(r != 1);
        base = kmap_atomic(page);
        set_bit(bit, base);
        kunmap_atomic(base);
        set_page_dirty_lock(page);
        put_page(page);
        return 0;
}

```
# get_user_pages(pin page)
    通常一个应用程序使用malloc或者mmap申请到的只是虚拟内存，只有在第一次访问该地址时触发page fault才为其申请物理内存称为按需分配内存。整个映射过程其实用户是无法感知，但是触发一次page fault非常耗时。有时应用程序为了优化性能，通常采用pin memory的形式即使用mmap/malloc时提前将虚拟内存与对应物理内存锁定，以提高性能，这也是很多程序常用优化方法。   

    pin memory好处还有另外一个优势就是可以防止内存被swap out置换到存储器中，如果在进程切换时该物理内存被swap out磁盘中，下次读取还需要从磁盘加载到内存中，整个过程非常耗时，通过使用pin memory可以将一些主要常用的内存锁住，以防止被置换出去同时防止进行各种原因造成的页迁移，以提高程序性能。   

    pin memory最大坏处就是：如果每个程序都大量使用pin memory，那么最后将会导致没有物理内存可用，所以一般社区开发不建议在大量长期时候的内存使用pin memory类型内存。   

    除了应用程序使用pin memory之外，在很多驱动程序中也经常用到pin memory， 例如网卡会将用于收报包内存使用pin memory内存以防止被置换出去。   

   内核中使用pin memory接口主要是get_user_pages()函数，该函数处理主要位于mm/gup.c文件中  
   
```
[ 163.592768] [<ffffff8008088c58>] dump_backtrace\+0x0/0x368
[ 163.598212] [<ffffff8008088fd4>] show_stack\+0x14/0x20
[ 163.603318] [<ffffff8008a3f7f8>] dump_stack\+0x9c/0xbc
[ 163.608421] [<ffffff800813c468>] dump_header.isra.6\+0x7c/0x194
[ 163.614302] [<ffffff800813bad8>] oom_kill_process\+0x280/0x500
[ 163.620098] [<ffffff800813c088>] out_of_memory\+0xe0/0x3e8
[ 163.625549] [<ffffff80081410f8>] __alloc_pages_nodemask\+0xa20/0xad8
[ 163.631865] [<ffffff8008163d2c>] __pte_alloc\+0x2c/0x130
[ 163.637141] [<ffffff8008167b48>] __handle_mm_fault\+0x678/0xb90
[ 163.643023] [<ffffff80081680b8>] handle_mm_fault\+0x58/0xa0
[ 163.648560] [<ffffff80081614d0>] __get_user_pages\+0x190/0x310
[ 163.654356] [<ffffff8008161c70>] populate_vma_page_range\+0x60/0x68
[ 163.660584] [<ffffff8008161d28>] __mm_populate\+0xb0/0x178
[ 163.666034] [<ffffff800816ddb4>] SyS_brk\+0x114/0x19
```

## __get_user_pages处理流程
 
_get_user_pages核心处理思想就是调用faultin_page ()触发page fault流程，为其申请一个物理页

# page offset


```
  unsigned long vaddr = (unsigned long)buf;
  unsigned long page_offset = vaddr & ~PAGE_MASK;
   myaddr = (char *)((unsigned long)page_addr+ page_offset) ;
```

不需要page size对齐也可以了
```
[root@centos7 get_user_pages]# ./test 
page size is 65536 
data is Mohan
[root@centos7 get_user_pages]# 
```