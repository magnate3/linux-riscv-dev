

#  swapp and  present

```
unsigned long data;
unsigned long offset = (vaddr / getpagesize()) * sizeof(uint64_t);
int fd = open("/proc/self/pagemap", O_RDONLY);
pread(fd, &data, sizeof(uint64_t), offset);
unsigned long PFN = data & 0x7fffffffffffff; // 0-54bits
int swapped = (data >> 62) & 1;
int present = (data >> 63) & 1;
unsigned long paddr = PFN * getpagesize() + address % getpagesize();
unsigned long kaddr = PHYS_OFFSET + getpagesize() * (PFN - PFN_MIN);
```

```
int pagemap_get_entry(PagemapEntry *entry, int pagemap_fd, uintptr_t vaddr)
{
        size_t nread;
        ssize_t ret;
        uint64_t data;

        nread = 0;
        while (nread < sizeof(data)) {
                ret = pread(pagemap_fd, &data, sizeof(data),
                                (vaddr / sysconf(_SC_PAGE_SIZE)) * sizeof(data) + nread);
                nread += ret;
                if (ret <= 0) {
                        return 1;
                }
        }
        entry->pfn = data & (((uint64_t)1 << 54) - 1);
        entry->soft_dirty = (data >> 54) & 1;
        entry->file_page = (data >> 61) & 1;
        entry->swapped = (data >> 62) & 1;
        entry->present = (data >> 63) & 1;
        return 0;
}
```

# mknod /dev/hybridmem c 224 0

```
mknod /dev/hybridmem c 224 0
```

# test2

mmap采用page fault    
```

static struct vm_operations_struct vma_ops =
{
    .fault = vma_fault,
};
```

```
root@ubuntux86:# insmod  hybridmem_test.ko 
root@ubuntux86:# ./mmap_test 
0
virt addr 7fe94afab000, phy addr 1bafea000,phy addr 1bafea000
phy addr 1600c1000
root@ubuntux86:# dmesg | tail -n 6
[21664.615390] alloc page frame struct is @ 00000000fd6fa855
[21664.615395] fault, is_write = 1, vaddr = 7fe94afab000, paddr = 1bafea000
[21664.615406] alloc page frame struct is @ 00000000a4a39d41
[21664.615409] fault, is_write = 0, vaddr = 7fe94afac000, paddr = 1600c1000
[21664.615678] page frame struct is @ 00000000fd6fa855, and user vaddr 7fe94afab000, paddr 1bafea000
[21664.615684] buffer phy addr 1bafea000, virt addr 7fe94afab000
root@ubuntux86:# 
```
phy addr 1bafea000 和内核态paddr = 1bafea000一致   
phy addr 1600c1000 和内核态paddr = 1600c1000一致 


# test3

采用 ret = remap_pfn_range(vma, vma->vm_start, pfn_start, size, vma->vm_page_prot);   

phy addr 0 和内核态paddr = 16f76b000不一致   
+ 1 只分配了一个getpagesize
```
int len = 1* getpagesize();
char* base = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
```
+ 2  printf("phy addr %lx\n", virtual_to_physical((size_t)base + getpagesize())); 


```
int main()
{
    int fd = open("/dev/hybridmem", O_RDWR);
    assert(fd > 0);

    // only one page
    int len = 1* getpagesize();
    char* base = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    assert(base != MAP_FAILED);

    strcpy(base + 2, "hello");

    printf("%d\n", *(base + getpagesize()+ 7));

    printf("virt addr %lx, phy addr %lx,phy addr %lx\n",(long unsigned)base, virtual_to_physical((size_t)base), mem_virt2phy(base));
    read(fd,base,64);
    printf("phy addr %lx\n", virtual_to_physical((size_t)base + getpagesize()));
    read(fd,(size_t)base + getpagesize(),64);
    munmap(base, len);
    close(fd); 
    return 0;
}
```
base的虚拟地址： 7fe6a222b000   
(size_t)base + getpagesize()的虚拟地址：7fe6a222c000

```
root@ubuntux86:# insmod  hybridmem_test.ko 
root@ubuntux86:# mknod /dev/hybridmem c 224 0
root@ubuntux86:# ./mmap_test 
0
Zero page frame number
page is not present!
virt addr 7fe6a222b000, phy addr 0,phy addr 0
phy addr 1549a0000
root@ubuntux86:# dmesg | tail -n 10
[  772.294082] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[  772.294087] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[  772.294092] nvme 0000:02:00.0:    [ 0] RxErr                 
[  805.689486] *********** moudle 'hybridmem' begin :
[  812.097825] mmap, start = 7fe6a222b000, end = 7fe6a222c000
[  812.097849] vaddr = 7fe6a222b000, paddr = 12c048000
[  812.098108] page frame struct is @ 00000000e618a5b4, and user vaddr 7fe6a222b000, paddr 12c048000
[  812.098115] buffer phy addr 12c048000, virt addr 7fe6a222b000
[  812.098141] page frame struct is @ 00000000fccf79cb, and user vaddr 7fe6a222c000, paddr 1549a0000
[  812.098145] buffer phy addr 1549a0000, virt addr 7fe6a222c000
```

# 通过逻辑地址查找页表Page Table
下面是基于x86体系结构，通过逻辑地址address查找Page Table指针的过程：   
```
static int __follow_pte_pmd(struct mm_struct *mm, unsigned long address,
                            unsigned long *start, unsigned long *end,
                            pte_t **ptepp, pmd_t **pmdpp, spinlock_t **ptlp)
{
        pgd_t *pgd;
        p4d_t *p4d;
        pud_t *pud;
        pmd_t *pmd;
        pte_t *ptep;

	/* 进程的Page Global Directory基指针存放在mm->pgd */
        pgd = pgd_offset(mm, address);	  /* 返回address指向该进程的Page Global Directory指针 */
        if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd)))
                goto out;

        p4d = p4d_offset(pgd, address);	  /* 在4级页面机制中，不做任何操作，直接返回pgd */
        if (p4d_none(*p4d) || unlikely(p4d_bad(*p4d)))
                goto out;

        pud = pud_offset(p4d, address);	  /* 返回address指向该进程的Page Upper Directory指针 */
        if (pud_none(*pud) || unlikely(pud_bad(*pud)))
                goto out;

        pmd = pmd_offset(pud, address);	  /* 返回address指向该进程的Page Middle Derectory指针 */
        /* ... */
        if (pmd_none(*pmd) || unlikely(pmd_bad(*pmd)))
                goto out;

        if (start && end) {
                *start = address & PAGE_MASK;
                *end = *start + PAGE_SIZE;
                mmu_notifier_invalidate_range_start(mm, *start, *end);
        }
        ptep = pte_offset_map_lock(mm, pmd, address, ptlp);   /* 返回address指向该进程的Page Table指针 */
	/* 判断Page Table是否保留在物理内存之中 */
        if (!pte_present(*ptep))
                goto unlock;
        *ptepp = ptep;
        *ptepp = ptep;
        return 0;
unlock:
        pte_unmap_unlock(ptep, *ptlp);
        if (start && end)
                mmu_notifier_invalidate_range_end(mm, *start, *end);
out:
        return -EINVAL;
}
```
+ 1 假如返回的目录项不存在，pgd_none()，pud_none和pmd_none 返回1.   
+ 2 pte_present 宏的值为 1 或 0，表示 _PAGE_PRESENT标志位。如果页表项不为 0，但标志位pte_present()的值为 0，则表示映射已经建立，但所映射的物理页面不在内存。   
## Page Table转为物理地址
```
int follow_phys(struct vm_area_struct *vma,
                unsigned long address, unsigned int flags,
                unsigned long *prot, resource_size_t *phys)
{
        /* ... */
	if (follow_pte(vma->vm_mm, address, &ptep, &ptl))
                goto out;
        pte = *ptep;
	/* ... */
        *phys = (resource_size_t)pte_pfn(pte) << PAGE_SHIFT;	/* Page Table转化为物理地址 */

        ret = 0;
unlock:
        pte_unmap_unlock(ptep, ptl);
out:
        return ret;
}
```
页表是一个元素为页表条目（Page Table Entry, PTE）的集合，每个虚拟页在页表中一个固定偏移量的位置上都有一个PTE. 所以两个虚拟地址是存在可能映射到同一个物理地址的.   

# pagemap_pmd_range
```
[   99.008467] [<c02f782c>] (pagemap_pmd_range) from [<c0270c7c>] (walk_pgd_range+0x108/0x184)
[   99.017758] [<c0270c7c>] (walk_pgd_range) from [<c0270e38>] (walk_page_range+0xe0/0x104)
[   99.026761] [<c0270e38>] (walk_page_range) from [<c02f7bc8>] (pagemap_read+0x1a8/0x2e0)
[   99.035672] [<c02f7bc8>] (pagemap_read) from [<c02905b8>] (__vfs_read+0x48/0x13c)
[   99.044003] [<c02905b8>] (__vfs_read) from [<c02913a0>] (vfs_read+0xa0/0x154)
[   99.051938] [<c02913a0>] (vfs_read) from [<c0292470>] (SyS_read+0x60/0xb0)
[   99.059596] [<c0292470>] (SyS_read) from [<c0107f40>] (ret_fast_syscall+0x0/0x48)

```

```
static pagemap_entry_t pte_to_pagemap_entry(struct pagemapread *pm,
                struct vm_area_struct *vma, unsigned long addr, pte_t pte)
{
        u64 frame = 0, flags = 0;
        struct page *page = NULL;

        if (pte_present(pte)) {
                if (pm->show_pfn)
                        frame = pte_pfn(pte);
                flags |= PM_PRESENT;
                page = vm_normal_page(vma, addr, pte);
                if (pte_soft_dirty(pte))
                        flags |= PM_SOFT_DIRTY;
        } else if (is_swap_pte(pte)) {
                swp_entry_t entry;
                if (pte_swp_soft_dirty(pte))
                        flags |= PM_SOFT_DIRTY;
                entry = pte_to_swp_entry(pte);
                if (pm->show_pfn)
                        frame = swp_type(entry) |
                                (swp_offset(entry) << MAX_SWAPFILES_SHIFT);
                flags |= PM_SWAP;
                if (is_migration_entry(entry))
                        page = migration_entry_to_page(entry);

                if (is_device_private_entry(entry))
                        page = device_private_entry_to_page(entry);
        }

        if (page && !PageAnon(page))
                flags |= PM_FILE;
        if (page && page_mapcount(page) == 1)
                flags |= PM_MMAP_EXCLUSIVE;
        if (vma->vm_flags & VM_SOFTDIRTY)
                flags |= PM_SOFT_DIRTY;

        return make_pme(frame, flags);
}
```

# references

[Linux 动态链接](http://0x4c43.cn/2018/0508/linux-dynamic-link/)   