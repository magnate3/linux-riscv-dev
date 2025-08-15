
#  alloc_zeroed_user_highpage_movable

**gfp_t flags = GFP_HIGHUSER_MOVABLE | __GFP_ZERO;**

```
/*  arch/arm64/mm/fault.c
 * Used during anonymous page fault handling.
 */
struct page *alloc_zeroed_user_highpage_movable(struct vm_area_struct *vma,
                                                unsigned long vaddr)
{
        gfp_t flags = GFP_HIGHUSER_MOVABLE | __GFP_ZERO;

        /*
         * If the page is mapped with PROT_MTE, initialise the tags at the
         * point of allocation and page zeroing as this is usually faster than
         * separate DC ZVA and STGM.
         */
        if (vma->vm_flags & VM_MTE)
                flags |= __GFP_ZEROTAGS;

        return alloc_page_vma(flags, vma, vaddr);
}
```

##  __alloc_pages_nodemask

**enum zone_type high_zoneidx = gfp_zone(gfp_mask);**

```
/* 
 * This is the 'heart' of the zoned buddy allocator. 
 */  
struct page *  
__alloc_pages_nodemask(gfp_t gfp_mask, unsigned int order,  
            struct zonelist *zonelist, nodemask_t *nodemask)  
{  
    enum zone_type high_zoneidx = gfp_zone(gfp_mask);  
    struct zone *preferred_zone;  
    struct page *page;  
    int migratetype = allocflags_to_migratetype(gfp_mask);  
  
    gfp_mask &= gfp_allowed_mask;  
  
    lockdep_trace_alloc(gfp_mask);  
  
    might_sleep_if(gfp_mask & __GFP_WAIT);  
  
    if (should_fail_alloc_page(gfp_mask, order))  
        return NULL;  
  
    /* 
     * Check the zones suitable for the gfp_mask contain at least one 
     * valid zone. It's possible to have an empty zonelist as a result 
     * of GFP_THISNODE and a memoryless node 
     */  
    if (unlikely(!zonelist->_zonerefs->zone))  
        return NULL;  
  
    get_mems_allowed();  
    /* The preferred zone is used for statistics later */  
    first_zones_zonelist(zonelist, high_zoneidx,  
                nodemask ? : &cpuset_current_mems_allowed,  
                &preferred_zone);  
    if (!preferred_zone) {  
        put_mems_allowed();  
        return NULL;  
    }  
  
    /* First allocation attempt */  
    page = get_page_from_freelist(gfp_mask|__GFP_HARDWALL, nodemask, order,  
            zonelist, high_zoneidx, ALLOC_WMARK_LOW|ALLOC_CPUSET,  
            preferred_zone, migratetype);  
    if (unlikely(!page))  
        page = __alloc_pages_slowpath(gfp_mask, order,  
                zonelist, high_zoneidx, nodemask,  
                preferred_zone, migratetype);  
    put_mems_allowed();  
  
    trace_mm_page_alloc(page, order, gfp_mask, migratetype);  
    return page;  
}  
```



# PageHighMem

```
#define PageHighMem(__p) is_highmem_idx(page_zonenum(__p))
#else
PAGEFLAG_FALSE(HighMem)
#endif
```

```
static inline int is_highmem_idx(enum zone_type idx)
{
#ifdef CONFIG_HIGHMEM
        return (idx == ZONE_HIGHMEM ||
                (idx == ZONE_MOVABLE && movable_zone == ZONE_HIGHMEM));
#else
        return 0;
#endif
}
```

```c

static inline enum zone_type page_zonenum(const struct page *page)
{
        ASSERT_EXCLUSIVE_BITS(page->flags, ZONES_MASK << ZONES_PGSHIFT);
        return (page->flags >> ZONES_PGSHIFT) & ZONES_MASK;
}
```

***操作系统没有配置CONFIG_HIGHMEM***
```
[root@centos7 boot]# grep CONFIG_HIGHMEM  config-4.14.0-115.el7a.0.1.aarch64
[root@centos7 boot]# 

ubuntu@ubuntux86:/boot$ grep CONFIG_HIGHMEM config-5.13.0-30-generic 
ubuntu@ubuntux86:/boot$
```

## ZONE_NORMAL

```
gfp_zone(gfp) = ZONE_MOVABLE 
```

# insmod  zone_test1.ko

```
[root@centos7 vma]# dmesg | tail -n 15
[51606.801898] page is not  high 
[53823.666599] alloc GFP_KERNEL page 
[53823.669992] <0>alloc_pages Successfully!
[53823.673906] <0>the zone is NORMAL.////////////////
[53823.677294] GFP_HIGHUSER_MOVABLE page
[53823.680940] <0>alloc_pages Successfully!
[53823.684851] <0>the zone is NORMAL./////////////
[53823.688239] GFP_DMA page
[53823.690762] <0>alloc_pages Successfully!
[53823.694671] <0>the zone is DMA.
[53823.697798] GFP_HIGHMEM page   ///////////
[53823.700664] <0>alloc_pages Successfully!
[53823.704575] <0>the zone is NORMAL.///////////
[54435.472009] exit the module……mtest_exit 
[54604.452038] <0>exit!
```


# page_address
```

/**
 * page_address - get the mapped virtual address of a page
 * @page: &struct page to get the virtual address of
 *
 * Returns the page's virtual address.
 */ 
void *page_address(const struct page *page)
{   
    unsigned long flags;
    void *ret;
    struct page_address_slot *pas;
 
    if (!PageHighMem(page))
        return lowmem_page_address(page);//返回逻辑地址  
    
    pas = page_slot(page);//得到hash数组中的一个元素，里面包含了链表
    ret = NULL;
    spin_lock_irqsave(&pas->lock, flags);
    if (!list_empty(&pas->lh)) {//判空
        struct page_address_map *pam;
 
        list_for_each_entry(pam, &pas->lh, list) {//扫描链表，链表上存放的是物理页和虚拟页的映射关系
            if (pam->page == page) {//如果找到了物理页，
                ret = pam->virtual;
                goto done;
            }
        }
    }
done:
    spin_unlock_irqrestore(&pas->lock, flags);
    return ret;
}
```

# page_to_pfn
```
#define __page_to_pfn(pg)					\
({	const struct page *__pg = (pg);				\
	int __sec = page_to_section(__pg);			\
	(unsigned long)(__pg - __section_mem_map_addr(__nr_to_section(__sec)));	\
})
#define page_to_pfn __page_to_pfn
#ifndef page_to_virt
#define page_to_virt(x) __va(PFN_PHYS(page_to_pfn(x)))
include/linux/pfn.h:21:#define PFN_PHYS(x)      ((phys_addr_t)(x) << PAGE_SHIFT)
include/asm-generic/page.h:74:#define __va(x) ((void *)((unsigned long) (x)))
include/asm-generic/page.h:78:#define pfn_to_virt(pfn)  __va((pfn) << PAGE_SHIFT)
#endif
```

```
static void
mtest_write_val(unsigned long addr, unsigned long val)
{
        struct vm_area_struct *vma;
        struct mm_struct *mm = task->mm;
        struct page *page;
        unsigned long kernel_addr;
        if (!task) {
            pr_info("The process is not exist \n");
            return ;
        }
        printk("mtest_write_val\n");
        down_read(&mm->mmap_sem);
        vma = find_vma(mm, addr);
        if (vma && addr >= vma->vm_start && (addr + sizeof(val)) < vma->vm_end) {
                if (!(vma->vm_flags & VM_WRITE)) {
                        printk("vma is not writable for 0x%lx\n", addr);
                        goto out;
                }
                page = my_follow_page(vma, addr);
                if (!page) {
                        printk("page not found  for 0x%lx\n", addr);
                        goto out;
                }
                kernel_addr = (unsigned long)page_address(page);
                kernel_addr += (addr&~PAGE_MASK);
                printk("write 0x%lx to address 0x%lx\n", val, kernel_addr);
                *(unsigned long *)kernel_addr = val;
                put_page(page);
        } else {
                printk("no vma found for %lx\n", addr);
        }
out:0
        up_read(&mm->mmap_sem);
}
```



![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/alloc_zeroed_user_highpage/pic/write.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/alloc_zeroed_user_highpage/pic/read.png)

**读的值是55，是下述命令写的**

```
            for(i=0;i <4096 *10; i++)
                    p[4096 * i] = 0x55;
```

## user 

```
[root@centos7 vma]# ./mmap_test2 
before mmap ->please exec: free -m


p addr:  0xffff777f0000 
after mmap ->please exec: free -m

will read....
after mmap ->please exec: free -m

will write....


after write ->please exec: free -m

please execute 'writeval cmmd 


read kernel :  aaaaaaaaaaaa 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/alloc_zeroed_user_highpage/pic/write2.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/alloc_zeroed_user_highpage/pic/read2.png)


# spare memory model
```
[root@centos7 boot]# grep CONFIG_SPARSEMEM config-4.14.0-115.el7a.0.1.aarch64 
CONFIG_SPARSEMEM_MANUAL=y
CONFIG_SPARSEMEM=y
CONFIG_SPARSEMEM_EXTREME=y
CONFIG_SPARSEMEM_VMEMMAP_ENABLE=y
CONFIG_SPARSEMEM_VMEMMAP=y


ubuntu@ubuntux86:/boot$ grep CONFIG_SPARSEMEM config-5.13.0-30-generic 
CONFIG_SPARSEMEM_MANUAL=y
CONFIG_SPARSEMEM=y
CONFIG_SPARSEMEM_EXTREME=y
CONFIG_SPARSEMEM_VMEMMAP_ENABLE=y
CONFIG_SPARSEMEM_VMEMMAP=y
ubuntu@ubuntux86:/boot$ 
```

# insmod alloc_page_test.ko 

```
[69075.604959] sizeof(struct page): 64
[69075.604962] virt : 0xffffa03f3d540000, phy:0x203f3d540000, page:0xffff7fe80fcf5500, pfn: 541015380
[69075.617358] virt : 0xffffa03f3d550000, phy:0x203f3d550000, page:0xffff7fe80fcf5540, pfn: 541015381
[69075.626282] virt : 0xffffa03f3d560000, phy:0x203f3d560000, page:0xffff7fe80fcf5580, pfn: 541015382
[69075.635206] virt : 0xffffa03f3d570000, phy:0x203f3d570000, page:0xffff7fe80fcf55c0, pfn: 541015383
[root@centos7 alloc_page]# 
```

# zone->zone_mem_map