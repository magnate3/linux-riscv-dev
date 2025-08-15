---
title: page
---







内存管理基本上是以数据结构`struct page`展开的，本文将详细围绕`struct page`展开。


## `struct page`结构体


`Linux`内核内存管理的实现是以`struct page`为核心，实际上每个物理页面都需要一个`struct page`数据结构来描述，因此为了降低成本，该结构中大量使用了`C`语言的联合体`Union`来优化其大小。

 


我们可以看出，一个`struct page`结构体大小为`64`字节，而一般情况下系统的页大小为`4096`字节（`4k`）。所以，一个系统上如果有`4GB`得内存，`struct page`结构体就需要`64MB`的内存，占比大约`1.5%`。

## `flags`标志成员

`flags`成员是页面标志位的集合，标志位是内存管理非常重要的部分，目前系统上主要有如下标志位：

```c
enum pageflags {
        PG_locked,              /* Page is locked. Don't touch. */
        PG_error,
        PG_referenced,
        PG_uptodate,
        PG_dirty,
        PG_lru,
        PG_active,
        PG_slab,
        PG_owner_priv_1,        /* Owner use. If pagecache, fs may use*/
        PG_arch_1,
        PG_reserved,
        PG_private,             /* If pagecache, has fs-private data */
        PG_private_2,           /* If pagecache, has fs aux data */
        PG_writeback,           /* Page is under writeback */
#ifdef CONFIG_PAGEFLAGS_EXTENDED
        PG_head,                /* A head page */
        PG_tail,                /* A tail page */
#else
        PG_compound,            /* A compound page */
#endif
        PG_swapcache,           /* Swap page: swp_entry_t in private */
        PG_mappedtodisk,        /* Has blocks allocated on-disk */
        PG_reclaim,             /* To be reclaimed asap */
        PG_swapbacked,          /* Page is backed by RAM/swap */
        PG_unevictable,         /* Page is "unevictable"  */
#ifdef CONFIG_MMU
        PG_mlocked,             /* Page is vma mlocked */
#endif
#ifdef CONFIG_ARCH_USES_PG_UNCACHED
        PG_uncached,            /* Page has been mapped as uncached */
#endif
#ifdef CONFIG_MEMORY_FAILURE
        PG_hwpoison,            /* hardware poisoned page. Don't touch */
#endif
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
        PG_compound_lock,
#endif
        __NR_PAGEFLAGS,

        /* Filesystems */
        PG_checked = PG_owner_priv_1,

        /* Two page bits are conscripted by FS-Cache to maintain local caching
         * state.  These bits are set on pages belonging to the netfs's inodes
         * when those inodes are being locally cached.
         */
        PG_fscache = PG_private_2,      /* page backed by cache */

        /* XEN */
        PG_pinned = PG_owner_priv_1,
        PG_savepinned = PG_dirty,

        /* SLOB */
        PG_slob_free = PG_private,
};
```

在实际的`x86-64`系统上，有如下`flags`:

```
crash> kmem -g
PAGE-FLAG        BIT  VALUE
PG_locked          0  0000001
PG_error           1  0000002
PG_referenced      2  0000004
PG_uptodate        3  0000008
PG_dirty           4  0000010
PG_lru             5  0000020
PG_active          6  0000040
PG_slab            7  0000080
PG_owner_priv_1    8  0000100
PG_arch_1          9  0000200
PG_reserved       10  0000400
PG_private        11  0000800
PG_private_2      12  0001000
PG_writeback      13  0002000
PG_head           14  0004000
PG_tail           15  0008000
PG_swapcache      16  0010000
PG_mappedtodisk   17  0020000
PG_reclaim        18  0040000
PG_swapbacked     19  0080000
PG_unevictable    20  0100000
PG_mlocked        21  0200000
PG_uncached       22  0400000
PG_hwpoison       23  0800000
PG_compound_lock  24  1000000
PG_checked         8  0000100
PG_fscache        12  0001000
PG_pinned          8  0000100
PG_savepinned      4  0000010
PG_slob_free      11  0000800
```


内核定义了一些标准宏，用于检查页面是否设置了某个特定的标志位或者用于设置、清除某个标志位，这些宏的名称有一定的模式，具体如下：

* `PageXXX()`用于检查页面是否设置了`PG_XXX`标志位；
* `SetPageXXX()`设置页面的`PG_XXX`标志位；
* `ClearPageXXX()`用于无条件的清除`PG_XXX`标志位。

```c
#define TESTPAGEFLAG(uname, lname)                                      \
static inline int Page##uname(const struct page *page)                  \
                        { return test_bit(PG_##lname, &page->flags); }

#define SETPAGEFLAG(uname, lname)                                       \
static inline void SetPage##uname(struct page *page)                    \
                        { set_bit(PG_##lname, &page->flags); }

#define CLEARPAGEFLAG(uname, lname)                                     \
static inline void ClearPage##uname(struct page *page)                  \
                        { clear_bit(PG_##lname, &page->flags); }

```

`flags`成员除了存放以上重要的标志位以外，还有另外一个很重要的作用，就是存放`section`编号、`node`结点编号、`zone`结点编号和`LAST_CPUPID`等。具体存放内容与内核配置有关。

 




*  `0-24`位用于存放页面标准位；
*  `25-29`位保留；
*  `30-51`位用于存放`last_cpupid`；
*  `52-53`位用于存放`zone id`；
*  `54-63`位用于存放`node id`。

## `mapping`成员

`struct page`中，`mapping`成员表示页面所指向的地址空间。内核中的地址空间通常有两个不通的地址空间，一个用于文件映射页面，例如在读取文件时，地址空间用于将文件内容数据与装载数据的存储介质区关联起来。另一个用于匿名映射。内核使用了一个简单直接的方式实现了『一个指针，两种用途』，`mapping`指针地址的最后两位用于判断是否指匿名映射或KSM页面的地址空间，如果是匿名页面，那么`mapping`指向匿名页面的地址空间数据结构`struct anon_vma`。

```c
#define PAGE_MAPPING_ANON       1
#define PAGE_MAPPING_KSM        2
#define PAGE_MAPPING_FLAGS      (PAGE_MAPPING_ANON | PAGE_MAPPING_KSM)

static inline int PageAnon(struct page *page)
{
        return ((unsigned long)page->mapping & PAGE_MAPPING_ANON) != 0;
}
```

## `_count`和`_mapcount`成员


`_count`和`_mapcount`成员是`struct page`结构中的两个非常重要的引用计数，并且都是`atomic_t`类型的变量。

其中，`_count`表示内核中引用该页面的次数：

*  `_count`的值为0时，表示该页面为空闲或者即将要被释放的页面；
*  `_count`的值大于0时，表示该页面已经被分配且内核正在使用，暂时不会释放。

内核中常用的加减`_count`引用计数的`API`为：`get_page` 和` put_page`，此外，内核中还有一对常用的变种宏：

```
#define page_cache_get(page)            get_page(page) 
#define page_cache_release(page)        put_page(page)
```

其中，`_mapcount`表示这个页面被进程映射的个数，即已经映射了多少个用于`pte`页表：

*  `_mapcount == -1`表示没有pte映射到页面；
*  `_mapcount == 0`表示只有父进程映射了页面；
*  `_mapcount > 0`表示除了父进程外还有其他进程映射了这个页面。


## 页面锁`PG_locked`


还记得前面提到的标志位中有一个叫做`PG_locked`，内核常用这个标志位来设置一个页面锁。

*  `lock_page()`函数用于申请页面锁，如果页面锁被其它进程占用了，就会睡眠等待；
*   `trylock_page()`函数用于去尝试申请页面锁，如果`PG_locked` 已经置位了，该函数返回`false`，说明有其他进程已经锁住了该页面，返回`true`表示获取页面锁成功。


```c

/**                                                                                                                                      
 * __lock_page - get a lock on the page, assuming we need to sleep to get it                                                             
 * @page: the page to lock                                                                                                               
 */                                                                                                                                      
void __lock_page(struct page *page)                                                                                                                                               
{                                                                                                                                        
        DEFINE_WAIT_BIT(wait, &page->flags, PG_locked);                                                                                  
                                                                                                                                         
        __wait_on_bit_lock(page_waitqueue(page), &wait, bit_wait_io,                                                                     
                                                        TASK_UNINTERRUPTIBLE);                                                           
}                                                                                                                                        
EXPORT_SYMBOL(__lock_page);

/* 
 * lock_page may only be called if we have the page's inode pinned.               
 */
static inline void lock_page(struct page *page) 
{
        might_sleep(); 
        if (!trylock_page(page)) 
                __lock_page(page);  
}


static inline int trylock_page(struct page *page) 
{                                                   
        return (likely(!test_and_set_bit_lock(PG_locked, &page->flags)));
}
```

*  `unlock_page()`函数用于释放页面锁，并唤醒等待页面锁的进程。

```c
/**                                                                                                                                      
 * unlock_page - unlock a locked page                                                                                                    
 * @page: the page                                                                                                                       
 *                                                                                                                                       
 * Unlocks the page and wakes up sleepers in ___wait_on_page_locked().                                                                   
 * Also wakes sleepers in wait_on_page_writeback() because the wakeup                                                                    
 * mechananism between PageLocked pages and PageWriteback pages is shared.                                                               
 * But that's OK - sleepers in wait_on_page_writeback() just go back to sleep.                                                           
 *                                                                                                                                       
 * The mb is necessary to enforce ordering between the clear_bit and the read                                                            
 * of the waitqueue (to avoid SMP races with a parallel wait_on_page_locked()).                                                          
 */                                                                                                                                      
void unlock_page(struct page *page)                                                                                                      
{                                                                                                                                        
        VM_BUG_ON_PAGE(!PageLocked(page), page);                                                                                         
        clear_bit_unlock(PG_locked, &page->flags);                                                                                       
        smp_mb__after_clear_bit();                                                                                                                                                
        wake_up_page(page, PG_locked);                                                                                                   
}                                                                                                                                        
EXPORT_SYMBOL(unlock_page);                                                                                                              
                             
```

## `struct page`数组

我们知道内核里每一个物理页面都有一个`struct page`结构体描述，这些`struct page`数组存放到哪里呢？

下面我们分析`x86-64`情况下，`struct page`数组的情况，现在的服务器使用的都是`NUMA`架构，一般有两个`numa node`，`centos7`的内核内存模型为`Sparse Memory Model`，所以，内存被划分为很多个`mem_section`，每个`mem_section`的大小为`128MB`。

![enter description here][3]

如上图所示，通过`crash`命令`kmem -n`我们可以看到，该服务器有两个`NUMA NODE`, node0 包含三个zone（DMA,DMA32,Normal），node1包含1个zone（Normal）。同时也展示了`mem_section`的情况。

`struct mem_section`的定义如下：

```
crash> struct -o mem_section
struct mem_section {
   [0] unsigned long section_mem_map;
   [8] unsigned long *pageblock_flags;
  [16] struct page_cgroup *page_cgroup;
  [24] unsigned long pad;
}
SIZE: 32
```
其中`section_mem_map`成员指向了`struct page`数组。我们看一下`mem_section[0][0]`这个`mem_section`的`section_mem_map`的值：

```
crash> p -x  mem_section[0][0]
$6 = {
  section_mem_map = 0xffffea0000000003, 
  pageblock_flags = 0xffff88107ffd0040, 
  page_cgroup = 0xffff88017fc80000, 
  pad = 0x0
}
```

说明`0xffffea0000000000`为第一个`struct page`数据结构的地址，地址的低两位用作其它用途。


通过`crash`命令`kmem -p`我们可以验证一下上面的分析是否正确：

```
crash> kmem -p
      PAGE         PHYSICAL      MAPPING       INDEX CNT FLAGS
ffffea0000000000          0                0        0  0 400 reserved
ffffea0000000040       1000                0        0  1 fffff00000400 reserved
ffffea0000000080       2000                0        0  1 fffff00000400 reserved
ffffea00000000c0       3000                0        0  1 fffff00000400 reserved
ffffea0000000100       4000                0        0  1 fffff00000400 reserved
ffffea0000000140       5000                0        0  1 fffff00000400 reserved
ffffea0000000180       6000                0        0  1 fffff00000400 reserved
ffffea00000001c0       7000                0        0  1 fffff00000400 reserved
ffffea0000000200       8000                0        0  1 fffff00000400 reserved
ffffea0000000240       9000                0        0  1 fffff00000400 reserved
ffffea0000000280       a000                0        0  1 fffff00000400 reserved
ffffea00000002c0       b000                0        0  1 fffff00000400 reserved
ffffea0000000300       c000                0        0  1 fffff00000400 reserved
```

可以看出，第一个页面的地址为：`ffffea0000000000`。


此外，我们通过`crash`命令`kmem -p`有如下片段：
```
ffffea00000846c0    211b000                0        1  0 1fffff00000000
ffffea0000084700    211c000                0 ffff88000211c600  1 1fffff00004080 slab,head
ffffea0000084740    211d000                0        1  0 1fffff00008000 tail
```

`ffffea0000084700` 描述的页面的flags为：`slab,head`，我们可以进行验证一下：

```
crash> page.flags -x ffffea0000084700
  flags = 0x1fffff00004080
crash> kmem -g 0x1fffff00004080
FLAGS: 1fffff00004080
  PAGE-FLAG        BIT  VALUE
  PG_slab            7  0000080
  PG_head           14  0004000
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/huge_page/memory-node.png)


#   mount -t hugetlbfs nodev /mnt/huge

```
[root@centos7 test1]# echo 64 > /sys/kernel/mm/hugepages/hugepages-524288kB/nr_hugepages
[root@centos7 test1]# mount -t hugetlbfs nodev /mnt/huge
[root@centos7 test1]#  cat /proc/meminfo  | grep -i huge
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
HugePages_Total:      64
HugePages_Free:       64
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:     524288 kB         //512M=0x20000000    29bit


```



```
[root@centos7 test1]# getconf -a | grep -i page
PAGESIZE                           65536
PAGE_SIZE                          65536
_AVPHYS_PAGES                      7651838
_PHYS_PAGES                        8365865
[root@centos7 test1]# 
```


```
[54237.909626] pgdir_SHIFT = 42
[54237.912498] PAGE_OFFSET = 0xffff800000000000
[54237.916760] P4D_SHIFT = 42
[54237.919456] PUD_SHIFT = 42
[54237.922151] PMD_SHIFT = 29
[54237.924852] PAGE_SHIFT = 16
[54237.927635] PTRS_PER_PGD = 64
[54237.930589] PTRS_PER_P4D = 1
[54237.933461] PTRS_PER_PUD = 1
[54237.936328] PTRS_PER_PMD = 8192
[54237.939456] PTRS_PER_PTE = 8192
[54237.942582] PAGE_MASK = 0xffffffffffff0000   ///16 bits
[54237.946664] PUD_MASK = 0xfffffc0000000000    ///42 bits
[54237.950656] PMD_MASK= 0xffffffffe0000000     ///20 bits
```

# ./test2 

```
[root@centos7 test1]# ./test2 
Returned address is 0x400000000000
First hex is 0
First hex is 3020100
Returned address is 0x400180000000
First hex is 0
First hex is 3020100
munmap addr1 0x400000000000, addr2 0x400180000000 , stack 0xffffd94f7d9c 
input char 

```

```
[root@centos7 test1]# dmesg | tail -n 60
[137270.013814] PTRS_PER_PTE = 8192
[137270.017027] PAGE_MASK = 0xffffffffffff0000
[137270.021191] PUD_MASK = 0xfffffc0000000000
[137270.025272] PMD_MASK= 0xffffffffe0000000
[137270.029270] create the filename mtest mtest_init sucess  
[137304.788320] mtest_write  ………..  
[137304.792151] The process pid 41323 
[137304.795638] the find_vpid result's count is: 8
[137304.800151] the find_vpid result's level is: 0
[137304.804669] The process is "test2" (pid 41323)
[137341.755667] mtest_write  ………..  
[137341.759495]   ------------------------------
[137341.763852]   virtual user addr: 0000400000000000
[137341.768621]   page: ffff7ff000000000
[137341.772275] vm flag == VM_HUGETLB is true 
[137341.776441]   pgd: ffffa05e6e7dfa80 (0000205fc3a40003) 
[137341.776442]   p4d: ffffa05e6e7dfa80 (0000205fc3a40003) 
[137341.781734]   pud: ffffa05e6e7dfa80 (0000205fc3a40003) 
[137341.787023] pmd_huge is true 
[137341.795359]   pmd: ffffa05fc3a40000 (00e8205c60000f51) 
[137341.795360] pmd_large(*pmd): 1, pmd_present(*pmd) : 1 
[137341.805854]   p4d_page: ffff7fe817f0e900
[137341.809847]   pud_page: ffff7fe817f0e900
[137341.813844]   pmd_page: ffff7fe817180000
[137341.817837]   physical addr: 0000205c60000000
[137341.822266]   page addr: 0000205c60000000
[137341.826344]   ------------------------------
[137373.923122] mtest_write  ………..  
[137373.926949]   ------------------------------
[137373.931297]   virtual user addr: 0000400180000000
[137373.936067]   page: ffff7ff000600000
[137373.939714] vm flag == VM_HUGETLB is true 
[137373.943883]   pgd: ffffa05e6e7dfa80 (0000205fc3a40003) 
[137373.943885]   p4d: ffffa05e6e7dfa80 (0000205fc3a40003) 
[137373.949177]   pud: ffffa05e6e7dfa80 (0000205fc3a40003) 
[137373.954468] pmd_huge is true 
[137373.962803]   pmd: ffffa05fc3a40060 (00e8205de0000f51) 
[137373.962805] pmd_large(*pmd): 1, pmd_present(*pmd) : 1 
[137373.973297]   p4d_page: ffff7fe817f0e900
[137373.977290]   pud_page: ffff7fe817f0e900
[137373.981288]   pmd_page: ffff7fe817780000
[137373.985281]   physical addr: 0000205de0000000
[137373.989703]   page addr: 0000205de0000000
[137373.993784]   ------------------------------
[137485.427325] mtest_write  ………..  
[137485.431158]   ------------------------------
[137485.435497]   virtual user addr: 0000ffffd94f7d9c
[137485.440273]   page: ffff7ffffff653c0
[137485.443921]   pgd: ffffa05e6e7dfbf8 (0000205fdc7c0003) 
[137485.443922]   p4d: ffffa05e6e7dfbf8 (0000205fdc7c0003) 
[137485.449208]   pud: ffffa05e6e7dfbf8 (0000205fdc7c0003) 
[137485.454500]   pmd: ffffa05fdc7cfff0 (0000205e6db50003) 
[137485.459792]   pte: ffffa05e6db5ca78 (00e8205f856f0f53) 
[137485.465080]   p4d_page: ffff7fe817f71f00
[137485.474366]   pud_page: ffff7fe817f71f00
[137485.478358]   pmd_page: ffff7fe8179b6d40
[137485.482355]   pte_page: ffff7fe817e15bc0
[137485.486347]   physical addr: 0000205f856f7d9c
[137485.490775]   page addr: 0000205f856f0000
[137485.494852]   ------------------------------
[root@centos7 test1]# 
```


## pmd_huge is true 




```
[root@centos7 test1]# ps -elf | grep test2
0 R root      41323  39424 99  80   0 - 98342 -      21:28 pts/0    00:00:24 ./test2
0 S root      41328  41007  0  80   0 -  1729 pipe_w 21:28 pts/1    00:00:00 grep --color=auto test2
[root@centos7 test1]# echo 'findtask41323' > /proc/mtest 
[root@centos7 test1]# echo 'testtlb0x400000000000' > /proc/mtest 
[root@centos7 test1]# dmesg | tail -n 10
[137341.781734]   pud: ffffa05e6e7dfa80 (0000205fc3a40003) 
[137341.787023] pmd_huge is true 
[137341.795359]   pmd: ffffa05fc3a40000 (00e8205c60000f51) 
[137341.795360] pmd_large(*pmd): 1, pmd_present(*pmd) : 1 
[137341.805854]   p4d_page: ffff7fe817f0e900
[137341.809847]   pud_page: ffff7fe817f0e900
[137341.813844]   pmd_page: ffff7fe817180000
[137341.817837]   physical addr: 0000205c60000000
[137341.822266]   page addr: 0000205c60000000
[137341.826344]   ------------------------------
```

```
[root@centos7 test1]# echo 'testtlb0x400180000000' > /proc/mtest 
[root@centos7 test1]# dmesg | tail -n 10
[137373.949177]   pud: ffffa05e6e7dfa80 (0000205fc3a40003) 
[137373.954468] pmd_huge is true 
[137373.962803]   pmd: ffffa05fc3a40060 (00e8205de0000f51) 
[137373.962805] pmd_large(*pmd): 1, pmd_present(*pmd) : 1 
[137373.973297]   p4d_page: ffff7fe817f0e900
[137373.977290]   pud_page: ffff7fe817f0e900
[137373.981288]   pmd_page: ffff7fe817780000
[137373.985281]   physical addr: 0000205de0000000
[137373.989703]   page addr: 0000205de0000000
[137373.993784]   ------------------------------
[root@centos7 test1]# 
```
 
 
## pmd_huge is false
 
### echo 'testtlb0xffffd94f7d9c' > /proc/mtest 
 
```
[root@centos7 test1]# echo 'testtlb0xffffd94f7d9c' > /proc/mtest 
[root@centos7 test1]# dmesg | tail -n 10
[137485.449208]   pud: ffffa05e6e7dfbf8 (0000205fdc7c0003) 
[137485.454500]   pmd: ffffa05fdc7cfff0 (0000205e6db50003) 
[137485.459792]   pte: ffffa05e6db5ca78 (00e8205f856f0f53) 
[137485.465080]   p4d_page: ffff7fe817f71f00
[137485.474366]   pud_page: ffff7fe817f71f00
[137485.478358]   pmd_page: ffff7fe8179b6d40
[137485.482355]   pte_page: ffff7fe817e15bc0
[137485.486347]   physical addr: 0000205f856f7d9c
[137485.490775]   page addr: 0000205f856f0000
[137485.494852]   ------------------------------
[root@centos7 test1]# 
```