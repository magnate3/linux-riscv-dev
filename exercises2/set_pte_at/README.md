This kex module implements mmap and includes a test rig to exercise it.
When the test rig invokes mmap on the device, kex uses alloc_page 
to get as many pages as the rig requested. It uses vm_insert_page on
each page to map it into rig's address space. These pages, while 
physically disjoint, appear as a single contiguous virtual memory area
(VMA) inside the rig address space. The VMA is visible using pmap:

```
root@ubuntux86:# insmod hybridmem_test.ko
root@ubuntux86:# grep blkext -rn *
root@ubuntux86:# cat /proc/devices | grep kex
510 kex
259 blkext
root@ubuntux86:# mknod devKex c 510 0
chmod a+rw devKex
root@ubuntux86:# ./mmap_test devKex 1
pid: 4475
press a key to exercise memory write 0x7f6dcd411000

press a key to terminate

root@ubuntux86:# 
```

Leave the rig waiting. In another terminal run pmap. See the dev VMA:

```
root@ubuntux86:# pmap -x 4475
4475:   ./mmap_test devKex 1
Address           Kbytes     RSS   Dirty Mode  Mapping
000055a610c2c000       4       4       4 r---- mmap_test
000055a610c2d000       4       4       4 r-x-- mmap_test
000055a610c2e000       4       4       4 r---- mmap_test
000055a610c2f000       4       4       4 r---- mmap_test
000055a610c30000       4       4       4 rw--- mmap_test
00007f6dcd1da000     136     136       0 r---- libc-2.31.so
00007f6dcd1fc000    1504     836       0 r-x-- libc-2.31.so
00007f6dcd374000     312     168       0 r---- libc-2.31.so
00007f6dcd3c2000      16      16      16 r---- libc-2.31.so
00007f6dcd3c6000       8       8       8 rw--- libc-2.31.so
00007f6dcd3c8000      24      20      20 rw---   [ anon ]
00007f6dcd3e5000       4       4       0 r---- ld-2.31.so
00007f6dcd3e6000     140     140       0 r-x-- ld-2.31.so
00007f6dcd409000      32      32       0 r---- ld-2.31.so
00007f6dcd411000       4       4       0 rw-s- devKex
00007f6dcd412000       4       4       4 r---- ld-2.31.so
00007f6dcd413000       4       4       4 rw--- ld-2.31.so
00007f6dcd414000       4       4       4 rw---   [ anon ]
00007fff2f145000     132      20      20 rw---   [ stack ]
00007fff2f1aa000      16       0       0 r----   [ anon ]
00007fff2f1ae000       8       4       0 r-x--   [ anon ]
ffffffffff600000       4       0       0 --x--   [ anon ]
---------------- ------- ------- ------- 
total kB            2372    1420      96
root@ubuntux86:# 
```
***00007f6dcd411000       4       4       0 rw-s- devKex***    
It's 4kb because rig asked for one page (the final argument), which is 4kb.
Now we can ask rig to write to this memory. In the original terminal, press
enter.  Then re-run pmap in the second terminal:

```
Address           Kbytes     RSS   Dirty Mode  Mapping
00007f6dcd411000       4       4       0 rw-s- devKex
```

Notice that the Dirty column now shows the mapped page is Dirty.  (TODO: is
this an MMU flag set on write to the page?). Press enter to terminate rig.

Clean up:

    % rm devKex
    % sudo rmmod hybridmem_test.ko
  
 kernel mesg    
```
root@ubuntux86:# dmesg | tail -n 10                
[11781.322088] vma->vm_end 140512583536640 vm_start 140512583532544 len 4096 pages 1 vm_pgoff 0
[11781.322101] inserted page 0 at 00000000578fb2ad
[11781.322106] completed inserting 1 pages
[11827.371169] vma->vm_end 140259426410496 vm_start 140259426406400 len 4096 pages 1 vm_pgoff 0
[11827.371182] inserted page 0 at 000000008d3af85a
[11827.371188] mapping->a_ops 00000000bc065379, name devKex
[11827.371192] completed inserting 1 pages
root@ubuntux86:# 
```  

# vm_insert_page -->  …… set_pte_at

+ 1 vm_insert_page --> insert_page -->  insert_page_into_pte_locked  --> set_pte_at   
+ 2 since vm_insert_page incremented its  refcount. that way call __free_page(page) to decrease refcount   
```
static int insert_page_into_pte_locked(struct mm_struct *mm, pte_t *pte,
                        unsigned long addr, struct page *page, pgprot_t prot)
{
        if (!pte_none(*pte))
                return -EBUSY;
        /* Ok, finally just insert the thing.. */
        get_page(page);
        inc_mm_counter_fast(mm, mm_counter_file(page));
        page_add_file_rmap(page, false);
        set_pte_at(mm, addr, pte, mk_pte(page, prot));
        return 0;
}
```
+ 1 page_add_file_rmap
在建立反向映射时，需要对匿名页和基于文件映射的页分别处理，这是因为管理这两种选项的数据结构不同。对匿名页简历反向映射的函数是page_add_anon_rmap，对基于文件映射的页的函数是page_add_file_rmap。    
page_add_anon_rmap 建立基于文件映射的页，对应的文件是***devKex***    
```
[11827.371188] mapping->a_ops 00000000bc065379, name devKex
```

page mapping 这个字段：
1.当page->mapping != NULL，并且bit[0] == 0，该page属于页缓存或文件映射，mapping指向inode address_space。     
2.当page->mapping != NULL，并且bit[0] ！= 0，该page属于匿名映射，且设置了PAGE_MAPPING_ANON位（bit[0] ）被，page->mapping指向struct anon_vma对象。     
```
static __always_inline bool folio_test_anon(struct folio *folio)
{
        return ((unsigned long)folio->mapping & PAGE_MAPPING_ANON) != 0;
}

static __always_inline bool PageAnon(struct page *page)
{
        return folio_test_anon(page_folio(page));
}

```
anon没有file   
```
int address_space(struct vm_area_struct *vma)
{
     //struct vm_area_struct *vma = vmf->vma;
     if (!vma->vm_file) {
         printk(KERN_INFO"anon vma %p\n", vma->anon_vma);
         return 0;
     }
     struct inode *inode = file_inode(vma->vm_file);
     struct address_space *mapping = inode->i_mapping;
     const char *name = vma->vm_file->f_path.dentry->d_iname;
     printk(KERN_INFO"mapping->a_ops %p, name %s\n", mapping->a_ops, name);
     return 0;
}
```

## vm_file 成员

vm_area_struct 结构体 中的 vm_file 成员 是 " 内存映射 “ 中的 ” 文件映射 " 类型中 被映射的 文件 , 如果是 " 匿名映射 " 类型的 " 内存映射 " , 该成员为 NULL ;   

# munmap --> free_page

```Text
在 munmap() 函数内部，会首先找到要解除映射关系的虚拟内存区域所对应的页表项，并将其从页表中删除。接着，munmap() 会检查与该区域相关联的物理页面是否需要被释放。如果某个物理页面不再被任何进程使用，则 munmap() 将调用 free_page() 等函数释放该页面占用的内存资源。

unmap_region() 是在 munmap() 函数内部被调用的一个子函数。它主要负责遍历指定区域中所有页表项，并调用相应的函数来处理每个页表项所对应的物理页面。具体而言，unmap_region() 会逐一检查当前页表项是否有效，如果是，则将该页表项对应的物理页面解除映射关系，并更新相关的内存管理数据结构
```

#  do_set_pte  -->  set_pte_at
do_set_pte() 函数将缺页异常处理流程中获取的物理地址写入页表项，完成物理地址和报异常的虚拟地址之间的连接：

```
// mm/memory.c: 3975
void do_set_pte(struct vm_fault *vmf, struct page *page, unsigned long addr)
{
	...
	/* 如果是写异常，将页表项的脏位置位，同时将写权限 W 位也置位 */
	if (write)
		entry = maybe_mkwrite(pte_mkdirty(entry), vma);
	/* copy-on-write page */
	/*
* 如果是私有写文件映射，由于其已经独立了，不再会影响文件页，所以视为
* 私有匿名页管理，将其加入私有匿名页的反射机制管理结构中，同时也将该
* 页加入 LRU 不活跃链表中，第一次访问不能证明其经常会被访问，所以暂且
* 放入不活跃链表。
* 如果是共享的文件页，将其加入文件页反射机制管理结构中。
* 上述两种情况都会调用 inc_mm_counter_fast() 增加该虚拟地址空间的引用
* 次数
*/
	if (write && !(vma->vm_flags & VM_SHARED)) {
		inc_mm_counter_fast(vma->vm_mm, MM_ANONPAGES);
		page_add_new_anon_rmap(page, vma, addr, false);
		lru_cache_add_inactive_or_unevictable(page, vma);
	} else {
		inc_mm_counter_fast(vma->vm_mm, mm_counter_file(page));
		page_add_file_rmap(page, false);
	}
	/* 将页对应的物理地址写入页表项，完成该缺页虚拟地址到物理页的最终映射 */
	set_pte_at(vma->vm_mm, addr, vmf->pte, entry);
}
```
do_set_pte() 流程总结如下：

+ 如果是写文件映射的异常，将页表项的脏位置位，同时将写权限 W 位也置位。

+ 将新获取的页面加入对应的管理结构中：   

如果是私有写文件映射，由于其已经独立了，不会影响文件页，所以视为私有匿名页管理，将其加入私有匿名页的反射机制管理结构中，同时也将该页加入 LRU 不活跃链表中，因为第一次访问不能证明其会经常被访问，所以暂且放入不活跃链表。   

如果是共享文件页，将其加入文件页反射机制管理结构中。   

上述两种情况都会调用 inc_mm_counter_fast() 增加该虚拟地址空间的引用次数。   

+ 最后调用 set_pte_at() 将页对应的物理地址写入页表项，完成该缺页的虚拟地址到物理页的最终映射。   

