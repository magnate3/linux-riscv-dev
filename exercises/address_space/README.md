
# insmod  vma_test1.ko 

```
       struct address_space_operations *ops=mapping->a_ops;
       if (ops){
           pr_info("mapping->a_ops: %p \n", ops);
       }
```

```
[root@centos7 address_space]# gcc mmap_fork.c -o mmap_fork
[root@centos7 address_space]# ./mmap_fork 
mmap: Success
resust addr : 0xffff9fba0000, and 9fba0000lx
integerSize addr : 0xffffd9be7e68, and d9be7e68lx
before wirte please findpage resust addr 

after wirte please findpage resust addr 
```
## echo 'findpage0xffff9fba0000' > /proc/mtest
```
[root@centos7 address_space]# rmmod  vma_test1.ko
[root@centos7 address_space]# insmod  vma_test1.ko 
[root@centos7 address_space]# echo 'findtask10284' > /proc/mtest
[root@centos7 address_space]# echo 'findpage0xffff9fba0000' > /proc/mtest
[root@centos7 address_space]# dmesg | tail -n 10
[25199.983260] page is file and compare rmap_walk_file
[25199.988119] mapping->a_ops: ffff0000088ded80 
[25199.992457] vma 0xffff9fba0000-0xffffafba0000 flag fb , vma task comm: mmap_fork and pid 10284 , d_iname : dev/zero 
[25200.002933] vma->vm_ops: ffff0000088df100 
[25200.007013] vma 0xffff9fba0000-0xffffafba0000 flag fb , vma task comm: mmap_fork and pid 10285 , d_iname : dev/zero 
[25200.017487] vma->vm_ops: ffff0000088df100 
[25200.021564]  page_mapcount(page) went negative! (2)
[25200.026422]  page->flags = 1fffff0000040038
[25200.030586]  page->count = 9
[25200.033452]  page->mapping = ffff805fd7282860
[root@centos7 address_space]# 
```

### mapping->a_ops: ffff0000088ded80 

```
[root@centos7 address_space]# cat /proc/kallsyms  | grep ffff0000088ded80
ffff0000088ded80 r shmem_aops
```

### vma->vm_ops: ffff0000088df100 

```
[root@centos7 address_space]# cat /proc/kallsyms  | grep ffff0000088df100 
ffff0000088df100 r shmem_vm_ops
[root@centos7 address_space]# 
```

## echo 'findpage0xffffd9be7e68' > /proc/mtest
```
[root@centos7 address_space]# echo 'findpage0xffffd9be7e68' > /proc/mtest
[root@centos7 address_space]# dmesg | tail -n 10
[25335.694250] page is anonoyous and compare  rmap_walk_anon 
[25335.699711] vma 0xffffd9bc0000-0xffffd9bf0000 flag 100173 , vma task comm: mmap_fork and pid 10284 , d_iname : no vm file 
[25335.710704] vma->vm_ops is null /////////////////////
[25335.713923] vma 0xffffd9bc0000-0xffffd9bf0000 flag 100173 , vma task comm: mmap_fork and pid 10285 , d_iname : no vm file 
[25335.724914] vma->vm_ops is null  /////////////////////
[25335.728128]  page_mapcount(page) went negative! (1)
[25335.732988]  page_mapcount(page) went negative! (1)
[25335.737844]  page->flags = 1fffff000004006c
[25335.742006]  page->count = 4
[25335.744876]  page->mapping = ffff805fcd61c2e9
[root@centos7 address_space]#
```


#  page_to_pgoff

```
__vma_address(struct page *page, struct vm_area_struct *vma)
{
    pgoff_t pgoff = page_to_pgoff(page);
    return vma->vm_start + ((pgoff - vma->vm_pgoff) << PAGE_SHIFT);
}
```

# insert_vm_struct
  
insert_vm_struct() 在线性区对象链表和内存描述符的红黑树中插入一个vm_area_struct结构。
这个函数使用两个参数：mm 指定进程内存描述符的地址，vma指定要插入的vm_area_struct对象的地址。其基本思路：

利用find_vma_links()寻找出将要插入的结点位置，其前驱结点和其父结点。
利用__vma_link_list()和__vma_link_rb()将结点分别插入链表和红黑树中，vma_link()是其前端函数。
注意在__vma_link_rb()利用vma_gap_update()更新vm_area_struct中rb_subtree_gap字段。
该字段在get_unmapped_area中已有介绍，加快了寻找满足条件的vma。
若线性区用于文件映射，那么利用__vma_link_file()处理。
线性区计数加一。

## __vma_link_file




#  page cache
page cache是通过vma->vm_file->f_mapping->page_tree来管理的，是一个radix tree，为address_space所有。也可以通过inode->i_mapping找到.
struct address_space *mapping = vma->vm_file->f_mapping

#flush_cache_page 


## vma_interval_tree_foreach(mpnt, &mapping->i_mmap, pgoff, pgoff)
  
```
static void flush_aliases(struct address_space *mapping, struct page *page)
{
	struct mm_struct *mm = current->active_mm;
	struct vm_area_struct *mpnt;
	pgoff_t pgoff;

	pgoff = page->index;

	flush_dcache_mmap_lock(mapping);
	vma_interval_tree_foreach(mpnt, &mapping->i_mmap, pgoff, pgoff) {
		unsigned long offset;

		if (mpnt->vm_mm != mm)
			continue;
		if (!(mpnt->vm_flags & VM_MAYSHARE))
			continue;

		offset = (pgoff - mpnt->vm_pgoff) << PAGE_SHIFT;
		flush_cache_page(mpnt, mpnt->vm_start + offset,
			page_to_pfn(page));
	}
	flush_dcache_mmap_unlock(mapping);
}
```

```
 arm64/include/asm/cacheflush.h
static inline void __flush_icache_all(void)
{
        asm("ic ialluis");
        dsb(ish);
}

static inline void flush_cache_page(struct vm_area_struct *vma,
                                    unsigned long user_addr, unsigned long pfn)
{
}
```

# references
[struct address_space_operations myfs](https://github.com/search?q=struct+address_space_operations+myfs&type=Code)