
# struct page

```
 struct page {
	unsigned long flags;		/* Atomic flags, some possibly
					 * updated asynchronously */
	union {
		struct address_space *mapping;	/* If low bit clear, points to
						 * inode address_space, or NULL.
						 * If page mapped as anonymous
						 * memory, low bit is set, and
						 * it points to anon_vma object:
						 * see PAGE_MAPPING_ANON below.
						 */
	};
}
```

mapping
这个结构体很关键，如果低位为0，它用于描述该页被映射的地址空间结构体，指向一个文件系统页缓存结构体struct address_space；
如果低位为1，那么它指向一个匿名映射结构体struct anon_vma。
因此这个mapping可能是一个(struct address_space *)类型的指针，也可能是一个(struct anon_vma *)类型的指针;

struct address_space *mapping;表示该页所在地址空间描述结构指针，用于内容为文件的页帧

（1）       如果page->mapping等于0，说明该页属于交换告诉缓存swap cache

（2）       如果page->mapping不等于0，但第0位为0，说明该页为匿名也，此时mapping指向一个struct anon_vma结构变量；

（3）       如果page->mapping不等于0，但第0位不为0，则apping指向一个struct address_space地址空间结构变量；
 

## mapping怎么区分匿名页映射的


内核中的映射分为两种，一种为匿名映射，一种为文件映射，匿名映射对应的结构体为struct anon_vma ；
文件映射会对应页缓存，结构体为struct address_space。由于地址对齐的关系，每个 struct address_space 和 struct anon_vma 结构体都不会存放在奇数地址上，所以假如不做任何处理，mapping正常情况下的最低位肯定是0，那么由于page结构体在内存中会大量存在，为了充分利用每个bit的空间，这里使用了如下操作：

保存struct anon_vma地址到mapping

```
page->mapping = (void *)&anon_vma + 1;
```

从mapping获取struct anon_vma地址
```
struct anon_vma * anon_vma = (struct anon_vma *)(page->mapping - 1);
```



通过保存struct anon_vma结构体地址时加1操作，提取struct anon_vma结构体地址时减1操作，从而可以利用在mapping中的最低位来区分当前page是否为匿名映射：
```
#define PAGE_MAPPING_ANON   0x1
#define PAGE_MAPPING_FLAGS	(PAGE_MAPPING_ANON | PAGE_MAPPING_MOVABLE)
static __always_inline int PageMappingFlags(struct page *page)
{   
    return ((unsigned long)page->mapping & PAGE_MAPPING_FLAGS) != 0;
}
    
static __always_inline int PageAnon(struct page *page)
{
    page = compound_head(page);
    return ((unsigned long)page->mapping & PAGE_MAPPING_ANON) != 0;
}
```

# 页面的反向映射
反向映射是指根据struct page数据结构找到所有映射到这个page的vma，
反向映射主要用于kswaped和页面迁移.反向映射主要调用try_to_unmap来进行


# map_walk

下面是一个分支函数，分为共享页，匿名页，文件映射页三种情况调用不同的处理函数
```
void rmap_walk(struct page *page, struct rmap_walk_control *rwc)
{
    if (unlikely(PageKsm(page)))
        rmap_walk_ksm(page, rwc);
    else if (PageAnon(page))
        rmap_walk_anon(page, rwc, false);
    else
        rmap_walk_file(page, rwc, false);
}
```

# mmap and anon

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/anon_page/anon.png)

```
static unsigned long foo(void)
{
  struct vm_area_struct *vma;
  unsigned long addr = 0;
  
  vma = kmem_cache_zalloc(vm_area_cachep, GFP_KERNEL);
  if (vma == NULL)
    goto out;

  addr = 0x10000;
  INIT_LIST_HEAD(&vma->anon_vma_chain);
  vma->vm_mm = current->mm;
  vma->vm_start = addr;
  vma->vm_end = addr + 0x1000;
  vma->vm_ops = &ralloc_vm_ops;
  vma->vm_flags = VM_READ | VM_WRITE | VM_MIXEDMAP;
  vma->vm_page_prot = vm_get_page_prot(vma->vm_flags);
  vma->vm_pgoff = 0x10000 >> PAGE_SHIFT;
  insert_vm_struct(current->mm, vma);

out:
  return addr;
}
```

# cow

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/anon_page/cow.png)




# page_lock_anon_vma


```
struct anon_vma *page_lock_anon_vma(struct page *page)
{
	struct anon_vma *anon_vma = NULL;
	struct anon_vma *root_anon_vma;
	unsigned long anon_mapping;

	rcu_read_lock();
	anon_mapping = (unsigned long) ACCESS_ONCE(page->mapping);
	if ((anon_mapping & PAGE_MAPPING_FLAGS) != PAGE_MAPPING_ANON)
		goto out;
	if (!page_mapped(page))
		goto out;

	anon_vma = (struct anon_vma *) (anon_mapping - PAGE_MAPPING_ANON);
	root_anon_vma = ACCESS_ONCE(anon_vma->root);
	if (mutex_trylock(&root_anon_vma->mutex)) {
		/*
                                                             
                                                             
                                      
   */
		if (!page_mapped(page)) {
			mutex_unlock(&root_anon_vma->mutex);
			anon_vma = NULL;
		}
		goto out;
	}

	/*                                 */
	if (!atomic_inc_not_zero(&anon_vma->refcount)) {
		anon_vma = NULL;
		goto out;
	}

	if (!page_mapped(page)) {
		put_anon_vma(anon_vma);
		anon_vma = NULL;
		goto out;
	}

	/*                                           */
	rcu_read_unlock();
	anon_vma_lock(anon_vma);

	if (atomic_dec_and_test(&anon_vma->refcount)) {
		/*
                                                      
                                                        
                                                     
   */
		anon_vma_unlock(anon_vma);
		__put_anon_vma(anon_vma);
		anon_vma = NULL;
	}

	return anon_vma;

out:
	rcu_read_unlock();
	return anon_vma;
}
```

```
static int try_to_unmap_anon(struct page *page, enum ttu_flags flags)
{
	struct anon_vma *anon_vma;
	struct anon_vma_chain *avc;
	int ret = SWAP_AGAIN;

	anon_vma = page_lock_anon_vma(page);
	if (!anon_vma)
		return ret;

	list_for_each_entry(avc, &anon_vma->head, same_anon_vma) {
		struct vm_area_struct *vma = avc->vma;
		unsigned long address;

		/*
                                                           
                                                         
                                                         
                                                        
                                                    
                                                 
   */
		if (IS_ENABLED(CONFIG_MIGRATION) && (flags & TTU_MIGRATION) &&
				is_vma_temporary_stack(vma))
			continue;

		address = vma_address(page, vma);
		if (address == -EFAULT)
			continue;
		ret = try_to_unmap_one(page, vma, address, flags);
		if (ret != SWAP_AGAIN || !page_mapped(page))
			break;
	}

	page_unlock_anon_vma(anon_vma);
	return ret;
}
```

# static void validate_mm(struct mm_struct *mm)



```
static void validate_mm(struct mm_struct *mm)
{
	int bug = 0;
	int i = 0;
	unsigned long highest_address = 0;
	struct vm_area_struct *vma = mm->mmap;

	while (vma) {
		struct anon_vma *anon_vma = vma->anon_vma;
		struct anon_vma_chain *avc;

		if (anon_vma) {
			anon_vma_lock_read(anon_vma);
			list_for_each_entry(avc, &vma->anon_vma_chain, same_vma)
				anon_vma_interval_tree_verify(avc);
			anon_vma_unlock_read(anon_vma);
		}

		highest_address = vm_end_gap(vma);
		vma = vma->vm_next;
		i++;
	}
	if (i != mm->map_count) {
		pr_emerg("map_count %d vm_next %d\n", mm->map_count, i);
		bug = 1;
	}
	if (highest_address != mm->highest_vm_end) {
		pr_emerg("mm->highest_vm_end %lx, found %lx\n",
			  mm->highest_vm_end, highest_address);
		bug = 1;
	}
	i = browse_rb(mm);
	if (i != mm->map_count) {
		if (i != -1)
			pr_emerg("map_count %d rb %d\n", mm->map_count, i);
		bug = 1;
	}
	VM_BUG_ON_MM(bug, mm);
}
```


#  page_mapped
```
/*
 * Return true if this page is mapped into pagetables.
 * For compound page it returns true if any subpage of compound page is mapped.
 */
bool page_mapped(struct page *page)
{
        int i;

        if (likely(!PageCompound(page)))
                return atomic_read(&page->_mapcount) >= 0;
        page = compound_head(page);
        if (atomic_read(compound_mapcount_ptr(page)) >= 0)
                return true;
        if (PageHuge(page))
                return false;
        for (i = 0; i < (1 << compound_order(page)); i++) {
                if (atomic_read(&page[i]._mapcount) >= 0)
                        return true;
        }
        return false;
}
EXPORT_SYMBOL(page_mapped);

struct anon_vma *page_anon_vma(struct page *page)
{
        unsigned long mapping;

        page = compound_head(page);
        mapping = (unsigned long)page->mapping;
        if ((mapping & PAGE_MAPPING_FLAGS) != PAGE_MAPPING_ANON)
                return NULL;
        return __page_rmapping(page);
}
```
# collect_procs_anon

```
/*
 * Collect processes when the error hit an anonymous page.
 */
static void collect_procs_anon(struct page *page, struct list_head *to_kill,
                              struct to_kill **tkc, int force_early)
{
        struct vm_area_struct *vma;
        struct task_struct *tsk;
        struct anon_vma *av;
        pgoff_t pgoff;

        av = page_lock_anon_vma_read(page);
        if (av == NULL) /* Not actually mapped anymore */
                return;

        pgoff = page_to_pgoff(page);
        read_lock(&tasklist_lock);
        for_each_process (tsk) {
                struct anon_vma_chain *vmac;
                struct task_struct *t = task_early_kill(tsk, force_early);

                if (!t)
                        continue;
                anon_vma_interval_tree_foreach(vmac, &av->rb_root,
                                               pgoff, pgoff) {
                        vma = vmac->vma;
                        if (!page_mapped_in_vma(page, vma))
                                continue;
                        if (vma->vm_mm == t->mm)
                                add_to_kill(t, page, vma, to_kill, tkc);
                }
        }
        read_unlock(&tasklist_lock);
        page_unlock_anon_vma_read(av);

```


# page_mapping

```

struct address_space *page_mapping(struct page *page)
{
        struct address_space *mapping;

        page = compound_head(page);

        /* This happens if someone calls flush_dcache_page on slab page */
        if (unlikely(PageSlab(page)))
                return NULL;

        if (unlikely(PageSwapCache(page))) {
                swp_entry_t entry;

                entry.val = page_private(page);
                return swap_address_space(entry);
        }

        mapping = page->mapping;
        if ((unsigned long)mapping & PAGE_MAPPING_ANON)
                return NULL;

        return (void *)((unsigned long)mapping & ~PAGE_MAPPING_FLAGS);
}
EXPORT_SYMBOL(page_mapping);

```



