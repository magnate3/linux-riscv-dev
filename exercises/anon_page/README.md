
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

# cow

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/anon_page/cow.png)