### 通过缺页中断来“填充”页表

内存管理并不直接分配物理内存，只有等你真正用的那一刻才会开始分配。只有访问虚拟内存的时候，发现没有映射多物理内存，页表也没有创建过，才触发缺页异常。进入内核调用 do_page_fault，一直调用到 __handle_mm_fault，__handle_mm_fault 调用 pud_alloc 和 pmd_alloc，来创建相应的页目录项，最后调用 handle_pte_fault 来创建页表项。

```c
static noinline void
__do_page_fault(struct pt_regs *regs, unsigned long error_code,
        unsigned long address){
    struct vm_area_struct *vma;
    struct task_struct *tsk;
    struct mm_struct *mm;
    tsk = current;
    mm = tsk->mm;
    // 判断缺页是否发生在内核
    if (unlikely(fault_in_kernel_space(address))) {
        if (vmalloc_fault(address) >= 0)
            return;
    }
    ......
    // 找到待访问地址所在的区域 vm_area_struct
    vma = find_vma(mm, address);
    ......
    fault = handle_mm_fault(vma, address, flags);
    ......

static int __handle_mm_fault(struct vm_area_struct *vma, unsigned long address,
        unsigned int flags){
    struct vm_fault vmf = {
        .vma = vma,
        .address = address & PAGE_MASK,
        .flags = flags,
        .pgoff = linear_page_index(vma, address),
        .gfp_mask = __get_fault_gfp_mask(vma),
    };
    struct mm_struct *mm = vma->vm_mm;
    pgd_t *pgd;
    p4d_t *p4d;
    int ret;
    pgd = pgd_offset(mm, address);
    p4d = p4d_alloc(mm, pgd, address);
    ......
    vmf.pud = pud_alloc(mm, p4d, address);
    ......
    vmf.pmd = pmd_alloc(mm, vmf.pud, address);
    ......
    return handle_pte_fault(&vmf);
}
```

以handle_pte_fault 的一种场景 do_anonymous_page为例：先通过 pte_alloc 分配一个页表项，然后通过 alloc_zeroed_user_highpage_movable 分配一个页，接下来要调用 mk_pte，将页表项指向新分配的物理页，set_pte_at 会将页表项塞到页表里面。

```c
static int do_anonymous_page(struct vm_fault *vmf){
    struct vm_area_struct *vma = vmf->vma;
    struct mem_cgroup *memcg;
    struct page *page;
    int ret = 0;
    pte_t entry;
    ......
    if (pte_alloc(vma->vm_mm, vmf->pmd, vmf->address))
        return VM_FAULT_OOM;
    ......
    page = alloc_zeroed_user_highpage_movable(vma, vmf->address);
    ......
    entry = mk_pte(page, vma->vm_page_prot);
    if (vma->vm_flags & VM_WRITE)
        entry = pte_mkwrite(pte_mkdirty(entry));
    vmf->pte = pte_offset_map_lock(vma->vm_mm, vmf->pmd, vmf->address,
            &vmf->ptl);
    ......
    set_pte_at(vma->vm_mm, vmf->address, vmf->pte, entry);
    ......
}

```


# 内核页表修改 set_pte_at

在vmalloc、vmap亦或者ioremap时，内核空间页表会进程调整。
以vmalloc和vmap为例，进行地址映射时，会调用map_vm_area，最终调用到vmap_pte_range
static int vmap_pte_range(pmd_t *pmd, unsigned long addr,
		unsigned long end, pgprot_t prot, struct page **pages, int *nr)
{
	pte_t *pte;

	/*
	 * nr is a running index into the array which helps higher level
	 * callers keep track of where we're up to.
	 */

	pte = pte_alloc_kernel(pmd, addr);											(1)
	if (!pte)
		return -ENOMEM;
	do {
		struct page *page = pages[*nr];

		if (WARN_ON(!pte_none(*pte)))
			return -EBUSY;
		if (WARN_ON(!page))
			return -ENOMEM;
		set_pte_at(&init_mm, addr, pte, mk_pte(page, prot));		/* 设置pte页表项内容 */
		(*nr)++;
	} while (pte++, addr += PAGE_SIZE, addr != end);
	return 0;
}

[分析匿名页(anonymous_page)映射](https://cloud.tencent.com/developer/article/1622985)
[Miseratio/linux-page-table-walkthrough](https://github.com/Miseratio/linux-page-table-walkthrough)