
#  S_DAX

```
{

	/* fake DAX */
	filp->f_inode->i_flags |= S_DAX;
	/* FPNMAP */
	vma->vm_flags |= VM_PFNMAP;

	return 0;
}
```

# get_unmapped_area 

[从内核世界透视 mmap 内存映射的本质](https://ost.51cto.com/posts/27290)   

# 透明页

+ centos7
```
[root@centos7 boot]# grep CONFIG_HAVE_ARCH_TRANSPARENT_HUGEPAGE_PUD config-4.14.0-115.el7a.0.1.aarch64 
[root@centos7 boot]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 boot]# 
```


```
[root@centos7 huge_fault]# insmod  huge_test.ko 
insmod: ERROR: could not insert module huge_test.ko: Unknown symbol in module
[root@centos7 huge_fault]# dmesg | tail -n 10
[12915.664792] node_data[3]->node_spanned_pages = 2097152.
[12915.670009] Zone DMA - 0
[12915.672535]   0  0 0 0
[12915.674886] Zone Normal - 1
[12915.677669]   20400000  2094961 2097152 2097152
[12915.682193] Zone Movable - 0
[12915.685063]   0  0 0 0
[12915.687415] You have 4 node(s) in your system!
[14215.824621] Goodbye, this is exit_mem_map().
[445151.121018] huge_test: Unknown symbol vmf_insert_pfn_pud (err 0)
[root@centos7 huge_fault]# cat /proc/kallsyms  | grep vmf_insert_pfn_pud
[root@centos7 huge_fault]# cat /proc/kallsyms  | grep vmf_insert
ffff00000828f718 T vmf_insert_pfn_pmd
ffff000008bb4448 r __ksymtab_vmf_insert_pfn_pmd
ffff000008bc43dc r __kstrtab_vmf_insert_pfn_pmd
```

```
int vmf_insert_pfn_pud(struct vm_area_struct *vma, unsigned long addr,
                        pud_t *pud, pfn_t pfn, bool write)
{
        pgprot_t pgprot = vma->vm_page_prot;
        /*
         * If we had pud_special, we could avoid all these restrictions,
         * but we need to be consistent with PTEs and architectures that
         * can't support a 'special' bit.
         */
        BUG_ON(!(vma->vm_flags & (VM_PFNMAP|VM_MIXEDMAP)));
        BUG_ON((vma->vm_flags & (VM_PFNMAP|VM_MIXEDMAP)) ==
                                                (VM_PFNMAP|VM_MIXEDMAP));
        BUG_ON((vma->vm_flags & VM_PFNMAP) && is_cow_mapping(vma->vm_flags));
        BUG_ON(!pfn_t_devmap(pfn));

        if (addr < vma->vm_start || addr >= vma->vm_end)
                return VM_FAULT_SIGBUS;

        track_pfn_insert(vma, &pgprot, pfn);

        insert_pfn_pud(vma, addr, pud, pfn, pgprot, write);
        return VM_FAULT_NOPAGE;
}
EXPORT_SYMBOL_GPL(vmf_insert_pfn_pud);
#endif /* CONFIG_HAVE_ARCH_TRANSPARENT_HUGEPAGE_PUD */
```

#  vm_huge_fault

```
[27552.288824]  dump_stack+0x7d/0x9c
[27552.288832]  vm_huge_fault+0x12/0x33 [huge_test]
[27552.288839]  __handle_mm_fault+0x447/0x8e0
[27552.288846]  handle_mm_fault+0xda/0x2b0
[27552.288851]  do_user_addr_fault+0x1bb/0x650
[27552.288857]  ? syscall_exit_to_user_mode+0x27/0x50
[27552.288864]  exc_page_fault+0x7d/0x170
[27552.288870]  ? asm_exc_page_fault+0x8/0x30
[27552.288878]  asm_exc_page_fault+0x1e/0x30
[27552.288884] RIP: 0033:0x5612086ac287
```


# get_unmapped_area

do_mmap --> get_unmapped_area  
```
[27552.288619]  dump_stack+0x7d/0x9c
[27552.288635]  test_get_unmapped_area+0x17/0x51 [huge_test]
[27552.288643]  get_unmapped_area+0x79/0x130
[27552.288651]  do_mmap+0xf6/0x570
[27552.288657]  ? security_mmap_file+0xa0/0xc0
[27552.288666]  vm_mmap_pgoff+0xd4/0x170
[27552.288678]  ksys_mmap_pgoff+0x1ef/0x270
[27552.288685]  __x64_sys_mmap+0x33/0x40
[27552.288691]  do_syscall_64+0x61/0xb0
[27552.288697]  ? exit_to_user_mode_prepare+0x3d/0x1c0
[27552.288708]  ? syscall_exit_to_user_mode+0x27/0x50
[27552.288715]  ? __x64_sys_openat+0x20/0x30
[27552.288723]  ? do_syscall_64+0x6e/0xb0
[27552.288727]  ? irqentry_exit_to_user_mode+0x9/0x20
[27552.288734]  ? irqentry_exit+0x19/0x30
[27552.288740]  ? exc_page_fault+0x8f/0x170
[27552.288745]  ? asm_exc_page_fault+0x8/0x30
[27552.288754]  entry_SYSCALL_64_after_hwframe+0x44/0xae
[27552.288762] RIP: 0033:0x7fcf7d70cb06
```