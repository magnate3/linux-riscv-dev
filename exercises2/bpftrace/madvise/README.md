
+ madvise   
```
root@ubuntux86:# bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise/pid == 1481/{printf("%s",ustack)}'
Attaching 1 probe...
^C

root@ubuntux86:# bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise/pid==1481/{printf("%s",ustack)}'
Attaching 1 probe...
^C

root@ubuntux86:# 
```

+  kprobe:hugepage_madvise   
```
root@ubuntux86:#  bpftrace -e 'kprobe:hugepage_madvise {printf("%s\n", kstack);}'
Attaching 1 probe...
^C

root@ubuntux86:#
```

+  kprobe:find_vma

```
root@ubuntux86:# bpftrace -e 'kprobe:find_vma {  @[kstack] = count(); }'
Attaching 1 probe...
^C

@[
    find_vma+1
    mmap_region+746
    do_mmap+930
    vm_mmap_pgoff+212
    ksys_mmap_pgoff+91
    __x64_sys_mmap+51
    do_syscall_64+97
    entry_SYSCALL_64_after_hwframe+68
]: 2
@[
    find_vma+1
    exc_page_fault+125
    asm_exc_page_fault+30
]: 14
@[
    find_vma+1
    __x64_sys_mprotect+31
    do_syscall_64+97
    entry_SYSCALL_64_after_hwframe+68
]: 28

root@ubuntux86:# 
```