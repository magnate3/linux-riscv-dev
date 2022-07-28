
# insmod  mmdump_test.ko pid=10284
```
[root@centos7 page_cache]# ps -elf | grep 10284
0 S root      10284   8090  0  80   0 -  4138 n_tty_ 03:48 pts/1    00:00:00 ./mmap_fork
1 S root      10285  10284  0  80   0 -  4138 wait_w 03:48 pts/1    00:00:00 ./mmap_fork
insmod  mmdump_test.ko pid=10284

[root@centos7 address_space]# gcc mmap_fork.c -o mmap_fork
[root@centos7 address_space]# ./mmap_fork 
mmap: Success
resust addr : 0xffff9fba0000, and 9fba0000lx
integerSize addr : 0xffffd9be7e68, and d9be7e68lx
before wirte please findpage resust addr 

after wirte please findpage resust addr 

```



```
[20108.715376] *************************** mmdump module load 
[20108.721072] **************** vma->vm_ops: ffff000001a52278 
[20108.726619] mapping->a_ops: ffff000001a520c8 
[20108.726620] 400000 410000 mmap_fork
[20108.734520] **************** vma->vm_ops: ffff000001a52278 
[20108.740066] mapping->a_ops: ffff000001a520c8 
[20108.740067] 410000 420000 mmap_fork
[20108.747965] **************** vma->vm_ops: ffff000001a52278 
[20108.753515] mapping->a_ops: ffff000001a520c8 
[20108.753516] 420000 430000 mmap_fork
[20108.761413] ****************** vma->vm_ops is null 
[20108.766268] 8bf0000 8c20000 [anon], vma->vm_file is null 
[20108.771645] ---------------start------------------
[20108.776415] bad_address 
[20108.778936] bad_address 
[20108.781462] PUD 5fd4f10003 
[20108.781462] bad_address 
[20108.786764] PMD 5fd6930003 
[20108.786765] bad_address 
[20108.792070] PTE e0005ee0750000|
[20108.792071] 11010011

[20108.798857] bad_address 
[20108.801383] bad_address 
[20108.803905] PUD 5fd4f10003 
[20108.803906] bad_address 
[20108.809207] PMD 5fd6930003 
[20108.809208] bad_address 
[20108.814513] PTE 0|
[20108.814513] 00000000

[20108.820175] bad_address 
[20108.822700] bad_address 
[20108.825222] PUD 5fd4f10003 
[20108.825223] bad_address 
[20108.830524] PMD 5fd6930003 
[20108.830525] bad_address 
[20108.835830] PTE 0|
[20108.835830] 00000000

[20108.841495] ----------------end---------------
[20108.845918] ****************** vma->vm_ops is null 
[20108.850778] ffff9fb90000 ffff9fba0000 [anon], vma->vm_file is null 
[20108.857015] ---------------start------------------
[20108.861787] bad_address 
[20108.864309] bad_address 
[20108.866831] PUD 5fd4f10003 
[20108.866832] bad_address 
[20108.872136] PMD 0 

[20108.875625] ----------------end---------------
[20108.880047] **************** vma->vm_ops: ffff0000088df100 
[20108.885597] mapping->a_ops: ffff0000088ded80 
[20108.885598] ffff9fba0000 ffffafba0000 dev/zero
[20108.894447] **************** vma->vm_ops: ffff000001a52278 
[20108.899993] mapping->a_ops: ffff000001a520c8 
[20108.899994] ffffafba0000 ffffafd10000 libc-2.17.so
[20108.909188] **************** vma->vm_ops: ffff000001a52278 
[20108.914739] mapping->a_ops: ffff000001a520c8 
[20108.914740] ffffafd10000 ffffafd20000 libc-2.17.so
[20108.923935] **************** vma->vm_ops: ffff000001a52278 
[20108.929481] mapping->a_ops: ffff000001a520c8 
[20108.929482] ffffafd20000 ffffafd30000 libc-2.17.so
[20108.938675] ****************** vma->vm_ops is null 
[20108.943534] ffffafd30000 ffffafd40000 [anon], vma->vm_file is null 
[20108.949771] ---------------start------------------
[20108.954544] bad_address 
[20108.957066] bad_address 
[20108.959588] PUD 5fd4f10003 
[20108.959589] bad_address 
[20108.964894] PMD 0 

[20108.968381] ----------------end---------------
[20108.972808] **************** vma->vm_ops: ffff0000088e1218 
[20108.978354] ffffafd40000 ffffafd50000 [anon], vma->vm_file is null 
[20108.984594] ---------------start------------------
[20108.989363] bad_address 
[20108.991890] bad_address 
[20108.994412] PUD 5fd4f10003 
[20108.994413] bad_address 
[20108.999714] PMD 0 

[20109.003205] ----------------end---------------
[20109.007628] **************** vma->vm_ops: ffff0000088e1218 
[20109.013178] ffffafd50000 ffffafd60000 [anon], vma->vm_file is null 
[20109.019416] ---------------start------------------
[20109.024185] bad_address 
[20109.026707] bad_address 
[20109.029229] PUD 5fd4f10003 
[20109.029230] bad_address 
[20109.034536] PMD 0 

[20109.038023] ----------------end---------------
[20109.042449] **************** vma->vm_ops: ffff000001a52278 
[20109.047995] mapping->a_ops: ffff000001a520c8 
[20109.047996] ffffafd60000 ffffafd80000 ld-2.17.so
[20109.057017] **************** vma->vm_ops: ffff000001a52278 
[20109.062567] mapping->a_ops: ffff000001a520c8 
[20109.062567] ffffafd80000 ffffafd90000 ld-2.17.so
[20109.071590] **************** vma->vm_ops: ffff000001a52278 
[20109.077135] mapping->a_ops: ffff000001a520c8 
[20109.077136] ffffafd90000 ffffafda0000 ld-2.17.so
[20109.086156] ****************** vma->vm_ops is null 
[20109.091015] ffffd9bc0000 ffffd9bf0000 [anon], vma->vm_file is null 
[20109.097252] ---------------start------------------
[20109.102024] bad_address 
[20109.104547] bad_address 
[20109.107068] PUD 5fd4f10003 
[20109.107069] bad_address 
[20109.112376] PMD 0 

[20109.115864] bad_address 
[20109.118386] bad_address 
[20109.120910] PUD 5fd4f10003 
[20109.120911] bad_address 
[20109.126213] PMD 0 

[20109.129701] bad_address 
[20109.132226] bad_address 
[20109.134749] PUD 5fd4f10003 
[20109.134749] bad_address 
[20109.140050] PMD 0 

[20109.143541] ----------------end---------------
```

## vma->vm_file is null 

```
[20109.007628] **************** vma->vm_ops: ffff0000088e1218 
[20109.013178] ffffafd50000 ffffafd60000 [anon], vma->vm_file is null 
```

## vma->vm_ops

### vma->vm_ops: ffff000001a52278

```
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff000001a52278
ffff000001a52278 r $d   [xfs]
ffff000001a52278 r xfs_file_vm_ops      [xfs]
```

### vma->vm_ops: ffff0000088df100

```
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff0000088df100
ffff0000088df100 r shmem_vm_ops
[root@centos7 page_cache]# 
```

###  vma->vm_ops: ffff0000088e1218 

```
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff0000088e1218
ffff0000088e1218 r special_mapping_vmops
[root@centos7 page_cache]# 
```

## mapping->a_ops

```
        if (vma->vm_file){
            file = vma->vm_file;
            mapping = file->f_mapping; 
            ops=mapping->a_ops;
            if (ops){
                pr_info("mapping->a_ops: %p \t", ops);
            }
            printk(KERN_INFO "%lx %lx %s\n", vma->vm_start, vma->vm_end, vma->vm_file->f_path.dentry->d_iname);
        }
```

### mapping->a_ops: ffff0000088ded80 
```
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff0000088ded80 
ffff0000088ded80 r shmem_aops
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff000001a520c8
ffff000001a520c8 r xfs_address_space_operations [xfs]
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff000001a520c8
```

### mapping->a_ops: ffff000001a520c8 
```
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff0000088ded80 
ffff0000088ded80 r shmem_aops
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff000001a520c8
ffff000001a520c8 r xfs_address_space_operations [xfs]
[root@centos7 page_cache]# cat /proc/kallsyms  | grep ffff000001a520c8
```