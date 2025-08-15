
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

#  insmod  mmdump_test.ko pid=1

```
[21179.313040] ----------------end---------------
[21179.317463] **************** vma->vm_ops: ffff000001a52278 
[21179.323014] mapping->a_ops: ffff000001a520c8 
[21179.323016] ffffa39c0000 ffffa39d0000 libuuid.so.1.3.0
[21179.332552] **************** vma->vm_ops: ffff000001a52278 
[21179.338102] mapping->a_ops: ffff000001a520c8 
[21179.338103] ffffa39d0000 ffffa39e0000 libuuid.so.1.3.0
[21179.347644] **************** vma->vm_ops: ffff000001a52278 
[21179.353194] mapping->a_ops: ffff000001a520c8 
[21179.353195] ffffa39e0000 ffffa39f0000 libuuid.so.1.3.0
[21179.362730] **************** vma->vm_ops: ffff000001a52278 
[21179.368281] mapping->a_ops: ffff000001a520c8 
[21179.368282] ffffa39f0000 ffffa3a30000 libblkid.so.1.1.0
[21179.377908] **************** vma->vm_ops: ffff000001a52278 
[21179.383459] mapping->a_ops: ffff000001a520c8 
[21179.383460] ffffa3a30000 ffffa3a40000 libblkid.so.1.1.0
[21179.393086] **************** vma->vm_ops: ffff000001a52278 
[21179.398632] mapping->a_ops: ffff000001a520c8 
[21179.398632] ffffa3a40000 ffffa3a50000 libblkid.so.1.1.0
[21179.408259] **************** vma->vm_ops: ffff000001a52278 
[21179.413810] mapping->a_ops: ffff000001a520c8 
[21179.413811] ffffa3a50000 ffffa3a70000 libz.so.1.2.7
[21179.423093] **************** vma->vm_ops: ffff000001a52278 
[21179.428639] mapping->a_ops: ffff000001a520c8 
[21179.428640] ffffa3a70000 ffffa3a80000 libz.so.1.2.7
[21179.437920] **************** vma->vm_ops: ffff000001a52278 
[21179.443471] mapping->a_ops: ffff000001a520c8 
[21179.443473] ffffa3a80000 ffffa3a90000 libz.so.1.2.7
[21179.452750] **************** vma->vm_ops: ffff000001a52278 
[21179.458300] mapping->a_ops: ffff000001a520c8 
[21179.458302] ffffa3a90000 ffffa3ac0000 liblzma.so.5.2.2
[21179.467842] **************** vma->vm_ops: ffff000001a52278 
[21179.473391] mapping->a_ops: ffff000001a520c8 
[21179.473392] ffffa3ac0000 ffffa3ad0000 liblzma.so.5.2.2
[21179.482928] **************** vma->vm_ops: ffff000001a52278 
[21179.488479] mapping->a_ops: ffff000001a520c8 
[21179.488481] ffffa3ad0000 ffffa3ae0000 liblzma.so.5.2.2
[21179.498021] **************** vma->vm_ops: ffff000001a52278 
[21179.503572] mapping->a_ops: ffff000001a520c8 
[21179.503574] ffffa3ae0000 ffffa3af0000 libcap-ng.so.0.0.0
[21179.513287] **************** vma->vm_ops: ffff000001a52278 
[21179.518833] mapping->a_ops: ffff000001a520c8 
[21179.518834] ffffa3af0000 ffffa3b00000 libcap-ng.so.0.0.0
[21179.528546] **************** vma->vm_ops: ffff000001a52278 
[21179.534097] mapping->a_ops: ffff000001a520c8 
[21179.534098] ffffa3b00000 ffffa3b10000 libcap-ng.so.0.0.0
[21179.543812] **************** vma->vm_ops: ffff000001a52278 
[21179.549358] mapping->a_ops: ffff000001a520c8 
[21179.549359] ffffa3b10000 ffffa3b20000 libattr.so.1.1.0
[21179.558898] **************** vma->vm_ops: ffff000001a52278 
[21179.564449] mapping->a_ops: ffff000001a520c8 
[21179.564450] ffffa3b20000 ffffa3b30000 libattr.so.1.1.0
[21179.573990] **************** vma->vm_ops: ffff000001a52278 
[21179.579536] mapping->a_ops: ffff000001a520c8 
[21179.579537] ffffa3b30000 ffffa3b40000 libattr.so.1.1.0
[21179.589078] **************** vma->vm_ops: ffff000001a52278 
[21179.594628] mapping->a_ops: ffff000001a520c8 
[21179.594629] ffffa3b40000 ffffa3b50000 libdl-2.17.so
[21179.603912] **************** vma->vm_ops: ffff000001a52278 
[21179.609458] mapping->a_ops: ffff000001a520c8 
[21179.609459] ffffa3b50000 ffffa3b60000 libdl-2.17.so
[21179.618740] **************** vma->vm_ops: ffff000001a52278 
[21179.624292] mapping->a_ops: ffff000001a520c8 
[21179.624293] ffffa3b60000 ffffa3b70000 libdl-2.17.so
[21179.633575] **************** vma->vm_ops: ffff000001a52278 
[21179.639121] mapping->a_ops: ffff000001a520c8 
[21179.639122] ffffa3b70000 ffffa3bb0000 libpcre.so.1.2.0
[21179.648663] **************** vma->vm_ops: ffff000001a52278 
[21179.654214] mapping->a_ops: ffff000001a520c8 
[21179.654215] ffffa3bb0000 ffffa3bc0000 libpcre.so.1.2.0
[21179.663755] **************** vma->vm_ops: ffff000001a52278 
[21179.669301] mapping->a_ops: ffff000001a520c8 
[21179.669302] ffffa3bc0000 ffffa3bd0000 libpcre.so.1.2.0
[21179.678842] **************** vma->vm_ops: ffff000001a52278 
[21179.684393] mapping->a_ops: ffff000001a520c8 
[21179.684394] ffffa3bd0000 ffffa3d40000 libc-2.17.so
[21179.693590] **************** vma->vm_ops: ffff000001a52278 
[21179.699136] mapping->a_ops: ffff000001a520c8 
[21179.699137] ffffa3d40000 ffffa3d50000 libc-2.17.so
[21179.708333] **************** vma->vm_ops: ffff000001a52278 
[21179.713890] mapping->a_ops: ffff000001a520c8 
[21179.713891] ffffa3d50000 ffffa3d60000 libc-2.17.so
[21179.723088] **************** vma->vm_ops: ffff000001a52278 
[21179.728634] mapping->a_ops: ffff000001a520c8 
[21179.728635] ffffa3d60000 ffffa3d80000 libpthread-2.17.so
[21179.738346] **************** vma->vm_ops: ffff000001a52278 
[21179.743897] mapping->a_ops: ffff000001a520c8 
[21179.743899] ffffa3d80000 ffffa3d90000 libpthread-2.17.so
[21179.753610] **************** vma->vm_ops: ffff000001a52278 
[21179.759156] mapping->a_ops: ffff000001a520c8 
[21179.759157] ffffa3d90000 ffffa3da0000 libpthread-2.17.so
[21179.768870] **************** vma->vm_ops: ffff000001a52278 
[21179.774420] mapping->a_ops: ffff000001a520c8 
[21179.774421] ffffa3da0000 ffffa3dc0000 libgcc_s-4.8.5-20150702.so.1
[21179.785000] **************** vma->vm_ops: ffff000001a52278 
[21179.790546] mapping->a_ops: ffff000001a520c8 
[21179.790547] ffffa3dc0000 ffffa3dd0000 libgcc_s-4.8.5-20150702.so.1
[21179.801123] **************** vma->vm_ops: ffff000001a52278 
[21179.806674] mapping->a_ops: ffff000001a520c8 
[21179.806675] ffffa3dd0000 ffffa3de0000 libgcc_s-4.8.5-20150702.so.1
[21179.817253] **************** vma->vm_ops: ffff000001a52278 
[21179.822799] mapping->a_ops: ffff000001a520c8 
[21179.822800] ffffa3de0000 ffffa3df0000 librt-2.17.so
[21179.832082] **************** vma->vm_ops: ffff000001a52278 
[21179.837631] mapping->a_ops: ffff000001a520c8 
[21179.837632] ffffa3df0000 ffffa3e00000 librt-2.17.so
[21179.846915] **************** vma->vm_ops: ffff000001a52278 
[21179.852461] mapping->a_ops: ffff000001a520c8 
[21179.852462] ffffa3e00000 ffffa3e10000 librt-2.17.so
[21179.861743] **************** vma->vm_ops: ffff000001a52278 
[21179.867295] mapping->a_ops: ffff000001a520c8 
[21179.867296] ffffa3e10000 ffffa3e50000 libmount.so.1.1.0
[21179.876923] **************** vma->vm_ops: ffff000001a52278 
[21179.882469] mapping->a_ops: ffff000001a520c8 
[21179.882470] ffffa3e50000 ffffa3e60000 libmount.so.1.1.0
[21179.892097] **************** vma->vm_ops: ffff000001a52278 
[21179.897646] mapping->a_ops: ffff000001a520c8 
[21179.897647] ffffa3e60000 ffffa3e70000 libmount.so.1.1.0
[21179.907274] **************** vma->vm_ops: ffff000001a52278 
[21179.912820] mapping->a_ops: ffff000001a520c8 
[21179.912821] ffffa3e70000 ffffa3e80000 libmount.so.1.1.0
[21179.922446] **************** vma->vm_ops: ffff000001a52278 
[21179.927996] mapping->a_ops: ffff000001a520c8 
[21179.927997] ffffa3e80000 ffffa3ea0000 libkmod.so.2.2.10
[21179.937624] **************** vma->vm_ops: ffff000001a52278 
[21179.943176] mapping->a_ops: ffff000001a520c8 
[21179.943177] ffffa3ea0000 ffffa3eb0000 libkmod.so.2.2.10
[21179.952799] **************** vma->vm_ops: ffff000001a52278 
[21179.958350] mapping->a_ops: ffff000001a520c8 
[21179.958351] ffffa3eb0000 ffffa3ec0000 libkmod.so.2.2.10
[21179.967978] **************** vma->vm_ops: ffff000001a52278 
[21179.973529] mapping->a_ops: ffff000001a520c8 
[21179.973530] ffffa3ec0000 ffffa3ee0000 libaudit.so.1.0.0
[21179.983157] **************** vma->vm_ops: ffff000001a52278 
[21179.988703] mapping->a_ops: ffff000001a520c8 
[21179.988704] ffffa3ee0000 ffffa3ef0000 libaudit.so.1.0.0
[21179.998329] **************** vma->vm_ops: ffff000001a52278 
[21180.003881] mapping->a_ops: ffff000001a520c8 
[21180.003882] ffffa3ef0000 ffffa3f00000 libaudit.so.1.0.0
[21180.013509] **************** vma->vm_ops: ffff000001a52278 
[21180.019055] mapping->a_ops: ffff000001a520c8 
[21180.019056] ffffa3f00000 ffffa3f10000 libpam.so.0.83.1
[21180.028598] **************** vma->vm_ops: ffff000001a52278 
[21180.034148] mapping->a_ops: ffff000001a520c8 
[21180.034150] ffffa3f10000 ffffa3f20000 libpam.so.0.83.1
[21180.043690] **************** vma->vm_ops: ffff000001a52278 
[21180.049236] mapping->a_ops: ffff000001a520c8 
[21180.049237] ffffa3f20000 ffffa3f30000 libpam.so.0.83.1
[21180.058778] **************** vma->vm_ops: ffff000001a52278 
[21180.064329] mapping->a_ops: ffff000001a520c8 
[21180.064331] ffffa3f30000 ffffa3f40000 libcap.so.2.22
[21180.073698] **************** vma->vm_ops: ffff000001a52278 
[21180.079244] mapping->a_ops: ffff000001a520c8 
[21180.079245] ffffa3f40000 ffffa3f50000 libcap.so.2.22
[21180.088614] **************** vma->vm_ops: ffff000001a52278 
[21180.094165] mapping->a_ops: ffff000001a520c8 
[21180.094166] ffffa3f50000 ffffa3f60000 libcap.so.2.22
[21180.103535] **************** vma->vm_ops: ffff000001a52278 
[21180.109081] mapping->a_ops: ffff000001a520c8 
[21180.109082] ffffa3f60000 ffffa3f90000 libselinux.so.1
[21180.118536] **************** vma->vm_ops: ffff000001a52278 
[21180.124086] mapping->a_ops: ffff000001a520c8 
[21180.124088] ffffa3f90000 ffffa3fa0000 libselinux.so.1
[21180.133541] **************** vma->vm_ops: ffff000001a52278 
[21180.139087] mapping->a_ops: ffff000001a520c8 
[21180.139088] ffffa3fa0000 ffffa3fb0000 libselinux.so.1
[21180.148542] ****************** vma->vm_ops is null 
[21180.153401] ffffa3fb0000 ffffa3fc0000 [anon], vma->vm_file is null 
[21180.159638] ---------------start------------------
[21180.164410] bad_address 
[21180.166932] bad_address 
[21180.169454] PUD 0 

[21180.172941] ----------------end---------------
[21180.177367] **************** vma->vm_ops: ffff0000088e1218 
[21180.182913] ffffa3fc0000 ffffa3fd0000 [anon], vma->vm_file is null 
[21180.189155] ---------------start------------------
[21180.193927] bad_address 
[21180.196449] bad_address 
[21180.198971] PUD 0 

[21180.202458] ----------------end---------------
[21180.206886] **************** vma->vm_ops: ffff0000088e1218 
[21180.212433] ffffa3fd0000 ffffa3fe0000 [anon], vma->vm_file is null 
[21180.218673] ---------------start------------------
[21180.223447] bad_address 
[21180.225969] bad_address 
[21180.228491] PUD 0 

[21180.231978] ----------------end---------------
[21180.236404] **************** vma->vm_ops: ffff000001a52278 
[21180.241951] mapping->a_ops: ffff000001a520c8 
[21180.241952] ffffa3fe0000 ffffa4000000 ld-2.17.so
[21180.250973] **************** vma->vm_ops: ffff000001a52278 
[21180.256522] mapping->a_ops: ffff000001a520c8 
[21180.256523] ffffa4000000 ffffa4010000 ld-2.17.so
[21180.265545] **************** vma->vm_ops: ffff000001a52278 
[21180.271091] mapping->a_ops: ffff000001a520c8 
[21180.271092] ffffa4010000 ffffa4020000 ld-2.17.so
[21180.280112] ****************** vma->vm_ops is null 
[21180.284972] ffffce940000 ffffce970000 [anon], vma->vm_file is null 
[21180.291210] ---------------start------------------
[21180.295980] bad_address 
[21180.298502] bad_address 
[21180.301024] PUD 0 

[21180.304517] bad_address 
[21180.307039] bad_address 
[21180.309561] PUD 0 

[21180.313052] bad_address 
[21180.315574] bad_address 
[21180.318096] PUD 0 
```