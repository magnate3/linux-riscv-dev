 # insmod  test1.ko
  ```
 [root@centos7 mem_test]# insmod test1.ko
[root@centos7 mem_test]# dmesg | tail
[3692466.537725] test memory
[3692466.540335] Big memory ok
[3692466.543121] ----------------------------
[3692466.547197] ----------------------------
[3692466.551273] vmalloc test...
[3692466.555540] vmalloc test ok 
[3692466.558593] ----------------------------
[3692466.562675] get_free_pages test...
[3692466.566236] __get_free_pages test ok [1]
[3692466.570313] ----------------------------
  ```
  
  
# insmod vmalloc_example.ko 
 ```
 [root@centos7 mem_test]# dmesg | tail -n 10
[3694374.667340] get_free_pages test...
[3694374.670905] __get_free_pages test ok [1]
[3694374.674982] ----------------------------
[3694556.600823] Module Memory Test End
[3695521.630409] vmalloc example init
[3695521.633797] ***************************************************
[3695521.639870] VMALLOC:     ffff000008000000 -- ffff7bdfffff0000
[3695521.645761] LOWMEM:      ffff800000000000 -- ffffa06000000000
[3695521.651656] ***************************************************
[3695521.658715] [*]unsigned int *VMALLOC_int:    Address: 0xa9530000`
```

#  [root@centos7 mem_test]# insmod kmap_example.ko 

```
[3696158.524456] kmap example init
[3696158.527585] ***************************************************
[3696158.533659] FIXMAP:      ffff7fdffe790000 --        0
[3696158.538862] VMALLOC:     ffff000008000000 -- ffff7bdfffff0000
[3696158.544753] LOWMEM:      ffff800000000000 -- ffffa06000000000
[3696158.550648] ***************************************************
[3696158.556714] [*]unsigned int *KMAP_int:       Address: 0x681e0000
[root@centos7 mem_test]# 
```

#  insmod atomic_example.ko

```
[3697435.536741] highapi atomic map example exit
[3697438.722556] highapi_atomic map example init
[3697438.726903] ***************************************************
[3697438.732968] FIXMAP:      ffff7fdffe790000 -- 
[3697438.737482] VMALLOC:     ffff000008000000 -- ffff7bdfffff0000
[3697438.743373] LOWMEM:      ffff800000000000 -- ffffa06000000000
[3697438.749267] ***************************************************
[3697438.755333] [*]unsigned int *KMAP_atomic:       Address: 0xc5170000
[root@centos7 mem_test]# 
```
# [root@centos7 mem_test]# insmod kmalloc_example.ko
```
 [3697713.581907] kmalloc example init
[3697713.585295] ***************************************************
[3697713.591360] FIXMAP:      ffff7fdffe790000 -- 
[3697713.595876] VMALLOC:     ffff000008000000 -- ffff7bdfffff0000
[3697713.601771] LOWMEM:      ffff800000000000 -- ffffa06000000000
[3697713.607661] ***************************************************
[3697713.613730] [*]unsigned int *NORMAL_int:     Address: 0xd081dc00
[root@centos7 mem_test]# 
```

# slab

##  cat /proc/slabinfo | grep slub_test

```
[root@centos7 mem_test]# cat /proc/slabinfo | grep slub_test
slub_test           4096   4096     16 4096    1 : tunables    0    0    0 : slabdata      1      1      0
```

## ls /sys/kernel/slab/slub_test/

```
[root@centos7 mem_test]# ls /sys/kernel/slab/slub_test/
aliases      destroy_by_rcu   objs_per_slab             reserved           total_objects
align        free_calls       order                     sanity_checks      trace
alloc_calls  hwcache_align    partial                   shrink             validate
cache_dma    min_partial      poison                    slabs
cpu_partial  objects          reclaim_account           slabs_cpu_partial
cpu_slabs    object_size      red_zone                  slab_size
ctor         objects_partial  remote_node_defrag_ratio  store_user
[root@centos7 mem_test]# 
```

# refercences

[linux-socfpga/extra_modules/](https://github.com/yeshen007/linux-socfpga/tree/32226c3069f827b405ac660f33b4865bdc75dfed/extra_modules)

[yeshen-md/高端内存使用.md](https://github.com/yeshen007/yeshen-md/blob/18a85f37e08c80fb282cdf3bba505eff97b3e3a0/%E9%AB%98%E7%AB%AF%E5%86%85%E5%AD%98%E4%BD%BF%E7%94%A8.md)

[双uboot](https://github.com/yeshen007/yeshen-pdf/blob/main/U-BOOT%E5%90%AF%E5%8A%A8%E8%BF%87%E7%A8%8B.pdf)
