![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/slab_inject/slab1.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/slab_inject/slab2.png)

# SLAB injection technique
Inject pages to Linux SLAB caches, which might be really useful when dealing with embedded devices having to allocate a lot of objects in atomic context.

Use this to inject pages to a cache:
```
make
sudo ./inject_page.sh cachename [number of pages]
```

To check existing caches:
```
sudo cat /proc/slabinfo
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/slab_inject/kmem_cache.png)

# ./inject_page.sh kmem_cache
```
[root@centos7 slab_inject]#  ./inject_page.sh kmem_cache
[root@centos7 slab_inject]# dmesg | tail -n 10
[54237.942582] PAGE_MASK = 0xffffffffffff0000
[54237.946664] PUD_MASK = 0xfffffc0000000000
[54237.950656] PMD_MASK= 0xffffffffe0000000
[54237.954570] create the filename mtest mtest_init sucess  
[55997.021429] slab_inject: module license 'MIT' taints kernel.
[55997.027075] Disabling lock debugging due to kernel taint
[55997.032930] Cache: kmem_cache
[55997.035885] Object size: 360
[55997.038752] Aligned size: 384
[55997.041706] Can fit 170 objects to page (page size=65536)
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/slab_inject/kmem1.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/slab_inject/node2.png)

#references


linux内核开发第22讲：页框和伙伴算法以及slab机制
