# insmod  slab_ex_test.ko 

```
[root@centos7 test2]# insmod  slab_ex_test.ko 
[root@centos7 test2]# dmesg | tail -n 10
[76892.172168] Goodbye!
[77363.261552] mem_test: Unknown symbol migratetype_names (err 0)
[77363.267388] mem_test: Unknown symbol set_flag_main (err 0)
[77363.272878] mem_test: Unknown symbol lru_cache_add (err 0)
[77666.202696] mem_test: Unknown symbol get_lruvec (err 0)
[78279.081106] [+] mem_ptr1: ffffa05e67b83a80, mem_ptr2: ffffa05e67b85780
[78279.087617] [+] page: ffff7fe81799ee00 
[78279.091436] [+][kmem_cache] name: kmalloc-128, size: 128
[78279.096729] [+] page: ffff7fe81799ee00 
[78279.100547] [+][kmem_cache] name: kmalloc-128, size: 128
```

# references

[ex_kernel_module/m_modules/mm/](https://github.com/magnate3/ex_kernel_module/tree/master/m_modules/mm)