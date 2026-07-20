# insmod  slab_ex_test.ko 

```
[root@centos7 test1]# insmod  slab_test.ko 
[root@centos7 test1]# dmesg | tail -n 10
[79496.585456] nod spanned pages = 2097152
[136473.427643] create mycache correctly
[136473.431300]  successfully created a object, kbuf_addr=0xffffa05e72e00000
[136473.438066] [+][kmem_cache]name : mycache, size : 150000
[136500.568899] destroyed a cache object
[136500.572574] destroyed mycache
[136740.123285] create mycache correctly
[136740.126942]  successfully created a object, kbuf_addr=0xffffa05e72e00000
[136740.133709] page_prt : ffff7fe8179cb800,  mem1_page: ffff7fe8179cb800
[136740.140207] [+][kmem_cache]name : mycache, size : 150000
[root@centos7 test1]# 
```

# references

[ex_kernel_module/m_modules/mm/](https://github.com/magnate3/ex_kernel_module/tree/master/m_modules/mm)