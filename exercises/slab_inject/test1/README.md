

# insmod  slab_test.ko 
```
[root@centos7 test1]# insmod  slab_test.ko 
[root@centos7 test1]# dmesg | tail -n 10
[58059.651743] kmem_cache name:kmalloc-256 
[58059.655647] kmem_cache name:kmalloc-128 
[58059.659558] kmem_cache name:kmem_cache_node 
[58059.663809] Cache: kmem_cache_node
[58059.667196] Object size: 64
[58059.669982] Aligned size: 128
[58059.672937] Can fit 512 objects to page (page size=65536)
[58059.678319] kmem_cache name: kmem_cache_node, objs[i]->name: other 
[59040.216668] create mycache correctly
[59040.220237]  successfully created a object, kbuf_addr=0xffffa03f60800000
[root@centos7 test1]# rmmod  slab_test.ko 
[root@centos7 test1]# dmesg | tail -n 10
[58059.659558] kmem_cache name:kmem_cache_node 
[58059.663809] Cache: kmem_cache_node
[58059.667196] Object size: 64
[58059.669982] Aligned size: 128
[58059.672937] Can fit 512 objects to page (page size=65536)
[58059.678319] kmem_cache name: kmem_cache_node, objs[i]->name: other 
[59040.216668] create mycache correctly
[59040.220237]  successfully created a object, kbuf_addr=0xffffa03f60800000
[59066.887960] destroyed a cache object
[59066.891563] destroyed mycache
[root@centos7 test1]# 
```