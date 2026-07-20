
Linux系统在长时间运行后，连续的大块物理内存会被分解为小块的物理内存。此时，系统中部署的业务如果需要连续的大块内存，则系统会先进入耗时较长的内存规整流程，进而会引起系统性能抖动。一般情况下，内存碎片化会产生类似于如下所示的内核堆栈信息。


```
0xffffffff8118f9cb compaction_alloc  ([kernel.kallsyms])
0xffffffff811c88a9 migrate_pages  ([kernel.kallsyms])
0xffffffff811901ee compact_zone  ([kernel.kallsyms])
0xffffffff8119041b compact_zone_order  ([kernel.kallsyms])
0xffffffff81190735 try_to_compact_pages  ([kernel.kallsyms])
0xffffffff81631cb4 __alloc_pages_direct_compact  ([kernel.kallsyms])
0xffffffff811741d5 __alloc_pages_nodemask  ([kernel.kallsyms])
0xffffffff811b5a79 alloc_pages_current  ([kernel.kallsyms])
0xffffffff811c0005 new_slab  ([kernel.kallsyms])
0xffffffff81633848 __slab_alloc  ([kernel.kallsyms])
0xffffffff811c5291 __kmalloc_node_track_caller  ([kernel.kallsyms])
0xffffffff8151a8c1 __kmalloc_reserve.isra.30  ([kernel.kallsyms])
0xffffffff8151b7cd alloc_sib  ([kernel.kallsyms])
0xffffffff815779e9 sk_stream_alloc_skb  ([kernel.kallsyms])
0xffffffff8157872d tcp_sendmsg  ([kernel.kallsyms])
0xffffffff815a26b4 inet_sendmsg  ([kernel.kallsyms])
0xffffffff81511017 sock_aio_write  ([kernel.kallsyms])
0xffffffff811df729 do_sync_readv_writev  ([kernel.kallsyms])
0xffffffff811e0cfe do_readv_writev  ([kernel.kallsyms])
```


# references


[Linux内存管理：伙伴系统实现](https://zhuanlan.zhihu.com/p/258921453)   