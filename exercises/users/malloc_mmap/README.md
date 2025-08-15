# ./malloc_mmap 
```
[root@centos7 hugepage]# ./malloc_mmap 
count by itself:
        heap_malloc_total=8323072 heap_free_total=0 heap_in_use=8323072
        mmap_total=12054528 mmap_count=72
count by mallinfo:
        heap_malloc_total=8519680 heap_free_total=63456 heap_in_use=8456224
        mmap_total=14483456 mmap_count=71
from malloc_stats:
Arena 0:
system bytes     =    8519680
in use bytes     =    8456224
Total (incl. mmap):
system bytes     =   23003136
in use bytes     =   22939680
max mmap regions =         71
max mmap bytes   =   14483456
**************************************** 
Total non-mmapped bytes (arena):       8519680
# of free chunks (ordblks):            1
# of free fastbin blocks (smblks):     0
# of mapped regions (hblks):           71
Bytes in mapped regions (hblkhd):      14483456
Max. total allocated space (usmblks):  0
Free bytes held in fastbins (fsmblks): 0
Total allocated space (uordblks):      8456224
Total free space (fordblks):           63456
Topmost releasable block (keepcost):   63456

after free
count by itself:
        heap_malloc_total=8323072 heap_free_total=4194304 heap_in_use=4128768
        mmap_total=6008832 mmap_count=36
count by mallinfo:
        heap_malloc_total=8519680 heap_free_total=4258784 heap_in_use=4260896
        mmap_total=7143424 mmap_count=35
from malloc_stats:
Arena 0:
system bytes     =    8519680
in use bytes     =    4260896
Total (incl. mmap):
system bytes     =   15663104
in use bytes     =   11404320
max mmap regions =         71
max mmap bytes   =   14483456
**************************************** 
Total non-mmapped bytes (arena):       8519680
# of free chunks (ordblks):            65
# of free fastbin blocks (smblks):     0
# of mapped regions (hblks):           35
Bytes in mapped regions (hblkhd):      7143424
Max. total allocated space (usmblks):  0
Free bytes held in fastbins (fsmblks): 0
Total allocated space (uordblks):      4260896
Total free space (fordblks):           4258784
Topmost releasable block (keepcost):   63456
```