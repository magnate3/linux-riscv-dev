
# 释放page
内核中释放page，主要有如下方式：   
+ 1、free_pages，对应于alloc_pages，参数中需要带order。如果是用alloc_pages方式分配的高阶内存(大于1个page)，则必须用这种方式释放，并带上order参数。否则会出现内存泄露。因为：   
    1）alloc_pages分配高阶内存时，只会设置分配到第一个的page的引用计数(为1)，后面的page不会设置引用计数。    
    2）如果使用put_page方式释放内存，只会释放一个page，如果该page是以alloc_pages方式分配的高阶内存，那后面的page就泄露了，无法释放了。   
+ 2、put_page，减小指定page的引用计数，当引用计数减为0时，将其释放。      
+ 3、区别：   
     a、free_pages可释放大于一个page(order不为0)的内存块，put_page只能释放单个page   
     b、如果释放的内存块为一个page，两者都是将该page放到hot_cold缓存中(便于提升下次访问效率)   