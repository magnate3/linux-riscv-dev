
# PAGE_SIZE
```
[root@centos7 alloc_page]# getconf PAGE_SIZE
65536
[root@centos7 alloc_page]# 
```

# insmod  alloc_page_test.ko
```
[root@centos7 alloc_page]# insmod  alloc_page_test.ko 
[root@centos7 alloc_page]# dmesg | tail -n 10
[  635.783243] exit
[  637.790246] sizeof(struct page): 64
[  637.790249] virt : 0xffffa05fc7100000, phy:0x205fc7100000, page:0xffff7fe817f1c400, pfn: 543147792
[  637.802661] virt : 0xffffa05fc7110000, phy:0x205fc7110000, page:0xffff7fe817f1c440, pfn: 543147793
[  637.811584] virt : 0xffffa05fc7120000, phy:0x205fc7120000, page:0xffff7fe817f1c480, pfn: 543147794
[  637.820501] virt : 0xffffa05fc7130000, phy:0x205fc7130000, page:0xffff7fe817f1c4c0, pfn: 543147795
[root@centos7 alloc_page]# rmmod  alloc_page_test
[root@centos7 alloc_page]# 
```
virt地址相差 0x10000=65536;phys物理地址相差0x10000；page指针相差0x40;struct page 大小sizeof(struct page)正是0x40；页帧pfn相差1;

# references

[page_to_pfn 、virt_to_page、 virt_to_phys、page、页帧pfn、内核虚拟地址、物理内存地址linux内核源码详解](https://blog.csdn.net/hu1610552336/article/details/113083454)