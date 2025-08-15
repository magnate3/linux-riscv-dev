
# before fork, mmap page is not anonoyous

```
[root@centos7 anon_page]# ./mmap_fork 
mmap: Success
resust addr : 0xffff78c40000, and 78c40000lx
integerSize addr : 0xffffea2a6a08, and ea2a6a08lx
before wirte please findpage resust addr 

after wirte please findpage resust addr 
```

```
[root@centos7 ~]#  echo 'findpage0xffffea2a6a08' >  /proc/mtest 
[root@centos7 ~]# dmesg | tail -n 10
[ 1471.347538] mtest_write_val
[ 1471.350320] page  found  for 0xffffea2a6a08
[ 1471.354496] find  0xffffea2a6a08 to kernel address 0xffffa03fe4056a08
[ 1471.360913] page is not  high 
[ 1471.363953] page is anonoyous
[ 1471.366993]  page_mapcount(page) went negative! (1)
[ 1471.371851]  page_mapcount(page) went negative! (1)
[ 1471.376706]  page->flags = 9fffff000004006c
[ 1471.380877]  page->count = 2
[ 1471.383744]  page->mapping = ffffa03fcae33759
```
***stack addr &integerSize is anonoyous***

````

[root@centos7 ~]#  echo 'findpage0xffff78c40000' >  /proc/mtest 
[root@centos7 ~]# dmesg | tail -n 10
[ 1506.654717] mtest_write  ………..  
[ 1506.658457] mtest_write_val
[ 1506.661250] page  found  for 0xffff78c40000
[ 1506.665414] find  0xffff78c40000 to kernel address 0xffffa03fe2cd0000
[ 1506.671832] page is not  high 
[ 1506.674873] page is not anonoyous or address_space  //////////////
[ 1506.679729]  page_mapcount(page) went negative! (1)
[ 1506.684588]  page->flags = 9fffff0000040018
[ 1506.688752]  page->count = 4
[ 1506.691623]  page->mapping = ffffa03fda2764e0
[root@centos7 ~]# 
```


# after fork 

```
[root@centos7 ~]#  echo 'findpage0xffff78c40000' >  /proc/mtest 
[root@centos7 ~]# dmesg | tail -n 10
[ 1722.341229] mtest_write  ………..  
[ 1722.344970] mtest_write_val
[ 1722.347762] page  found  for 0xffff78c40000
[ 1722.351926] find  0xffff78c40000 to kernel address 0xffffa03fe2cd0000
[ 1722.358342] page is not  high 
[ 1722.361383] page is not anonoyous or address_space 
[ 1722.366238]  page_mapcount(page) went negative! (2)
[ 1722.371098]  page->flags = 9fffff0000040038
[ 1722.375262]  page->count = 5
[ 1722.378136]  page->mapping = ffffa03fda2764e0
[root@centos7 ~]#  echo 'findpage0xffffea2a6a08' >  /proc/mtest 
[root@centos7 ~]# dmesg | tail -n 10
[ 1782.729932] mtest_write_val
[ 1782.732715] page  found  for 0xffffea2a6a08
[ 1782.736889] find  0xffffea2a6a08 to kernel address 0xffffa03ff0f96a08
[ 1782.743300] page is not  high 
[ 1782.746345] page is anonoyous 
[ 1782.749388]  page_mapcount(page) went negative! (1)
[ 1782.754243]  page_mapcount(page) went negative! (1)
[ 1782.759102]  page->flags = 9fffff000004006c
[ 1782.763266]  page->count = 2
[ 1782.766139]  page->mapping = ffffa03fcae33759
[root@centos7 ~]# 
```