
# insmod  rb_tree_test.ko 

```
[932543.992759] RB Tree test start!
[932543.996175] 1000 INSERT TIME : 72351ns
[932544.000080] 1000 SEARCH TIME : 62762ns
[932544.004014] 1000 DELETE TIME : 87462ns
[932544.009649] 10000 INSERT TIME : 1015592ns
[932544.014826] 10000 SEARCH TIME : 853995ns
[932544.020006] 10000 DELETE TIME : 950699ns
[932544.067313] 100000 INSERT TIME : 38453792ns
[932544.091133] 100000 SEARCH TIME : 17170216ns
[932544.110375] 100000 DELETE TIME : 12597971ns
```

#   insmod  rb_tree_test2.ko 


```
********** rbtree_fifo testing!! **********
[  600.794142] rt_priority: 0
[  600.796842] scheduling policy: 0
[  600.800057] first vruntime: 2297510442
[  600.804445] 
////////// insert //////////
[  600.809891] insert(1000 entries): 0.000064822 secs
[  600.815490] insert(10000 entries): 0.000830094 secs
[  600.835104] insert(100000 entries): 0.014753340 secs
[  600.840050] rt_priority(after insert): 0
[  600.843954] 
////////// search //////////
[  600.849347] search(1000 entries): 0.000016311 secs
[  600.854266] search(10000 entries): 0.000151073 secs
[  600.861323] search(100000 entries): 0.002196117 secs
[  600.866270] rt_priority(after search): 0
[  600.870175] 
////////// delete //////////
[  600.875584] delete(1000 entries): 0.000037441 secs
[  600.880721] delete(10000 entries): 0.000362836 secs
[  600.890357] delete(100000 entries): 0.004780261 secs
[  600.895299] rt_priority(after delete): 0
[  600.899209] second vruntime: 2399877652

```


# insmod  rb_tree_cached_test4.ko 

```
[root@centos7 rb_tree]# insmod  rb_tree_cached_test4.ko 
[root@centos7 rb_tree]# rmmod  rb_tree_cached_test4.ko 
[root@centos7 rb_tree]# dmesg | tail -n 30

[ 3695.999265] 
********** rbtree_fifo testing!! **********
[ 3696.005944] rt_priority: 0
[ 3696.008639] scheduling policy: 0
[ 3696.011852] first vruntime: 3833543427
[ 3696.015590] value:  -2135644201  
 
[ 3696.015592] value:  -1501900621  
 
[ 3696.020447] value:  -917625545  
 
[ 3696.025307] value:  -554721006  
 
[ 3696.030076] value:  -134444930  
 
[ 3696.034847] value:  -41069305  
 
[ 3696.039617] value:  -25386540  
 
[ 3696.044301] value:  79275557  
 
[ 3696.048982] value:  600473977  
 
[ 3696.053581] value:  864658644  
 
[ 3696.058262] second vruntime: 3881183377

[ 3704.766153] 
Bye module
```