
# insmod  mmap-test1.ko 
 
 ***一次remap***
 
 ```
 [root@centos7 kernel]# insmod  mmap-test1.ko 
[root@centos7 kernel]# 
[ 1077.719802] ******* alloc mem ffffa05fd9940000 ********** 
[ 1080.453077] *********** my_mmap *******vma->vm_start:  ffffae4f0000 , len :30000 
[root@centos7 kernel]
 ```
 
 ```
 root@centos7 user]# ./test 
addr: 0xffffae4f0000  //// equals vma->vm_start

Write/Read test ...
0x66616365
0x66616365
0x66616365
[root@centos7 user]# 
 ```
 
 
#  insmod  mmap-test2.ko 

 (1) 驱动初始化时预先分配好 3 个 PAGE。
 (2)上层执行 mmap 系统调用，底层驱动在 mmap 回调函数中不建立映射关系，而是将本地实现的 vm_ops 挂接到进程的 vma->vm_ops 指针上，然后函数返回。
 (3)上层获取到一个未经映射的进程地址空间，并对其进行内存读写操作，导致触发缺页异常。缺页异常最终会调用前面挂接的 vm_ops->fault() 回调接口，在该接回调中通过 vm_insert_page() 建立物理内存与用户地址空间的映射关系。
异常返回后，应用程序就可以继续之前被中断的读写操作了
 (4) 一次分配连续地址，kzalloc(PAGE_SIZE * 3, GFP_KERNEL);
 
 
 
***发生多次fault***



```
[ 1840.642501] ******* alloc mem2 ffffa03f02e80000 ********** 
[ 1843.921380] *********** my_fault2 *******vma->vm_start:  ffffad440000 , offset :0 
[ 1843.928666] *********** my_fault2 *******vma->vm_start:  ffffad450000 , offset :10000 
[ 1843.936292] *********** my_fault2 *******vma->vm_start:  ffffad460000 , offset :20000 
```

```
[root@centos7 user]# ./test 
addr: 0xffffad440000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
```



#  insmod   mmap-test3.ko

```
[10986.567927] ---[ end trace 63aa51e5eb532fbe ]---
[12934.561016] ******* mmap 3 ********** 
[12938.351397] vma->vm_end ffffa0650000 vm_start ffffa0620000 len 30000 
[12938.357822] *********** my_fault3 *******vma->vm_start:  ffffa0620000 , pgoff :0 , page addr: ffffa03fd9870000 
[12938.367876] vma->vm_end ffffa0650000 vm_start ffffa0620000 len 30000 
[12938.374294] *********** my_fault3 *******vma->vm_start:  ffffa0630000 , pgoff :1 , page addr: ffffa03fd8960000 
[12938.384339] vma->vm_end ffffa0650000 vm_start ffffa0620000 len 30000 
[12938.390749] *********** my_fault3 *******vma->vm_start:  ffffa0640000 , pgoff :2 , page addr: ffffa03fd61b0000 
[root@centos7 kernel]# 
[root@centos7 kernel]# 
```

```
[root@centos7 user]# ./test 
[root@centos7 user]# ./test 
addr: 0xffffa0620000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
```

# insmod mmap-test4.ko

```
[12175.589806] ******* alloc mem4 ffffa05fd1c40000 ********** 
[12178.365479] *********** my_fault4 *******vma->vm_start:  ffff89980000 ,  pgoff:  0
[12178.373037] *********** my_fault4 *******vma->vm_start:  ffff89990000 ,  pgoff:  1
[12178.380578] *********** my_fault4 *******vma->vm_start:  ffff899a0000 ,  pgoff:  2
[root@centos7 kernel]# 
```

```
[root@centos7 user]# ./test 
addr: 0xffff89980000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
```


# references

[DRM 驱动 mmap 详解：（一）预备知识](https://blog.csdn.net/hexiaolong2009/article/details/107592704)