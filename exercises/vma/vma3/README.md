
# ./mmap_test2 

```
[root@centos7 alloc_page]# ./mmap_test2 
before mmap ->please exec: free -m


p addr:  0xffff85480000 

```

# after mmap
```
[root@centos7 vma]# insmod  vma_test1.ko 
[root@centos7 vma]# ps -elf | grep  mmap_test2 
0 S root      10659   8679  0  80   0 -  2599 wait_w 05:47 pts/2    00:00:00 ./mmap_test2
0 S root      11231   6665  0  80   0 -  1729 pipe_w 05:51 pts/0    00:00:00 grep --color=auto mmap_test2
[root@centos7 vma]# echo 'findtask10659'>  /proc/mtest
[root@centos7 vma]# echo 'va2phy0xffff85480000'>  /proc/mtest
[root@centos7 vma]# dmesg | tail -n 10
[ 5877.118815] exit the module……mtest_exit 
[ 5886.806368] create the filename mtest mtest_init sucess  
[ 5901.466961] mtest_write  ………..  
[ 5901.470709] The process pid 10659 
[ 5901.474095] the find_vpid result's count is: 9
[ 5901.478525] the find_vpid result's level is: 0
[ 5901.482949] The process is "mmap_test2" (pid 10659)
[ 5908.818642] mtest_write  ………..  
[ 5908.822378] virt Address: 0x    ffff85480000
[ 5908.826628] swap ID: 0x00000000
```
# before read
```
[root@centos7 vma]# ps -elf | grep  mmap_test2 
0 S root      11363   8679  0  80   0 -  2599 wait_w 06:01 pts/2    00:00:00 ./mmap_test2
0 S root      11366   6665  0  80   0 -  1729 pipe_w 06:01 pts/0    00:00:00 grep --color=auto mmap_test2
[root@centos7 vma]# echo 'findtask11363'>  /proc/mtest
[root@centos7 vma]# echo 'va2phy0xffff8e4b0000'>  /proc/mtest
[root@centos7 vma]# dmesg | tail -n 10
[ 6411.520126] Physical Address: 0x000f7000
pfn: 0x00f7
[ 6490.671424] mtest_write  ………..  
[ 6490.675166] The process pid 11363 
[ 6490.678564] the find_vpid result's count is: 9
[ 6490.682988] the find_vpid result's level is: 0
[ 6490.687411] The process is "mmap_test2" (pid 11363)
[ 6517.360548] mtest_write  ………..  
[ 6517.364286] virt Address: 0x    ffff8e4b0000
[ 6517.368547] swap ID: 0x00000000
```

```
[root@centos7 vma]# echo 'findpage0xffff8e4b0000'>  /proc/mtest
[root@centos7 vma]# dmesg | tail -n 10
[ 6490.675166] The process pid 11363 
[ 6490.678564] the find_vpid result's count is: 9
[ 6490.682988] the find_vpid result's level is: 0
[ 6490.687411] The process is "mmap_test2" (pid 11363)
[ 6517.360548] mtest_write  ………..  
[ 6517.364286] virt Address: 0x    ffff8e4b0000
[ 6517.368547] swap ID: 0x00000000
[ 6541.810200] mtest_write  ………..  
[ 6541.813935] mtest_write_val
[ 6541.816718] page not found  for 0xffff8e4b0000
[root@centos7 vma]# 
```

# after read

```
[root@centos7 vma]# echo 'va2phy0xffff8e4b0000'>  /proc/mtest
[root@centos7 vma]# dmesg | tail -n 10
[ 6517.360548] mtest_write  ………..  
[ 6517.364286] virt Address: 0x    ffff8e4b0000
[ 6517.368547] swap ID: 0x00000000
[ 6541.810200] mtest_write  ………..  
[ 6541.813935] mtest_write_val
[ 6541.816718] page not found  for 0xffff8e4b0000
[ 6594.614462] mtest_write  ………..  
[ 6594.618204] virt Address: 0x    ffff8e4b0000
[ 6594.622454] Physical Address: 0x000f7000
pfn: 0x00f7
```

```
[root@centos7 vma]# echo 'findpage0xffff8e4b0000'>  /proc/mtest
[root@centos7 vma]# dmesg | tail -n 10
[ 6541.813935] mtest_write_val
[ 6541.816718] page not found  for 0xffff8e4b0000
[ 6594.614462] mtest_write  ………..  
[ 6594.618204] virt Address: 0x    ffff8e4b0000
[ 6594.622454] Physical Address: 0x000f7000
pfn: 0x00f7
[ 6612.369039] mtest_write  ………..  
[ 6612.372775] mtest_write_val
[ 6612.375557] page  found  for 0xffff8e4b0000
[ 6612.379736] find  0xffff8e4b0000 to kernel address 0xffff800000f70000
[root@centos7 vma]# 
```
***you can see Physical Address: 0x000f7000, even cow will happen after write ***

#after write

```
[root@centos7 vma]# echo 'va2phy0xffff8e4b0000'>  /proc/mtest
[root@centos7 vma]# dmesg | tail -n 10
[ 6594.622454] Physical Address: 0x000f7000
pfn: 0x00f7
[ 6612.369039] mtest_write  ………..  
[ 6612.372775] mtest_write_val
[ 6612.375557] page  found  for 0xffff8e4b0000
[ 6612.379736] find  0xffff8e4b0000 to kernel address 0xffff800000f70000
[ 6738.391246] mtest_write  ………..  
[ 6738.394995] virt Address: 0x    ffff8e4b0000
[ 6738.399245] Physical Address: 0x000f7000
pfn: 0x00f7
```

```

[root@centos7 vma]# echo 'findpage0xffff8e4b0000'>  /proc/mtest
[root@centos7 vma]# dmesg | tail -n 10
[ 6612.375557] page  found  for 0xffff8e4b0000
[ 6612.379736] find  0xffff8e4b0000 to kernel address 0xffff800000f70000
[ 6738.391246] mtest_write  ………..  
[ 6738.394995] virt Address: 0x    ffff8e4b0000
[ 6738.399245] Physical Address: 0x000f7000
pfn: 0x00f7
[ 6744.207809] mtest_write  ………..  
[ 6744.211544] mtest_write_val
[ 6744.214346] page  found  for 0xffff8e4b0000
[ 6744.218511] find  0xffff8e4b0000 to kernel address 0xffff800000f70000
[root@centos7 vma]# 
```


