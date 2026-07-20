
# ./mmap_test2 

```
[root@centos7 alloc_page]# ./mmap_test2 
before mmap ->please exec: free -m


p addr:  0xffff85480000 

```

# after mmap
```
make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
[root@centos7 vma]# ps -elf | grep  mmap_test2 
0 S root       7271   6699  0  80   0 -    39 wait_w 22:49 pts/1    00:00:00 ./mmap_test2
0 S root       7281   6680  0  80   0 -  1729 pipe_w 22:50 pts/0    00:00:00 grep --color=auto mmap_test2
[root@centos7 vma]# insmod  vma_test1.ko
[root@centos7 vma]# echo 'findtask7271'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
[  449.904745] mtest_write  ………..  
[  449.908497] The process pid 7271 
[  449.911797] the find_vpid result's count is: 9
[  449.916227] the find_vpid result's level is: 0
[  449.920650] The process is "mmap_test2" (pid 7271)
[root@centos7 vma]# echo 'va2phy0xffff867b0000'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
[  449.916227] the find_vpid result's level is: 0
[  449.920650] The process is "mmap_test2" (pid 7271)
[  504.671816] mtest_write  ………..  
[  504.675563] virt Address: 0x    ffff867b0000
[  504.679814] swap ID: 0x00000000
```

```
[root@centos7 vma]# echo 'findpage0xffff867b0000'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
[  504.675563] virt Address: 0x    ffff867b0000
[  504.679814] swap ID: 0x00000000
[  553.303352] mtest_write  ………..  
[  553.307107] mtest_write_val
[  553.309897] page not found  for 0xffff867b0000
[root@centos7 vma]# 
```
# before read
```
make[1]: Leaving directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
[root@centos7 vma]# ps -elf | grep  mmap_test2 
0 S root       7271   6699  0  80   0 -    39 wait_w 22:49 pts/1    00:00:00 ./mmap_test2
0 S root       7281   6680  0  80   0 -  1729 pipe_w 22:50 pts/0    00:00:00 grep --color=auto mmap_test2
[root@centos7 vma]# insmod  vma_test1.ko
[root@centos7 vma]# echo 'findtask7271'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
[  449.904745] mtest_write  ………..  
[  449.908497] The process pid 7271 
[  449.911797] the find_vpid result's count is: 9
[  449.916227] the find_vpid result's level is: 0
[  449.920650] The process is "mmap_test2" (pid 7271)
[root@centos7 vma]# echo 'va2phy0xffff867b0000'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
[  449.916227] the find_vpid result's level is: 0
[  449.920650] The process is "mmap_test2" (pid 7271)
[  504.671816] mtest_write  ………..  
[  504.675563] virt Address: 0x    ffff867b0000
[  504.679814] swap ID: 0x00000000
```

```
[root@centos7 vma]# echo 'findpage0xffff867b0000'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
[  504.675563] virt Address: 0x    ffff867b0000
[  504.679814] swap ID: 0x00000000
[  553.303352] mtest_write  ………..  
[  553.307107] mtest_write_val
[  553.309897] page not found  for 0xffff867b0000
[root@centos7 vma]# 
```

# after read
```
[root@centos7 vma]# echo 'va2phy0xffff867b0000'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
[  553.309897] page not found  for 0xffff867b0000
[  657.334719] mtest_write  ………..  
[  657.338460] virt Address: 0x    ffff867b0000
[  657.342723] Physical Address: 0x000f7000
pfn: 0x00f7
```

```
[root@centos7 vma]# echo 'findpage0xffff867b0000'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
pfn: 0x00f7
[  665.348470] mtest_write  ………..  
[  665.352206] mtest_write_val
[  665.355007] page  found  for 0xffff867b0000
[  665.359172] find  0xffff867b0000 to kernel address 0xffff800000f70000
[root@centos7 vma]# 
```
***you can see Physical Address: 0x000f7000, even cow will happen after write ***

#after write
```
[root@centos7 vma]# echo 'va2phy0xffff867b0000'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
[  665.359172] find  0xffff867b0000 to kernel address 0xffff800000f70000
[  730.743325] mtest_write  ………..  
[  730.747066] virt Address: 0x    ffff867b0000
[  730.751317] Physical Address: 0x205f8b83000
pfn: 0x205f8b83
```

```
[root@centos7 vma]# echo 'findpage0xffff867b0000'>  /proc/mtest
[root@centos7 vma]# dmesg |tail -n 5
pfn: 0x205f8b83
[  736.718462] mtest_write  ………..  
[  736.722206] mtest_write_val
[  736.724995] page  found  for 0xffff867b0000
[  736.729159] find  0xffff867b0000 to kernel address 0xffffa05f8b830000
[root@centos7 vma]# 
```

***you can see: before write Physical Address: 0x000f7000, and after write  Physical Address: 0x205f8b83000. because cow will happen after write ***

