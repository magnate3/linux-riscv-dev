# resust addr
```
[root@centos7 anon_page]# ./mmap_fork 
mmap: Success
resust addr : 0xffffadf90000, and adf90000lx
integerSize addr : 0xffffe7ab6338, and e7ab6338lx
before wirte please findpage resust addr 

after wirte please findpage resust addr 
```

# mmap and after write resust addr

```
[root@centos7 anon_page]# rmmod  vma_test1.ko 
[root@centos7 anon_page]# insmod  vma_test1.ko 
[root@centos7 anon_page]# echo 'findtask7673'  > /proc/mtest 
[root@centos7 anon_page]#  echo 'findpage0xffffadf90000'  > /proc/mtest
[root@centos7 anon_page]# dmesg | tail -n 10
[15758.318396] mtest_write_val
[15758.321203] page  found  for 0xffffadf90000
[15758.325367] find  0xffffadf90000 to kernel address 0xffffa03f97b20000
[15758.331783] page is not  high 
[15758.334825] page is file and compare rmap_walk_file
[15758.339682] vma 0xffffadf90000-0xffffbdf90000 flag fb , vma task comm: mmap_fork 
[15758.347136]  page_mapcount(page) went negative! (1)
[15758.351996]  page->flags = 9fffff0000040038
[15758.356160]  page->count = 4
[15758.359027]  page->mapping = ffffa03fd70f2860
```

# after fork

```
[root@centos7 anon_page]#  echo 'findpage0xffffadf90000'  > /proc/mtest
[root@centos7 anon_page]# dmesg | tail -n 10
[15805.450355] page  found  for 0xffffadf90000
[15805.454519] find  0xffffadf90000 to kernel address 0xffffa03f97b20000
[15805.460937] page is not  high 
[15805.463978] page is file and compare rmap_walk_file
[15805.468834] vma 0xffffadf90000-0xffffbdf90000 flag fb , vma task comm: mmap_fork 
[15805.476287] vma 0xffffadf90000-0xffffbdf90000 flag fb , vma task comm: mmap_fork 
[15805.483738]  page_mapcount(page) went negative! (2)
[15805.488594]  page->flags = 9fffff0000040038
[15805.492763]  page->count = 6
[15805.495631]  page->mapping = ffffa03fd70f2860
```
*count change from 4 to 6*


```
[root@centos7 anon_page]# dmesg | tail -n 70
[15805.468834] vma 0xffffadf90000-0xffffbdf90000 flag fb , vma task comm: mmap_fork 
[15805.476287] vma 0xffffadf90000-0xffffbdf90000 flag fb , vma task comm: mmap_fork 
[15805.483738]  page_mapcount(page) went negative! (2)
[15805.488594]  page->flags = 9fffff0000040038
[15805.492763]  page->count = 6
[15805.495631]  page->mapping = ffffa03fd70f2860
[16130.738249] mtest_write  ………..  
[16130.741989] The current process is mmap_fork
[16130.746253] mtest_dump_vma_list
[16130.749386] VMA 0x400000-0x410000 
[16130.749387] READ 
[16130.752774] EXEC 

[16130.758092] VMA 0x410000-0x420000 
[16130.758092] READ 

[16130.764885] VMA 0x420000-0x430000 
[16130.764886] WRITE 
[16130.768273] READ 

[16130.773682] VMA 0x291d0000-0x29200000 
[16130.773683] WRITE 
[16130.777416] READ 

[16130.782820] VMA 0xffffadf80000-0xffffadf90000 
[16130.782821] WRITE 
[16130.787248] READ 

[16130.792653] VMA 0xffffadf90000-0xffffbdf90000  ///////////////////////
[16130.792653] WRITE 
[16130.797080] READ 

[16130.802484] VMA 0xffffbdf90000-0xffffbe100000 
[16130.802485] READ 
[16130.806911] EXEC 

[16130.812228] VMA 0xffffbe100000-0xffffbe110000 
[16130.812229] READ 

[16130.820057] VMA 0xffffbe110000-0xffffbe120000 
[16130.820057] WRITE 
[16130.824483] READ 

[16130.829887] VMA 0xffffbe120000-0xffffbe130000 
[16130.829888] WRITE 
[16130.834314] READ 

[16130.839719] VMA 0xffffbe130000-0xffffbe140000 
[16130.839719] READ 

[16130.847547] VMA 0xffffbe140000-0xffffbe150000 
[16130.847547] READ 
[16130.851969] EXEC 

[16130.857291] VMA 0xffffbe150000-0xffffbe170000 
[16130.857292] READ 
[16130.861714] EXEC 

[16130.867035] VMA 0xffffbe170000-0xffffbe180000 
[16130.867035] READ 

[16130.874863] VMA 0xffffbe180000-0xffffbe190000 
[16130.874864] WRITE 
[16130.879287] READ 

[16130.884695] VMA 0xffffe7a90000-0xffffe7ac0000 
[16130.884696] WRITE 
[16130.889119] READ 

[16130.894527]  vma count : 16 
```

#child process

## echo 'findtask18338'  > /proc/mtest 

```
[root@centos7 anon_page]#  ps -elf | grep mmap_fork
0 S root       7673   6601  0  80   0 -  4138 n_tty_ Jul25 pts/1    00:00:00 ./mmap_fork
1 S root      18338   7673  0  80   0 -  4138 wait_w 03:55 pts/1    00:00:00 ./mmap_fork
0 S root      18535   6582  0  80   0 -  1729 pipe_w 04:08 pts/0    00:00:00 grep --color=auto mmap_fork

```

##  echo 'findpage0xffffadf90000'  > /proc/mtest

```
[root@centos7 anon_page]# echo 'findpage0xffffadf90000'  > /proc/mtest
[root@centos7 anon_page]# dmesg | tail -n 10
[17069.410449] page  found  for 0xffffadf90000
[17069.414613] find  0xffffadf90000 to kernel address 0xffffa03f97b20000
[17069.421033] page is not  high 
[17069.424076] page is file and compare rmap_walk_file
[17069.428938] vma 0xffffadf90000-0xffffbdf90000 flag fb , vma task comm: mmap_fork  //why two 
[17069.436387] vma 0xffffadf90000-0xffffbdf90000 flag fb , vma task comm: mmap_fork  //why two 
[17069.443841]  page_mapcount(page) went negative! (2)
[17069.448703]  page->flags = 9fffff0000040038
[17069.452866]  page->count = 8
[17069.455733]  page->mapping = ffffa03fd70f2860
```
***page->mapping is  the same as   parent***



## echo 'listvma'  > /proc/mtest 
```
[root@centos7 anon_page]# echo 'listvma'  > /proc/mtest 
[root@centos7 anon_page]# dmesg | tail -n 70
[16619.688440] vma 0xffffe7a90000-0xffffe7ac0000 flag 100173 , vma task comm: mmap_fork  
[16619.696325]  page_mapcount(page) went negative! (1)
[16619.701184]  page_mapcount(page) went negative! (1)
[16619.706044]  page->flags = 9fffff000004006c
[16619.710211]  page->count = 4
[16619.713078]  page->mapping = ffffa03fcbffcad1
[16812.411676] mtest_write  ………..  
[16812.415413] The current process is mmap_fork
[16812.419664] mtest_dump_vma_list
[16812.422805] VMA 0x400000-0x410000 
[16812.422806] READ 
[16812.426193] EXEC 

[16812.431514] VMA 0x410000-0x420000 
[16812.431515] READ 

[16812.438304] VMA 0x420000-0x430000 
[16812.438304] WRITE 
[16812.441695] READ 

[16812.447100] VMA 0x291d0000-0x29200000 
[16812.447100] WRITE 
[16812.450830] READ 

[16812.456239] VMA 0xffffadf80000-0xffffadf90000 
[16812.456239] WRITE 
[16812.460661] READ 

[16812.466072] VMA 0xffffadf90000-0xffffbdf90000  //////
[16812.466073] WRITE 
[16812.470496] READ 

[16812.475906] VMA 0xffffbdf90000-0xffffbe100000 
[16812.475907] READ 
[16812.480329] EXEC 

[16812.485650] VMA 0xffffbe100000-0xffffbe110000 
[16812.485651] READ 

[16812.493478] VMA 0xffffbe110000-0xffffbe120000 
[16812.493479] WRITE 
[16812.497901] READ 

[16812.503309] VMA 0xffffbe120000-0xffffbe130000 
[16812.503310] WRITE 
[16812.507732] READ 

[16812.513140] VMA 0xffffbe130000-0xffffbe140000 
[16812.513141] READ 

[16812.520964] VMA 0xffffbe140000-0xffffbe150000 
[16812.520965] READ 
[16812.525391] EXEC 

[16812.530709] VMA 0xffffbe150000-0xffffbe170000 
[16812.530709] READ 
[16812.535136] EXEC 

[16812.540453] VMA 0xffffbe170000-0xffffbe180000 
[16812.540454] READ 

[16812.548282] VMA 0xffffbe180000-0xffffbe190000 
[16812.548283] WRITE 
[16812.552709] READ 

[16812.558113] VMA 0xffffe7a90000-0xffffe7ac0000 //////////////////this
[16812.558114] WRITE 
[16812.562540] READ 

[16812.567945]  vma count : 16 
```

# difference of parent and child


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/anon_page/walk_parent.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/anon_page/walk_child.png)