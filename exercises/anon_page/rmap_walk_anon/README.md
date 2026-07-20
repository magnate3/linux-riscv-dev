
# ps -elf
```
[root@centos7 linux-4.14.115]# ps -elf | grep mmap_fork
0 S root       7673   6601  0  80   0 -  4138 n_tty_ Jul25 pts/1    00:00:00 ./mmap_fork
```
# integerSize addr
```
[root@centos7 anon_page]# ./mmap_fork 
mmap: Success
resust addr : 0xffffadf90000, and adf90000lx
integerSize addr : 0xffffe7ab6338, and e7ab6338lx
before wirte please findpage resust addr 

after wirte please findpage resust addr 
```

***the following is done before fork***

# echo 'findpage0xffffe7ab6338'  > /proc/mtest

```
[root@centos7 anon_page]# insmod  vma_test1.ko 
[root@centos7 anon_page]# echo 'findtask7673'  > /proc/mtest 
[root@centos7 anon_page]#  echo 'findpage0xffffe7ab6338'  > /proc/mtest
[root@centos7 anon_page]# dmesg | tail -n 10
[14598.763446] page  found  for 0xffffe7ab6338
[14598.767610] find  0xffffe7ab6338 to kernel address 0xffffa03feaa76338
[14598.774029] page is not  high 
[14598.777073] page is anonoyous and compare  rmap_walk_anon 
[14598.782539] vma 0xffffe7a90000-0xffffe7ac0000 flag 100173 , vma task comm: mmap_fork 
[14598.790334]  page_mapcount(page) went negative! (1)
[14598.795196]  page_mapcount(page) went negative! (1)
[14598.800054]  page->flags = 9fffff000004006c
[14598.804221]  page->count = 2
[14598.807090]  page->mapping = ffffa03fcbff3969
```
***rmap_walk_anon***

#  echo 'listvma'  > /proc/mtest

```
[14630.308362] mtest_dump_vma_list
[14630.311501] VMA 0x400000-0x410000 
[14630.311502] READ 
[14630.314889] EXEC 

[14630.320213] VMA 0x410000-0x420000 
[14630.320214] READ 

[14630.327003] VMA 0x420000-0x430000 
[14630.327004] WRITE 
[14630.330394] READ 

[14630.335799] VMA 0x291d0000-0x29200000 
[14630.335800] WRITE 
[14630.339533] READ 

[14630.344944] VMA 0xffffadf80000-0xffffadf90000 
[14630.344945] WRITE 
[14630.349367] READ 

[14630.354777] VMA 0xffffadf90000-0xffffbdf90000 
[14630.354777] WRITE 
[14630.359200] READ 

[14630.364609] VMA 0xffffbdf90000-0xffffbe100000 
[14630.364610] READ 
[14630.369032] EXEC 

[14630.374354] VMA 0xffffbe100000-0xffffbe110000 
[14630.374355] READ 

[14630.382183] VMA 0xffffbe110000-0xffffbe120000 
[14630.382183] WRITE 
[14630.386606] READ 

[14630.392014] VMA 0xffffbe120000-0xffffbe130000 
[14630.392015] WRITE 
[14630.396437] READ 

[14630.401846] VMA 0xffffbe130000-0xffffbe140000 
[14630.401847] READ 

[14630.409671] VMA 0xffffbe140000-0xffffbe150000 
[14630.409672] READ 
[14630.414099] EXEC 

[14630.419417] VMA 0xffffbe150000-0xffffbe170000 
[14630.419418] READ 
[14630.423845] EXEC 

[14630.429164] VMA 0xffffbe170000-0xffffbe180000 
[14630.429164] READ 

[14630.436993] VMA 0xffffbe180000-0xffffbe190000 
[14630.436993] WRITE 
[14630.441421] READ 

[14630.446826] VMA 0xffffe7a90000-0xffffe7ac0000   /////////////// this
[14630.446827] WRITE 
[14630.451253] READ 

[14630.456658]  vma count : 16 
```

#child process

## echo 'findtask18338'  > /proc/mtest 

```
[root@centos7 anon_page]#  ps -elf | grep mmap_fork
0 S root       7673   6601  0  80   0 -  4138 n_tty_ Jul25 pts/1    00:00:00 ./mmap_fork
1 S root      18338   7673  0  80   0 -  4138 wait_w 03:55 pts/1    00:00:00 ./mmap_fork
0 S root      18535   6582  0  80   0 -  1729 pipe_w 04:08 pts/0    00:00:00 grep --color=auto mmap_fork

```

## echo 'findpage0xffffe7ab6338'  > /proc/mtest

```
[root@centos7 anon_page]# echo 'findpage0xffffe7ab6338'  > /proc/mtest
[root@centos7 anon_page]# dmesg | tail -n 10
[16619.669348] page  found  for 0xffffe7ab6338
[16619.673512] find  0xffffe7ab6338 to kernel address 0xffffa03feaa76338
[16619.679930] page is not  high 
[16619.682972] page is anonoyous and compare  rmap_walk_anon 
[16619.688440] vma 0xffffe7a90000-0xffffe7ac0000 flag 100173 , vma task comm: mmap_fork  
[16619.696325]  page_mapcount(page) went negative! (1)
[16619.701184]  page_mapcount(page) went negative! (1)
[16619.706044]  page->flags = 9fffff000004006c
[16619.710211]  page->count = 4
[16619.713078]  page->mapping = ffffa03fcbffcad1
```
***page->mapping is differnet from parent***


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

[16812.466072] VMA 0xffffadf90000-0xffffbdf90000 
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
