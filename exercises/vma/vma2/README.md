# insmod  vma_test1.ko 
```
[root@centos7 vma]# insmod  vma_test1.ko 
[root@centos7 vma]# ps -elf | grep test
0 S root       6502   6074  0  80   0 -    39 wait_w 04:37 pts/0    00:00:00 ./test
0 S root       6814   6470  0  80   0 -  1729 pipe_w 04:40 pts/1    00:00:00 grep --color=auto test
[root@centos7 vma]# echo 'findtask6502'>  /proc/mtest 
[root@centos7 vma]# echo 'listvma' >  /proc/mtest // before mmap
[root@centos7 vma]# echo 'listvma' >  /proc/mtest  // after mmap
 
```

## mmap addr 

```
[root@centos7 vma]# ./test 
before mmap ->please exec: free -m


p addr:  0xffffa51d0000 

after mmap ->please exec: free -m

before read...
```
## before mmap

```
[  394.439277] mtest_write  ………..  
[  394.443014] The process pid 6502 
[  394.446324] the find_vpid result's count is: 9
[  394.450749] the find_vpid result's level is: 0
[  394.455178] The process is "test" (pid 6502)
[  410.664513] mtest_write  ………..  
[  410.668250] The current process is test
[  410.672068] mtest_dump_vma_list
[  410.675202] VMA 0x400000-0x410000 
[  410.675203] READ 
[  410.678589] EXEC 

[  410.683908] VMA 0x410000-0x420000 
[  410.683908] READ 

[  410.690701] VMA 0x420000-0x430000 
[  410.690702] WRITE 
[  410.694087] READ 

[  410.699497] VMA 0xffffaf1d0000-0xffffaf1e0000 
[  410.699498] WRITE 
[  410.703920] READ 

[  410.709331] VMA 0xffffaf1e0000-0xffffaf350000 
[  410.709332] READ 
[  410.713754] EXEC 

[  410.719078] VMA 0xffffaf350000-0xffffaf360000 
[  410.719079] READ 

[  410.726908] VMA 0xffffaf360000-0xffffaf370000 
[  410.726908] WRITE 
[  410.731331] READ 

[  410.736741] VMA 0xffffaf370000-0xffffaf380000 
[  410.736741] WRITE 
[  410.741164] READ 

[  410.746573] VMA 0xffffaf380000-0xffffaf390000 
[  410.746574] READ 

[  410.754397] VMA 0xffffaf390000-0xffffaf3a0000 
[  410.754398] READ 
[  410.758825] EXEC 

[  410.764144] VMA 0xffffaf3a0000-0xffffaf3c0000 
[  410.764145] READ 
[  410.768572] EXEC 

[  410.773891] VMA 0xffffaf3c0000-0xffffaf3d0000 
[  410.773892] READ 

[  410.781721] VMA 0xffffaf3d0000-0xffffaf3e0000 
[  410.781721] WRITE 
[  410.786148] READ 

[  410.791553] VMA 0xfffff0780000-0xfffff07b0000 
[  410.791554] WRITE 
[  410.795981] READ 

[  410.801386]  vma count : 14 
```

## after mmap


```
[root@centos7 ~]# dmesg | tail -n 50

[  559.049951] VMA 0x410000-0x420000 
[  559.049952] READ 

[  559.056743] VMA 0x420000-0x430000 
[  559.056744] WRITE 
[  559.060129] READ 

[  559.065539] VMA 0xffffa51d0000-0xffffaf1e0000 
[  559.065540] WRITE 
[  559.069962] READ 

[  559.075372] VMA 0xffffaf1e0000-0xffffaf350000 
[  559.075372] READ 
[  559.079795] EXEC 

[  559.085118] VMA 0xffffaf350000-0xffffaf360000 
[  559.085119] READ 

[  559.092948] VMA 0xffffaf360000-0xffffaf370000 
[  559.092948] WRITE 
[  559.097371] READ 

[  559.102781] VMA 0xffffaf370000-0xffffaf380000 
[  559.102781] WRITE 
[  559.107204] READ 

[  559.112614] VMA 0xffffaf380000-0xffffaf390000 
[  559.112614] READ 

[  559.120438] VMA 0xffffaf390000-0xffffaf3a0000 
[  559.120439] READ 
[  559.124867] EXEC 

[  559.130186] VMA 0xffffaf3a0000-0xffffaf3c0000 
[  559.130187] READ 
[  559.134615] EXEC 

[  559.139934] VMA 0xffffaf3c0000-0xffffaf3d0000 
[  559.139934] READ 

[  559.147763] VMA 0xffffaf3d0000-0xffffaf3e0000 
[  559.147764] WRITE 
[  559.152192] READ 

[  559.157596] VMA 0xfffff0780000-0xfffff07b0000 
[  559.157597] WRITE 
[  559.162026] READ 

[  559.167430]  vma count : 14 
```

***compare before count = 14  and after count =14***

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vma7.png)


# mmap not has page
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/vma/pic/vma6.png)
