

# test3

## insmod  mm_test.ko  processid=30844

```
[root@centos7 test2]# ps -elf | grep mmap_fork
0 S root      30844  28539  0  80   0 -  4138 n_tty_ 04:13 pts/0    00:00:00 ./mmap_fork
1 S root      30845  30844  0  80   0 -  4138 wait_w 04:14 pts/0    00:00:00 ./mmap_fork
```

```
[101169.101620] PCB of the process with given pid = 30844  
[101169.106906] 1 - descriptor number= 0  
[101169.110724] 2 - current file position number = 0  
[101169.115584] 3 - user id = 0  
[101169.118626] 4 - process access mode = 393219  
[101169.123140] 5 - ***** name of the file = 0  
[101169.127476] 6 - inode number of the file = 3  
[101169.131988] 7 - file length = 0 bytes   
[101169.135979] 8 - number of blocks allocated to file = 0    
[101169.141528] 1 - descriptor number= 1  
[101169.145346] 2 - current file position number = 0  
[101169.150201] 3 - user id = 0  
[101169.153245] 4 - process access mode = 393219  
[101169.157754] 5 - ***** name of the file = 0  
[101169.162094] 6 - inode number of the file = 3  
[101169.166604] 7 - file length = 0 bytes   
[101169.170594] 8 - number of blocks allocated to file = 0    
[101169.176143] 1 - descriptor number= 2  
[101169.179961] 2 - current file position number = 0  
[101169.184819] 3 - user id = 0  
[101169.187860] 4 - process access mode = 393219  
[101169.192372] 5 - ***** name of the file = 0  
[101169.196708] 6 - inode number of the file = 3  
[101169.201221] 7 - file length = 0 bytes   
[101169.205212] 8 - number of blocks allocated to file = 0    
[101169.210757] 1 - descriptor number= 3  
[101169.214578] 2 - current file position number = 0  
[101169.219435] 3 - user id = 0  
[101169.222479] 4 - process access mode = 262174  
[101169.226988] 5 - ***** name of the file = zero  
[101169.231585] 6 - inode number of the file = 1028  
[101169.236355] 7 - file length = 0 bytes   
[101169.240344] 8 - number of blocks allocated to file = 0    
[101169.245894] 9 - name of the current directory of the process = /root/programming/kernel/anon_page  
[101169.254986] 10 - blocks that are cached for the process in the page cache = 0
```


# test2

```
[root@centos7 test2]# ps -elf | grep app
0 S root      32623  28539  7  80   0 -    42 wait_w 04:45 pts/0    00:00:00 ./app
```
## insmod  mm_test.ko  processid=32623

```
[root@centos7 test2]# insmod  mm_test.ko  processid=32623
[root@centos7 test2]# dmesg | tail -n 60
[103061.168729] 11 - the storage device the block is in = 0 
[103061.175759] 12 Storage Device (Search Key): 8388611
[103061.175759] 13 Block Number: 98128253
[103061.180705] Use Count: 68
[103061.184436] 3
[103061.187130] 11 - the storage device the block is in = 0 
[103061.194163] 12 Storage Device (Search Key): 8388611
[103061.194164] 13 Block Number: 98128253
[103061.199110] Use Count: 69
[103061.202843] 3
[103061.205537] 11 - the storage device the block is in = 0 
[103061.212569] 12 Storage Device (Search Key): 8388611
[103061.212570] 13 Block Number: 98128253
[103061.217512] Use Count: 70
[103061.221247] 3
[103061.223942] 11 - the storage device the block is in = 0 
[103061.230972] 12 Storage Device (Search Key): 8388611
[103061.230973] 13 Block Number: 98128253
[103061.235915] Use Count: 71
[103061.239651] 3
[103061.242346] 11 - the storage device the block is in = 0 
[103061.249379] 12 Storage Device (Search Key): 8388611
[103061.249379] 13 Block Number: 98128253
[103061.254321] Use Count: 72
[103061.258052] 3
[103061.260751] 11 - the storage device the block is in = 0 
[103061.267779] 12 Storage Device (Search Key): 8388611
[103061.267780] 13 Block Number: 98128253
[103061.272724] Use Count: 73
[103061.276456] 3
[103061.279154] 11 - the storage device the block is in = 0 
[103061.286184] 12 Storage Device (Search Key): 8388611
[103061.286185] 13 Block Number: 98128253
[103061.291129] Use Count: 74
[103061.294861] 3
[103061.297555] 11 - the storage device the block is in = 0 
[103061.304589] 12 Storage Device (Search Key): 8388611
[103061.304590] 13 Block Number: 98128253
[103061.309535] Use Count: 75
[103061.313266] 3
[103061.315960] 11 - the storage device the block is in = 0 
[103061.322994] 12 Storage Device (Search Key): 8388611
[103061.322995] 13 Block Number: 98128253
[103061.327937] Use Count: 76
[103061.331672] 3
[103061.334366] 11 - the storage device the block is in = 0 
[103061.341398] 12 Storage Device (Search Key): 8388611
[103061.341399] 13 Block Number: 98128253
[103061.346340] Use Count: 77
[103061.350075] 3
[103061.352769] 11 - the storage device the block is in = 0 
[103061.359804] 12 Storage Device (Search Key): 8388611
[103061.359805] 13 Block Number: 98128253
[103061.364747] Use Count: 78
[103061.368478] 3
[103061.371176] 11 - the storage device the block is in = 0 
[103061.378205] 12 Storage Device (Search Key): 8388611
[103061.378206] 13 Block Number: 98128253
[103061.383153] 9 - name of the current directory of the process = /root/programming/kernel/page_cache/test2  
[103061.396581] 10 - blocks that are cached for the process in the page cache = 77 
[root@centos7 test2]# dmesg | grep 'name of the file'
[100865.483625] 5 - ***** name of the file = 0  
[100865.518249] 5 - ***** name of the file = 0  
[100865.552868] 5 - ***** name of the file = 0  
[100865.588263] 5 - ***** name of the file = test1.txt  
[101169.123140] 5 - ***** name of the file = 0  
[101169.157754] 5 - ***** name of the file = 0  
[101169.192372] 5 - ***** name of the file = 0  
[101169.226988] 5 - ***** name of the file = zero  
[102910.112938] 5 - ***** name of the file = 0  
[102910.147558] 5 - ***** name of the file = 0  
[102910.182180] 5 - ***** name of the file = 0  
[102910.216797] 5 - ***** name of the file = zero  
[103059.845583] 5 - ***** name of the file = 0  
[103059.880201] 5 - ***** name of the file = 0  
[103059.914815] 5 - ***** name of the file = 0  
[103059.950207] 5 - ***** name of the file = test1.txt  
[root@centos7 test2]#
```