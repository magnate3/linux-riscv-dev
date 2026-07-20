# make

```
root@centos7 ngx_slab2]# gcc main.c  ncx_slab.c -o main  -g -O0 -Wall -D LOG_LEVEL=4
[root@centos7 ngx_slab2]# ./main 
[I] [ncx_slab_stat:1063] pool_size : 4030480 bytes
[I] [ncx_slab_stat:1064] used_size : 256 bytes
[I] [ncx_slab_stat:1065] used_pct  : 0%

[I] [ncx_slab_stat:1067] total page count : 61
[I] [ncx_slab_stat:1068] free page count  : 60

[I] [ncx_slab_stat:1070] small slab use page : 1,       bytes : 256
[I] [ncx_slab_stat:1071] exact slab use page : 0,       bytes : 0
[I] [ncx_slab_stat:1072] big   slab use page : 0,       bytes : 0
[I] [ncx_slab_stat:1073] page slab use page  : 0,       bytes : 0

[I] [ncx_slab_stat:1075] max free pages : 60

[root@centos7 ngx_slab2]# 
```