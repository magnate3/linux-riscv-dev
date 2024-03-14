
# build and run

```
[root@centos7 dlmalloc]#  gcc dlmalloc.c -O3 -fPIC -shared -o libdlmalloc.so -std=gnu99
[root@centos7 dlmalloc]# 
```

```
[root@centos7 thirdmemory]# g++ dlmalloc_test.cpp  -o test -I ./dlmalloc   -ldlmalloc  -L./dlmalloc
[root@centos7 thirdmemory]# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./dlmalloc
[root@centos7 thirdmemory]# ./test 
mspace foot_print: 67174400.
destroy_mspace 67174400 bytes.
[root@centos7 thirdmemory]# 

```