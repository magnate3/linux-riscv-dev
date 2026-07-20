
# pool test

```
gcc -Wall -g -I../src/core/  -I../src/os/unix/   -I../src/auto  -c ../src/core/ngx_string.c  -o ../src/core/ngx_string.o
gcc -Wall -g -I../src/core/  -I../src/os/unix/   -I../src/auto  -c ../src/os/unix/ngx_alloc.c  -o ../src/os/unix/ngx_alloc.o
gcc -Wall -g -I../src/core/  -I../src/os/unix/   -I../src/auto  -c ../src/core/ngx_palloc.c  -o ../src/core/ngx_palloc.o
gcc -Wall -g -I../src/core/  -I../src/os/unix/   -I../src/auto  -c pool.c  -o pool.o
gcc -Wall -g -I../src/core/  -I../src/os/unix/   -I../src/auto  -o mycc ../src/os/unix/ngx_errno.o ../src/core/ngx_log.o ../src/core/ngx_string.o ../src/os/unix/ngx_alloc.o ../src/core/ngx_palloc.o pool.o  
Simple compiler named mycc has been compiled
[root@centos7 examples]# ./mycc 
sp            :bokko
ngx_strlen(sp):5
[root@centos7 examples]# 
```


