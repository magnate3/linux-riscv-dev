# Nginx_---
Nginx_内存池-删减版

## 编译

```shell
gcc demo_main.c mem_alloc.c mem_alloc.h mem_core.h mem_pool_palloc.c mem_pool_palloc.h -o demo_main
````



## 运行结果

- 带参数的话用普通方式申请内存
- 不带参数的话用内存池

```shell
➜  Nginx_mem_pool time ./demo_main 1
use malloc/free
./demo_main 1  11.75s user 0.00s system 95% cpu 12.241 total
➜  Nginx_mem_pool time ./demo_main 1
use malloc/free
./demo_main 1  11.90s user 0.00s system 95% cpu 12.402 total
➜  Nginx_mem_pool time ./demo_main 1
use malloc/free
./demo_main 1  11.75s user 0.00s system 95% cpu 12.240 total
➜  Nginx_mem_pool time ./demo_main
use mempool.
./demo_main  9.14s user 0.00s system 95% cpu 9.531 total
➜  Nginx_mem_pool time ./demo_main
use mempool.
./demo_main  9.15s user 0.00s system 95% cpu 9.556 total
➜  Nginx_mem_pool time ./demo_main
use mempool.
./demo_main  9.25s user 0.00s system 96% cpu 9.599 total
```
