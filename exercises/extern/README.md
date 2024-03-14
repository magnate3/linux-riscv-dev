
# __prci_init_clocks_fu740

```
root@ubuntux86:/work/linux-5.13# grep __prci_init_clocks_fu740 -rn *
drivers/clk/sifive/fu740-prci.h:14:extern struct __prci_clock __prci_init_clocks_fu740[NUM_CLOCK_FU740]; //应用
drivers/clk/sifive/fu740-prci.h:17:     .clks = __prci_init_clocks_fu740,
drivers/clk/sifive/fu740-prci.h:18:     .num_clks = ARRAY_SIZE(__prci_init_clocks_fu740),
drivers/clk/sifive/fu740-prci.c:82:struct __prci_clock __prci_init_clocks_fu740[] = {  // 实现
root@ubuntux86:/work/linux-5.13# vim 
```
# struct internal_config

```
[root@centos7 lib]# grep internal_config -rn * | grep struct
librte_eal/linux/eal/eal.c:96:struct internal_config internal_config;
librte_eal/common/eal_internal_cfg.h:39:struct internal_config {
librte_eal/common/eal_internal_cfg.h:86:extern struct internal_config internal_config; /**< Global EAL configuration. */
```

# extern ngx_uint_t  ngx_pagesize

```
[root@centos7 nginx_pool]# grep ngx_pagesize -rn *
Binary file demo_main matches
demo_main.c:15: ngx_pagesize = getpagesize();
demo_main.c:16: //printf("pagesize: %zu\n",ngx_pagesize);
mem_alloc.c:11:ngx_uint_t  ngx_pagesize;
mem_alloc.c:12:ngx_uint_t  ngx_pagesize_shift;
mem_alloc.h:39:extern ngx_uint_t  ngx_pagesize;
mem_alloc.h:40:extern ngx_uint_t  ngx_pagesize_shift;
mem_pool_palloc.h:15: * NGX_MAX_ALLOC_FROM_POOL should be (ngx_pagesize - 1), i.e. 4095 on x86.
mem_pool_palloc.h:18:#define NGX_MAX_ALLOC_FROM_POOL  (ngx_pagesize - 1)
[root@centos7 nginx_pool]# 
```

# extern static

```
[root@centos7 src]# grep  ngx_pid -rn *
config.h:63:#define init_pid() ngx_pid = getpid()
config.h:78:typedef pid_t       ngx_pid_t;
config.h:81:static ngx_pid_t    ngx_pid;
ngx_shmtx.c:12:extern ngx_pid_t ngx_pid;
```

```
[root@centos7 test]# cat test.h
static int val =3;
[root@centos7 test]# cat test.c
#include<stdio.h>
#include"test.h"

int main()
{
    extern int val ;
    printf("value %d \n", val);
    return 0;
}
[root@centos7 test]# gcc test.c -o test
[root@centos7 test]# ./test 
value 3 
[root@centos7 test]# 
```