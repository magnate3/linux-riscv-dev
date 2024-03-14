
#define 1
 

```
cat main.c 
#include<stdio.h>
int  ngx_pagesize;
#define NGX_MAX_ALLOC_FROM_POOL  (ngx_pagesize - 1)
int main()
{
    ngx_pagesize = getpagesize();
    printf("pagesize: %zu\n",ngx_pagesize); 
    printf("NGX_MAX_ALLOC_FROM_POOL : %zu\n",NGX_MAX_ALLOC_FROM_POOL); 
    return 0;
}
```

```
[root@centos7 test]# gcc -o   nginx main.c 
[root@centos7 test]# ./nginx 
pagesize: 65536
NGX_MAX_ALLOC_FROM_POOL : 65535
```