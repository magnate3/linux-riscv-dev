
# test1

```C
 cat construct.c 
#include <stdio.h>
#include <stdlib.h>

static void __attribute__ ((constructor)) __reg_module(void)
{
    printf("__reg_module called.\n");
}
static  __attribute__((constructor(101))) void before1()
{
    
    printf("before1\n");
}
static  __attribute__((constructor(102))) void before2()
{
    
    printf("before2\n");
}
static  __attribute__((constructor(102))) void before3()
{
    
    printf("before3\n");
}


static void __attribute__ ((destructor)) __unreg_module(void)
{
    printf("__unreg_module called.\n");
}

int main(int argc, const char *argv[])
{
    printf("main called.\n");
    
    return 0;
}
```



```Shell
 ./construct 
before1
before2
before3
__reg_module called.
main called.
__unreg_module called.
```