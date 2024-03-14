

```
#include <stdlib.h>  //for malloc(),realloc()
#include <stddef.h>  //for size_t
#include <memory.h>  //for memcpy()
#include <stdio.h>  //for memcpy()
int fun(int i, int j)
{
    printf("%s, i =  %d, and j = %d  \n", __func__, i, j);
    return 0;
}
int (*test(int a))(int i, int j)
{
    printf("%s, a =  %d \n", __func__, a);
    return fun;
}
int main()
{
    test(3)(4,5);
    return 0;
}
```
运行结果
```
test, a =  3 
fun, i =  4, and j = 5  
```