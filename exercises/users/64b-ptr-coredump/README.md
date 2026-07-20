

#  64位平台上，函数返回指针时遇到的段错误问题

# 数据长度

int	4字节

int *	8字节


# demo1

cat test.c 

```

#include <stdio.h>

static int var = 0x5;

int *func(void)
{
        printf("var: %d, &var: %p\n", var, &var);
        return &var;
}
```

cat main.c 
```

#include <stdio.h>

int main(int argc, const char *argv[])
{
        int *ret = NULL;

        ret = func();
        printf("*addr: %p\n", ret);
        printf("*ret: %d\n", *ret);

        return 0;
}
```

编译运行：

```
gcc test.c  main.c  -o main
main.c: In function ‘main’:
main.c:7:13: warning: assignment makes pointer from integer without a cast [enabled by default
```
编译也提示把interger赋值给一个指针变量

```
./main
var: 5, &var: 0x55ac51c53010
Segmentation fault (core dumped)
```

可以看到，二者果然不同，ret的值仅仅是&var的低4字节的内容。

经过Google，发现，原来是没有在a.c中对func()进行声明，如果没有对func声明，GCC会默认认为func返回的是int类型，而x86_64上，指针占用8个字节，int类型仅仅占用4个字节，所以在赋值时只会把&val的低4字节赋值给ret。

可以参考：

https://stackoverflow.com/questions/23144151/64-bit-function-returns-32-bit-pointer

https://stackoverflow.com/questions/14589314/segmentation-fault-while-accessing-the-return-address-from-a-c-function-in-64-bi


## 问题解决

修改如下，在test.c中增加对func的声明，告诉编译器func的返回值是一个指针类型，需要占用8个字节长度

```
#include <stdio.h>

extern int *func(void);

int main(int argc, const char *argv[])
{
        int *ret = NULL;

        ret = func();
        printf("ret: %p, *ret: %d\n", ret, *ret);

        return 0;
}
```
输出
```
var: 5, &var: 0x561cb45bd010
ret: 0x561cb45bd010, *ret: 5
```


# demo2

代码目录
.
├── common.h
├── main2.c
├── test2.c


 cat test2.c
```
 
#include "common.h"

static struct ABC abc = {
        1, 2, 3
};

struct ABC *func(void)
{
        return &abc;
}
```

 cat main2.c 
 
```
extern struct ABC *func(void);

int main(int argc, const char *argv[])
{
        int *ret = NULL;

        struct ABC *abc = func();
        printf("abc: %p\n", abc);

        return 0;
}
```

gcc test2.c  main2.c  -o main
运行
```
abc: 0x55ed86710010
```

没有问题，但是如果在main2.c中去访问结构体成员的话，编译就会失败：

```
 gcc test2.c  main2.c  -o main
main2.c: In function ‘main’:
main2.c:11:43: error: dereferencing pointer to incomplete type
         printf("a: %d, b: %d, c:%d\n", abc->a, abc->b, abc->c);
                                           ^
main2.c:11:51: error: dereferencing pointer to incomplete type
         printf("a: %d, b: %d, c:%d\n", abc->a, abc->b, abc->c);
                                                   ^
main2.c:11:59: error: dereferencing pointer to incomplete type
         printf("a: %d, b: %d, c:%d\n", abc->a, abc->b, abc->c);
```
原因是，在x86_64上，不管是什么类型的指针，GCC都会会分配8个字节，这个不会有问题，但是如果要访问指针指向的结构体成员的话，就需要告诉GCC这个成员具体是什么样子。解决办法同时是需要声明一下结构体类型，这里包含对应的头文件即可。

```
#include <stdio.h>
#include "common.h"

extern struct ABC *func(void);

int main(int argc, const char *argv[])
{
        int *ret = NULL;

        struct ABC *abc = func();
        printf("abc: %p\n", abc);

        printf("a: %d, b: %d, c:%d\n", abc->a, abc->b, abc->c);

        return 0;
}
```




