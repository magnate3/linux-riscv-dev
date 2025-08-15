#include <stdio.h>
#include <stdlib.h>
struct test1
{
    int a;
    int b[0];
};
struct test2
{
    int a;
    int *b;
};
struct test3
{
    int a;
    int reserved;//占位符，64位系统中保证结构体字节对齐，test2中是由编译器对齐的，所以两个结构体占用空间相同
    int *b;
};
int main()
{
    struct test1 *var1;
    struct test2 *var2;
    int iLength = 10;
    int i;
    printf("the length of struct test1:%d\n",sizeof(struct test1));
    printf("the length of struct test2:%d\n",sizeof(struct test2));
    printf("the length of struct test3:%d\n",sizeof(struct test3));
    var1=(struct test1*)malloc(sizeof(struct test1) + sizeof(int) * iLength);
    var1->a=iLength;
    for(i=0; i < var1->a; i++)
    {
        var1->b[i]=i;
        printf("var1->b[%d]=%d\t", i, var1->b[i]);
    }
    printf("\n"); 
    printf("p var1 = %p\n", var1);
    printf("p var1->a %p\n", &var1->a);
    printf("var1->b %p\n", var1->b);
    printf("p var1->b %p\n", &var1->b);
    printf("p var1->b[0] %p\n", &var1->b[0]);
    printf("p var1->b[1] %p\n", &var1->b[1]);
    printf("\n\n");
    var2=(struct test2*)malloc(sizeof(struct test2));
    var2->a=iLength;
    var2->b=(int *)malloc(sizeof(int) * iLength);
    for(i=0; i < var2->a; i++)
    {
        var2->b[i]=i;
        printf("var2->b[%d]=%d\t", i, var2->b[i]);
    }
    printf("\n"); 
    printf("p var2 = %p\n", var2);
    printf("p var2->a %p\n", &var2->a);
    printf("var2->b %p\n", var2->b);
    printf("p var2->b %p\n", &var2->b);
    printf("p var2->b[0] %p\n", &var2->b[0]);
    printf("p var2->b[1] %p\n", &var2->b[1]);
    free(var1);
    free(var2->b);
    free(var2);
    return 0;
}