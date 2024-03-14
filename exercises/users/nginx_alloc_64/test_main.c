#include "ngx_mem_pool_x64.h"
#include<stdio.h>
typedef struct Data stData;
struct Data
{
    char *ptr;
    char *pfile;
};

void func1(void *p)
{
    char *p_ = (char *)p;
    printf("User out ptr memory:%p \n" ,(void *)p_);
    printf("free ptr mem! \n" );
    free(p_);
}
void func2(void *pf)
{
}
int main(int argc,char** argv)
{

    //  创建内存池
    g_pool =  ngx_create_pool(1024);
    void *p1 = ngx_palloc(128); // 从小块内存池分配的
    if(p1 == nullptr)
    {
        printf("ngx_palloc 128 bytes fail... \n");
        goto out;
    }

    stData *p2 = (stData *)ngx_palloc(512); // 从大块内存池分配的
    if(p2 == nullptr)
    {
        printf("ngx_palloc 512 bytes fail... \n");
        goto out;
    }
    printf("pool %p, p1:%p, p2 %p \n",g_pool, p1, p2);
out:
     ngx_destroy_pool();
    // 指向用户自定义的外部资源
    ///p2->ptr = (char *)malloc(12);
    ///strcpy(p2->ptr, "hello world");
    ///p2->pfile = fopen("data.txt", "w");
    
    ///ngx_pool_cleanup_s *c1 = ngx_pool_cleanup_add(sizeof(char*));
    ///c1->handler = func1;
    ///c1->data = p2->ptr;

    //ngx_pool_cleanup_s *c2 = ngx_pool_cleanup_add(sizeof(FILE*));
    //c2->handler = func2;
    //c2->data = p2->pfile;

    // 清理函数以挂入析构函数中
    // 1.调用所有的预置的清理函数
    // 2.释放大块内存
    // 3.释放小块内存池所有内存

    return 0;
}
