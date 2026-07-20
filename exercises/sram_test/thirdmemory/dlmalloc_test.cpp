#include <iostream>
#include "dlmalloc.h"
#include <cstdio>
#include <stdint.h>
using namespace std;

//链接libdlmalloc.so: g++ -o test test.cpp -L . -ldlmalloc
//使用dlmalloc替换标准库的malloc free
mspace g_mspace = NULL; //全局内存块 线程不安全
//__thread mspace g_mspace = NULL; //全局内存块 线程安全 但需要在每个线程中create_mspace 需要定义宏 MULTI_TRHEAD
int main(int argc, char** argv)
{
    g_mspace = _get_mspace();
	if (!g_mspace) 
    {
		g_mspace = create_mspace(DEFAULT_MSPACE_CAPACITY, 0);
        //printf("create_mspace: 0x%x.\n", (void*)g_mspace);
	}
    else
    {
        // printf("_get_mspace: 0x%x.\n", (void*)g_mspace);
    }
	_set_mspace(g_mspace);
    //以下所有内存分配都在g_mspace中
    int* p = (int*)malloc(sizeof(int)); //malloc 4 bytes
    free(p);
    p = NULL;
    //新建局部内存块 独立内存的好处是，这块内存写坏了，不会影响其他内存块
    mspace m_mspace = create_mspace(DEFAULT_MSPACE_CAPACITY, 0);
    // printf("create_mspace: 0x%x.\n", (void*)m_mspace);
    _set_mspace(m_mspace); //切换当前内存池
    int* q = (int*)malloc(sizeof(int));
    //默认情况下不可以在g_mspace中释放m_mspace申请的内存(abort), 需要定义FOOTER宏,会自动找到内存所在的内存块
    //如果是多线程模式下则不用定义了，已经默认定义了
    // _set_mspace(g_mspace); 
    free(q);
    printf("mspace foot_print: %d.\n", mspace_footprint(m_mspace));
    int64_t bytes = destroy_mspace(m_mspace); //会释放内存池中全部申请的内存，但是不会调用析构函数，所以最好还是能手动free, delete
    printf("destroy_mspace %d bytes.\n", bytes);
    
    return 0;
}