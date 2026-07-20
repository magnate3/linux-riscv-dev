#ifndef __NGINX_ALLOC_H__
#define __NGINX_ALLOC_H__
#include <stdlib.h>
#include <string.h>

#define nullptr NULL
/*
    实现 Nginx 内存池的移植
*/

/// @brief 类型重定义（原 C 代码中使用 typedef 此处使用 using）
typedef unsigned char u_char;
typedef unsigned int ngx_uint_t;

// typedef unsigned int uintptr_t;

/// @brief 内存前置声明
struct ngx_pool_s;

/// @brief 回调清理函数的类型
typedef void (*ngx_pool_cleanup_pt)(void *data);

/// @brief 清理操作的内存块头信息
struct ngx_pool_cleanup_s
{
    ngx_pool_cleanup_pt handler; // 回调函数指针
    void *data;                  // 回调函数参数
    struct ngx_pool_cleanup_s *next;    // 下一个清理内存块
};

/// @brief 大块内存的头部信息
struct ngx_pool_large_s
{
    struct ngx_pool_large_s *next; // 下一个大块内存头部信息
    void *alloc;            // 分配出去的大块内存的起始地址
};

/// @brief 分配小块内存的内存池头部数据信息
struct ngx_pool_data_t
{
    u_char *last;      // 小块内存池中可用内存的起始地址
    u_char *end;       // 小块内存池中可用内存的结束地址
    struct ngx_pool_s *next;  // 下一个小块内存池头地址
    ngx_uint_t failed; // 当前小块内存分配失败次数
};

/// @brief ngx入口内存池的头信息和管理成功信息（由于不需要 chain 和 log 指针所以将其删掉）
struct ngx_pool_s
{
    struct ngx_pool_data_t d;           // 小块内存数据相关的存储情况（上一个结构体）
    size_t max;                  // 小块内存与大块内存的分界线
    struct ngx_pool_s *current;         // 指向当前使用的小块内存池（初始指向第一个）
    struct ngx_pool_large_s *large;     // 大块内存头信息串联的链表的入口地址
    struct ngx_pool_cleanup_s *cleanup; // 指向所有预置清理操作内存块的入口
};

// buf 缓冲区清 0
#define ngx_memzero(buf, n)       (void) memset(buf, 0, n)
// 把数值 d 调整到临近的 a 的倍数
#define ngx_align(d, a) (((d) + (a - 1)) & ~(a - 1))
// 把指针 p 调整到 a 的临近的倍数
#define ngx_align_ptr(p, a)                                                   \
    (u_char *) (((unsigned long int) (p) + ((unsigned long int) a - 1)) & ~((unsigned long int) a - 1))
// 小块内存分配考虑字节对齐时的单位
#define NGX_ALIGNMENT   sizeof(unsigned long)    /* platform word */
#define NGX_PAGESIZE 4096
// 默认一个物理页面的大小 4K
//const int ngx_pagesize = NGX_PAGESIZE;
// ngx 小块内存池可分配的最大空间
#define NGX_MAX_ALLOC_FROM_POOL (NGX_PAGESIZE - 1)
// 定义常量，表示一个默认的 ngx 内存池开辟的大小（有些程序会直接创建 16K 大小的内存池）
#define NGX_DEFAULT_POOL_SIZE  (16 * 1024);
// 内存池大小按照 16 字节进行对齐
#define  NGX_POOL_ALIGNMENT  16
// ngx 小块内存池最小的 size 调整成 NGX_POOL_ALIGNMENT 临近的倍数对齐
// 因为小块内存池的头部大小就是 16 字节，若最终开辟的小块内存还不足 16 字节
// 那么连内存头都放不下，所以要通过该函数将其大小起码调整为 16 的倍数
// 此处 2 * sizeof(ngx_pool_large_s) = 16
// 表示除了内存头以外最小要再开辟 16 字节的空间
//const int NGX_MIN_POOL_SIZE = ngx_align((sizeof(struct ngx_pool_s) + 2 * sizeof(struct ngx_pool_large_s)), NGX_POOL_ALIGNMENT);
#define  NGX_MIN_POOL_SIZE   ngx_align((sizeof(struct ngx_pool_s) + 2 * sizeof(struct ngx_pool_large_s)), NGX_POOL_ALIGNMENT)

/// @brief 创建指定大小的内存池（由于 C++ 实现，直接将创建的内存池的地址保存于 pool 即可）
/// @param size 指定内存池的大小（小块内存池 < 4096）
/// @return 是否创建成功
void *ngx_create_pool(size_t size);

/// @brief 考虑内存字节对齐
/// @param size 从内存池申请 size 大小的内存
/// @return 无
void *ngx_palloc(size_t size);

/// @brief 不考虑内存字节对齐
/// @param size 从内存池申请 size 大小的内存
/// @return
void *ngx_pnalloc(size_t size);

/// @brief 调用 ngx_palloc 分配内存，区别在于会将内存初始化为 0
/// @param size 从内存池申请 size 大小的内存
/// @return
void *ngx_pcalloc(size_t size);

/// @brief 释放大块内存（ngx 中不会释放小块内存 - 可能存在问题）
/// @param p 大块内存起始地址
/// @return
void ngx_pfree(void *p);

/// @brief 内存池重置
void ngx_rest_pool();

/// @brief 内存池销毁
void ngx_destroy_pool();

/// @brief 添加回调清理操作函数头信息
/// @param size 头信息大小
/// @return 清理操作块头信息节点
struct ngx_pool_cleanup_s *ngx_pool_cleanup_add(size_t size);

// 指向 Nginx 内存池的入口指针
struct ngx_pool_s *g_pool;

/// @brief 小块内存分配
/// @param size
/// @param align
/// @return
void *ngx_palloc_small(size_t size, ngx_uint_t align);

/// @brief 大块内存分配
/// @param size
/// @return
void *ngx_palloc_large(size_t size);

/// @brief 分配新的小块内存池
/// @param size
/// @return
void *ngx_palloc_block(size_t size);
#endif
