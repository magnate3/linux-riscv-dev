
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_CONFIG_H_INCLUDED_
#define _NGX_CONFIG_H_INCLUDED_
#include <ngx_linux_config.h>
/*
于是在linux的头文件中查找这个类型的定义，在/usr/include/stdint.h这个头文件中找到了这个类型的定义（不知道怎么在这里插入图片，所以使用文字）：

[cpp] view plaincopy
117  Types for `void *' pointers.
118 #if __WORDSIZE == 64
119 # ifndef __intptr_t_defined
120 typedef long int        intptr_t;
121 #  define __intptr_t_defined
122 # endif
123 typedef unsigned long int   uintptr_t;
124 #else
125 # ifndef __intptr_t_defined
126 typedef int         intptr_t;
127 #  define __intptr_t_defined
128 # endif
129 typedef unsigned int        uintptr_t;
130 #endif

很明显intptr_t不是指针类型，但是上边的一句注释（ Types for `void *' pointers. ）让人很疑惑。既然不是指针类型，但是为什么说类型是为了“void *”指针？
又查了一下在《深入分析Linux内核源码》中找到了答案，原文描述如下：

尽管在混合不同数据类型时你必须小心, 有时有很好的理由这样做. 一种情况是因为内存存取, 与内核相关时是特殊的. 概念上, 尽管地址是指针,
内存管理常常使用一个无符号的整数类型更好地完成; 内核对待物理内存如同一个大数组, 并且内存地址只是一个数组索引. 进一步地, 一个指针容易解引用;
当直接处理内存存取时, 你几乎从不想以这种方式解引用. 使用一个整数类型避免了这种解引用, 因此避免了 bug. 因此, 内核中通常的内存地址常常是 unsigned long,
利用了指针和长整型一直是相同大小的这个事实, 至少在 Linux 目前支持的所有平台上.

因为其所值的原因, C99 标准定义了 intptr_t 和 uintptr_t 类型给一个可以持有一个指针值的整型变量. 但是, 这些类型几乎没在 2.6 内核中使用
*/
/*
Nginx使用ngx_int_t封装有符号整型，使用ngx_uint_t封装无符号整型。Nginx各模块的变量定义都是如此使用的，建议读者沿用Nginx的习惯，以此替代int和unsinged int。

在Linux平台下，Nginx对ngx_int_t和ngx_uint_t的定义如下：
typedef intptr_t        ngx_int_t;
typedef uintptr_t       ngx_uint_t;
*/
typedef intptr_t        ngx_int_t;
typedef uintptr_t       ngx_uint_t;

#ifndef NGX_ALIGNMENT
#define NGX_ALIGNMENT   sizeof(unsigned long)    /* platform word */
#endif

#define ngx_align(d, a)     (((d) + (a - 1)) & ~(a - 1))
//// 将 m 对其到内存对齐地址
#define ngx_align_ptr(p, a)                                                    \
    (u_char *) (((uintptr_t) (p) + ((uintptr_t) a - 1)) & ~((uintptr_t) a - 1))
/* TODO: auto_conf: ngx_inline   inline __inline __inline__ */
#ifndef ngx_inline
#define ngx_inline      inline
#endif
typedef int                      ngx_fd_t;

typedef struct {
    size_t      len;
    u_char     *data;
} ngx_str_t;

#define ngx_memzero(buf, n)       (void) memset(buf, 0, n)
#define ngx_memset(buf, c, n)     (void) memset(buf, c, n)

#define ngx_strlen(s)       strlen((const char *) s)

typedef pid_t       ngx_pid_t;

#if (NGX_HAVE_SCHED_YIELD)
#define ngx_sched_yield()  sched_yield()
#else
#define ngx_sched_yield()  usleep(1)
#endif

typedef long                        ngx_atomic_int_t;
typedef uint64_t                    ngx_atomic_uint_t;
typedef volatile ngx_atomic_uint_t  ngx_atomic_t;
static ngx_inline ngx_atomic_uint_t ngx_atomic_cmp_set(ngx_atomic_t *lock, ngx_atomic_uint_t old,
     ngx_atomic_uint_t set)
{
     if (*lock == old) {
         *lock = set;
         return 1;
     }

     return 0;
}
static ngx_inline ngx_atomic_int_t ngx_atomic_fetch_add(ngx_atomic_t *value, ngx_atomic_int_t add)
{
     ngx_atomic_int_t  old;
     old = *value;
     *value += add;

     return old;
}


#if ( __i386__ || __i386 || __amd64__ || __amd64 )
#define ngx_cpu_pause()             __asm__ ("pause")
#else
#define ngx_cpu_pause()
#endif
typedef int               ngx_err_t;

#define ngx_errno                  errno
#define NGX_EINTR         EINTR
#endif /* _NGX_CONFIG_H_INCLUDED_ */
