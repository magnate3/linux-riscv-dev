
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_CONFIG_H_INCLUDED_
#define _NGX_CONFIG_H_INCLUDED_
#include <ngx_linux_config.h>
/*
������linux��ͷ�ļ��в���������͵Ķ��壬��/usr/include/stdint.h���ͷ�ļ����ҵ���������͵Ķ��壨��֪����ô���������ͼƬ������ʹ�����֣���

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

������intptr_t����ָ�����ͣ������ϱߵ�һ��ע�ͣ� Types for `void *' pointers. �����˺��ɻ󡣼�Ȼ����ָ�����ͣ�����Ϊʲô˵������Ϊ�ˡ�void *��ָ�룿
�ֲ���һ���ڡ��������Linux�ں�Դ�롷���ҵ��˴𰸣�ԭ���������£�

�����ڻ�ϲ�ͬ��������ʱ�����С��, ��ʱ�кܺõ�����������. һ���������Ϊ�ڴ��ȡ, ���ں����ʱ�������. ������, ���ܵ�ַ��ָ��,
�ڴ������ʹ��һ���޷��ŵ��������͸��õ����; �ں˶Դ������ڴ���ͬһ��������, �����ڴ��ַֻ��һ����������. ��һ����, һ��ָ�����׽�����;
��ֱ�Ӵ����ڴ��ȡʱ, �㼸���Ӳ��������ַ�ʽ������. ʹ��һ���������ͱ��������ֽ�����, ��˱����� bug. ���, �ں���ͨ�����ڴ��ַ������ unsigned long,
������ָ��ͳ�����һֱ����ͬ��С�������ʵ, ������ Linux Ŀǰ֧�ֵ�����ƽ̨��.

��Ϊ����ֵ��ԭ��, C99 ��׼������ intptr_t �� uintptr_t ���͸�һ�����Գ���һ��ָ��ֵ�����ͱ���. ����, ��Щ���ͼ���û�� 2.6 �ں���ʹ��
*/
/*
Nginxʹ��ngx_int_t��װ�з������ͣ�ʹ��ngx_uint_t��װ�޷������͡�Nginx��ģ��ı������嶼�����ʹ�õģ������������Nginx��ϰ�ߣ��Դ����int��unsinged int��

��Linuxƽ̨�£�Nginx��ngx_int_t��ngx_uint_t�Ķ������£�
typedef intptr_t        ngx_int_t;
typedef uintptr_t       ngx_uint_t;
*/
typedef intptr_t        ngx_int_t;
typedef uintptr_t       ngx_uint_t;

#ifndef NGX_ALIGNMENT
#define NGX_ALIGNMENT   sizeof(unsigned long)    /* platform word */
#endif

#define ngx_align(d, a)     (((d) + (a - 1)) & ~(a - 1))
//// �� m ���䵽�ڴ�����ַ
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
