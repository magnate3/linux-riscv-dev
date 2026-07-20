
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_CORE_H_INCLUDED_
#define _NGX_CORE_H_INCLUDED_


#include <ngx_config.h>



typedef struct ngx_pool_s        ngx_pool_t;
typedef struct ngx_chain_s       ngx_chain_t;
typedef struct ngx_log_s         ngx_log_t;
typedef struct ngx_open_file_s   ngx_open_file_t;




/*
这4个返回码引发的Nginx一系列动作。

NGX_OK：表示成功。Nginx将会继续执行该请求的后续动作（如执行subrequest或撤销这个请求）。

NGX_DECLINED：继续在NGX_HTTP_CONTENT_PHASE阶段寻找下一个对于该请求感兴趣的HTTP模块来再次处理这个请求。

NGX_DONE：表示到此为止，同时HTTP框架将暂时不再继续执行这个请求的后续部分。事实上，这时会检查连接的类型，如果是keepalive类型的用户请求，
    就会保持住HTTP连接，然后把控制权交给Nginx。这个返回码很有用，考虑以下场景：在一个请求中我们必须访问一个耗时极长的操作（比如某个网络调用），
    这样会阻塞住Nginx，又因为我们没有把控制权交还给Nginx，而是在ngx_http_mytest_handler中让Nginx worker进程休眠了（如等待网络的回包），
    所以，这就会导致Nginx出现性能问题，该进程上的其他用户请求也得不到响应。可如果我们把这个耗时极长的操作分为上下两个部分（就像Linux内核
    中对中断处理的划分），上半部分和下半部分都是无阻塞的（耗时很少的操作），这样，在ngx_http_mytest_handler进入时调用上半部分，然后返回NGX_DONE，
    把控制交还给Nginx，从而让Nginx继续处理其他请求。在下半部分被触发时（这里不探讨具体的实现方式，事实上使用upstream方式做反向代理时用的就是
    这种思想），再回调下半部分处理方法，这样就可以保证Nginx的高性能特性了。

NGX_ERROR：表示错误。这时会调用ngx_http_terminate_request终止请求。如果还有POST子请求，那么将会在执行完POST请求后再终止本次请求。
*/
#define  NGX_OK          0
#define  NGX_ERROR      -1
#define  NGX_AGAIN      -2
#define  NGX_BUSY       -3
#define  NGX_DONE       -4
#define  NGX_DECLINED   -5
#define  NGX_ABORT      -6

#include <ngx_shmem.h>
#include <ngx_log.h>
#include <ngx_alloc.h>
#include <ngx_palloc.h>
#include <ngx_shmtx.h>
#include <ngx_slab.h>

//getconf PAGE_SIZE 命令可以查看
ngx_uint_t  ngx_pagesize;//见ngx_os_init  返回一个分页的大小，单位为字节(Byte)。该值为系统的分页大小，不一定会和硬件分页大小相同。
//ngx_pagesize为4M，ngx_pagesize_shift应该为12
ngx_uint_t  ngx_pagesize_shift; //ngx_pagesize进行移位的次数，见for (n = ngx_pagesize; n >>= 1; ngx_pagesize_shift++) { /* void */ }
//初始化参考ngx_init_cycle，最终有一个全局类型的ngx_cycle_s，即ngx_cycle
//volatile ngx_cycle_t  *ngx_cycle; //ngx_cycle = cycle;  赋值见main->ngx_init_cycle
ngx_log_t    *logLocal;
ngx_int_t   ngx_ncpu; //cpu个数
ngx_pid_t     ngx_pid;//ngx_pid = ngx_getpid(); 在子进程中为子进程pid，在master中为master的pid







#endif /* _NGX_CORE_H_INCLUDED_ */
