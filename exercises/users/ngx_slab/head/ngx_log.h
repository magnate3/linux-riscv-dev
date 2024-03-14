
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#ifndef _NGX_LOG_H_INCLUDED_
#define _NGX_LOG_H_INCLUDED_


#include <ngx_config.h>
#include <ngx_core.h>

#define NGX_LOG_STDERR            0
#define NGX_LOG_EMERG             1
#define NGX_LOG_ALERT             2
#define NGX_LOG_CRIT              3
#define NGX_LOG_ERR               4
#define NGX_LOG_WARN              5 //如果level > NGX_LOG_WARN则不会在屏幕前台打印，见ngx_log_error_core
#define NGX_LOG_NOTICE            6
#define NGX_LOG_INFO              7
#define NGX_LOG_DEBUG             8
struct ngx_log_s {
    //如果设置的log级别为debug，则会在ngx_log_set_levels把level设置为NGX_LOG_DEBUG_ALL
    //ngx_log_set_levels
    ngx_uint_t           log_level;//日志级别或者日志类型  默认为NGX_LOG_ERR  如果通过error_log  logs/error.log  info;则为设置的等级  比该级别下的日志可以打印
    ngx_open_file_t     *file; //日志文件

    ngx_atomic_uint_t    connection;//连接数，不为O时会输出到日志中

    time_t               disk_full_time;

    /* 记录日志时的回调方法。当handler已经实现（不为NULL），并且不是DEBUG调试级别时，才会调用handler钩子方法 */
  

    /*
    每个模块都可以自定义data的使用方法。通常，data参数都是在实现了上面的handler回调方法后
    才使用的。例如，HTTP框架就定义了handler方法，并在data中放入了这个请求的上下文信息，这样每次输出日
    志时都会把这个请求URI输出到日志的尾部
    */
    void                *data; //指向ngx_http_log_ctx_t，见ngx_http_init_connection

    
    void                *wdata;

    /*
     * we declare "action" as "char *" because the actions are usually
     * the static strings and in the "u_char *" case we have to override
     * their types all the time
     */

    /*
    表示当前的动作。实际上，action与data是一样的，只有在实现了handler回调方法后才会使
    用。例如，HTTP框架就在handler方法中检查action是否为NULL，如果不为NULL，就会在日志后加入“while
    ”+action，以此表示当前日志是在进行什么操作，帮助定位问题
    */
    char                *action;
    //ngx_log_insert插入，在ngx_log_error_core找到对应级别的日志配置进行输出，因为可以配置error_log不同级别的日志存储在不同的日志文件中
    ngx_log_t           *next;
};

void ngx_log_error(ngx_uint_t level, ngx_log_t *log,...);
#endif /* _NGX_LOG_H_INCLUDED_ */

