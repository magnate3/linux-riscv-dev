
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#include <ngx_config.h>
#include <ngx_core.h>


//static char *ngx_error_log(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);
//static char *ngx_log_set_levels(ngx_conf_t *cf, ngx_log_t *log);
//static void ngx_log_insert(ngx_log_t *log, ngx_log_t *new_log);
//
//
//static ngx_command_t  ngx_errlog_commands[] = {
//
//    {ngx_string("error_log"),
//     NGX_MAIN_CONF|NGX_CONF_1MORE,
//     ngx_error_log,
//     0,
//     0,
//     NULL},
//
//    ngx_null_command
//};
//
//
//static ngx_core_module_t  ngx_errlog_module_ctx = {
//    ngx_string("errlog"),
//    NULL,
//    NULL
//};
//
//
//ngx_module_t  ngx_errlog_module = {
//    NGX_MODULE_V1,
//    &ngx_errlog_module_ctx,                /* module context */
//    ngx_errlog_commands,                   /* module directives */
//    NGX_CORE_MODULE,                       /* module type */
//    NULL,                                  /* init master */
//    NULL,                                  /* init module */
//    NULL,                                  /* init process */
//    NULL,                                  /* init thread */
//    NULL,                                  /* exit thread */
//    NULL,                                  /* exit process */
//    NULL,                                  /* exit master */
//    NGX_MODULE_V1_PADDING
//};
//
//
//static ngx_log_t        ngx_log;
//static ngx_open_file_t  ngx_log_file;
//ngx_uint_t              ngx_use_stderr = 1;
//
//
//static ngx_str_t err_levels[] = {
//    ngx_null_string,
//    ngx_string("emerg"),
//    ngx_string("alert"),
//    ngx_string("crit"),
//    ngx_string("error"),
//    ngx_string("warn"),
//    ngx_string("notice"),
//    ngx_string("info"),
//    ngx_string("debug")
//};
//
//static const char *debug_levels[] = {
//    "debug_core", "debug_alloc", "debug_mutex", "debug_event",
//    "debug_http", "debug_mail", "debug_mysql"
//};


#if (NGX_HAVE_VARIADIC_MACROS)

#define NGX_LOG_PREFIX "nginxErr: "
void
ngx_log_error_core(ngx_uint_t level, ngx_log_t *log, ngx_err_t err,
    const char *fmt, ...)
#else

void
ngx_log_error_core(ngx_uint_t level, ngx_log_t *log, ngx_err_t err,
    const char *fmt, va_list args)

#endif
{
    u_char   *p, *last;
    va_list   args;
    u_char    errstr[NGX_MAX_ERROR_STR];
    last = errstr + NGX_MAX_ERROR_STR;
    p = errstr + sizeof(NGX_LOG_PREFIX);
    //p = errstr + 7;
    
    ngx_memcpy(errstr, NGX_LOG_PREFIX, sizeof(NGX_LOG_PREFIX));
    //ngx_memcpy(errstr, "nginx: ", 7);

    va_start(args, fmt);
    p = ngx_vslprintf(p, last, fmt, args);
    va_end(args);

    if (err) {
        p = ngx_log_errno(p, last, err);
    }

    if (p > last - NGX_LINEFEED_SIZE) {
        p = last - NGX_LINEFEED_SIZE;
    }

    ngx_linefeed(p);

    (void) ngx_write_console(ngx_stderr, errstr, p - errstr);
}


#if !(NGX_HAVE_VARIADIC_MACROS)

void ngx_cdecl
ngx_log_error(ngx_uint_t level, ngx_log_t *log, ngx_err_t err,
    const char *fmt, ...)
{
    va_list  args;

    if (log->log_level >= level) {
        va_start(args, fmt);
        ngx_log_error_core(level, log, err, fmt, args);
        va_end(args);
    }
}


void ngx_cdecl
ngx_log_debug_core(ngx_log_t *log, ngx_err_t err, const char *fmt, ...)
{
#if 0
    va_list  args;

    va_start(args, fmt);
    ngx_log_error_core(NGX_LOG_DEBUG, log, err, fmt, args);
    va_end(args);
#else
    u_char   *p, *last;
    va_list   args;
    u_char    errstr[NGX_MAX_ERROR_STR];

    last = errstr + NGX_MAX_ERROR_STR;
    p = errstr + 7;

    ngx_memcpy(errstr, "nginx: ", 7);

    va_start(args, fmt);
    p = ngx_vslprintf(p, last, fmt, args);
    va_end(args);

    if (err) {
        p = ngx_log_errno(p, last, err);
    }

    if (p > last - NGX_LINEFEED_SIZE) {
        p = last - NGX_LINEFEED_SIZE;
    }

    ngx_linefeed(p);

    (void) ngx_write_console(ngx_stderr, errstr, p - errstr);
#endif
}

#endif


void ngx_cdecl
ngx_log_abort(ngx_err_t err, const char *fmt, ...)
{
    u_char   *p;
    va_list   args;
    u_char    errstr[NGX_MAX_CONF_ERRSTR];

    va_start(args, fmt);
    p = ngx_vsnprintf(errstr, sizeof(errstr) - 1, fmt, args);
    va_end(args);

    //ngx_log_error(NGX_LOG_ALERT, ngx_cycle->log, err,
    //              "%*s", p - errstr, errstr);
}


void ngx_cdecl
ngx_log_stderr(ngx_err_t err, const char *fmt, ...)
{
    u_char   *p, *last;
    va_list   args;
    u_char    errstr[NGX_MAX_ERROR_STR];

    last = errstr + NGX_MAX_ERROR_STR;
    p = errstr + 7;

    ngx_memcpy(errstr, "nginx: ", 7);

    va_start(args, fmt);
    p = ngx_vslprintf(p, last, fmt, args);
    va_end(args);

    if (err) {
        p = ngx_log_errno(p, last, err);
    }

    if (p > last - NGX_LINEFEED_SIZE) {
        p = last - NGX_LINEFEED_SIZE;
    }

    ngx_linefeed(p);

    (void) ngx_write_console(ngx_stderr, errstr, p - errstr);
}


u_char *
ngx_log_errno(u_char *buf, u_char *last, ngx_err_t err)
{
    if (buf > last - 50) {

        /* leave a space for an error code */

        buf = last - 50;
        *buf++ = '.';
        *buf++ = '.';
        *buf++ = '.';
    }

#if (NGX_WIN32)
    buf = ngx_slprintf(buf, last, ((unsigned) err < 0x80000000)
                                       ? " (%d: " : " (%Xd: ", err);
#else
    buf = ngx_slprintf(buf, last, " (%d: ", err);
#endif

    buf = ngx_strerror(err, buf, last - buf);

    if (buf < last) {
        *buf++ = ')';
    }

    return buf;
}


///ngx_log_t *
///ngx_log_init(u_char *prefix)
///{
///
///    return &ngx_log;
///}


//ngx_int_t
//ngx_log_open_default(ngx_cycle_t *cycle)
//{
//
//    return NGX_OK;
//}
//
//
//ngx_int_t
//ngx_log_redirect_stderr(ngx_cycle_t *cycle)
//{
//
//    return NGX_OK;
//}


//ngx_log_t *
//ngx_log_get_file_log(ngx_log_t *head)
//{
//
//    return NULL;
//}
//
//
//static char *
//ngx_log_set_levels(ngx_conf_t *cf, ngx_log_t *log)
//{
//
//    return NGX_CONF_OK;
//}
//
//
//static char *
//ngx_error_log(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
//{
//    ngx_log_t  *dummy;
//
//    dummy = &cf->cycle->new_log;
//
//    return ngx_log_set_log(cf, &dummy);
//}
//
//
//char *
//ngx_log_set_log(ngx_conf_t *cf, ngx_log_t **head)
//{
//
//    return NGX_CONF_OK;
//}
//
//
//static void
//ngx_log_insert(ngx_log_t *log, ngx_log_t *new_log)
//{
//}
