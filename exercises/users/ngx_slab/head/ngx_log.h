
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
#define NGX_LOG_WARN              5 //���level > NGX_LOG_WARN�򲻻�����Ļǰ̨��ӡ����ngx_log_error_core
#define NGX_LOG_NOTICE            6
#define NGX_LOG_INFO              7
#define NGX_LOG_DEBUG             8
struct ngx_log_s {
    //������õ�log����Ϊdebug�������ngx_log_set_levels��level����ΪNGX_LOG_DEBUG_ALL
    //ngx_log_set_levels
    ngx_uint_t           log_level;//��־���������־����  Ĭ��ΪNGX_LOG_ERR  ���ͨ��error_log  logs/error.log  info;��Ϊ���õĵȼ�  �ȸü����µ���־���Դ�ӡ
    ngx_open_file_t     *file; //��־�ļ�

    ngx_atomic_uint_t    connection;//����������ΪOʱ���������־��

    time_t               disk_full_time;

    /* ��¼��־ʱ�Ļص���������handler�Ѿ�ʵ�֣���ΪNULL�������Ҳ���DEBUG���Լ���ʱ���Ż����handler���ӷ��� */
  

    /*
    ÿ��ģ�鶼�����Զ���data��ʹ�÷�����ͨ����data����������ʵ���������handler�ص�������
    ��ʹ�õġ����磬HTTP��ܾͶ�����handler����������data�з���������������������Ϣ������ÿ�������
    ־ʱ������������URI�������־��β��
    */
    void                *data; //ָ��ngx_http_log_ctx_t����ngx_http_init_connection

    
    void                *wdata;

    /*
     * we declare "action" as "char *" because the actions are usually
     * the static strings and in the "u_char *" case we have to override
     * their types all the time
     */

    /*
    ��ʾ��ǰ�Ķ�����ʵ���ϣ�action��data��һ���ģ�ֻ����ʵ����handler�ص�������Ż�ʹ
    �á����磬HTTP��ܾ���handler�����м��action�Ƿ�ΪNULL�������ΪNULL���ͻ�����־����롰while
    ��+action���Դ˱�ʾ��ǰ��־���ڽ���ʲô������������λ����
    */
    char                *action;
    //ngx_log_insert���룬��ngx_log_error_core�ҵ���Ӧ�������־���ý����������Ϊ��������error_log��ͬ�������־�洢�ڲ�ͬ����־�ļ���
    ngx_log_t           *next;
};

void ngx_log_error(ngx_uint_t level, ngx_log_t *log,...);
#endif /* _NGX_LOG_H_INCLUDED_ */

