
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
��4��������������Nginxһϵ�ж�����

NGX_OK����ʾ�ɹ���Nginx�������ִ�и�����ĺ�����������ִ��subrequest����������󣩡�

NGX_DECLINED��������NGX_HTTP_CONTENT_PHASE�׶�Ѱ����һ�����ڸ��������Ȥ��HTTPģ�����ٴδ����������

NGX_DONE����ʾ����Ϊֹ��ͬʱHTTP��ܽ���ʱ���ټ���ִ���������ĺ������֡���ʵ�ϣ���ʱ�������ӵ����ͣ������keepalive���͵��û�����
    �ͻᱣ��סHTTP���ӣ�Ȼ��ѿ���Ȩ����Nginx���������������ã��������³�������һ�����������Ǳ������һ����ʱ�����Ĳ���������ĳ��������ã���
    ����������סNginx������Ϊ����û�аѿ���Ȩ������Nginx��������ngx_http_mytest_handler����Nginx worker���������ˣ���ȴ�����Ļذ�����
    ���ԣ���ͻᵼ��Nginx�����������⣬�ý����ϵ������û�����Ҳ�ò�����Ӧ����������ǰ������ʱ�����Ĳ�����Ϊ�����������֣�����Linux�ں�
    �ж��жϴ���Ļ��֣����ϰ벿�ֺ��°벿�ֶ����������ģ���ʱ���ٵĲ���������������ngx_http_mytest_handler����ʱ�����ϰ벿�֣�Ȼ�󷵻�NGX_DONE��
    �ѿ��ƽ�����Nginx���Ӷ���Nginx�������������������°벿�ֱ�����ʱ�����ﲻ̽�־����ʵ�ַ�ʽ����ʵ��ʹ��upstream��ʽ���������ʱ�õľ���
    ����˼�룩���ٻص��°벿�ִ������������Ϳ��Ա�֤Nginx�ĸ����������ˡ�

NGX_ERROR����ʾ������ʱ�����ngx_http_terminate_request��ֹ�����������POST��������ô������ִ����POST���������ֹ��������
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

//getconf PAGE_SIZE ������Բ鿴
ngx_uint_t  ngx_pagesize;//��ngx_os_init  ����һ����ҳ�Ĵ�С����λΪ�ֽ�(Byte)����ֵΪϵͳ�ķ�ҳ��С����һ�����Ӳ����ҳ��С��ͬ��
//ngx_pagesizeΪ4M��ngx_pagesize_shiftӦ��Ϊ12
ngx_uint_t  ngx_pagesize_shift; //ngx_pagesize������λ�Ĵ�������for (n = ngx_pagesize; n >>= 1; ngx_pagesize_shift++) { /* void */ }
//��ʼ���ο�ngx_init_cycle��������һ��ȫ�����͵�ngx_cycle_s����ngx_cycle
//volatile ngx_cycle_t  *ngx_cycle; //ngx_cycle = cycle;  ��ֵ��main->ngx_init_cycle
ngx_log_t    *logLocal;
ngx_int_t   ngx_ncpu; //cpu����
ngx_pid_t     ngx_pid;//ngx_pid = ngx_getpid(); ���ӽ�����Ϊ�ӽ���pid����master��Ϊmaster��pid







#endif /* _NGX_CORE_H_INCLUDED_ */
