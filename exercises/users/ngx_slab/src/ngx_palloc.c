
/*
 * Copyright (C) Igor Sysoev
 * Copyright (C) Nginx, Inc.
 */


#include <ngx_config.h>
#include <ngx_core.h>


static void *ngx_palloc_block(ngx_pool_t *pool, size_t size);
static void *ngx_palloc_large(ngx_pool_t *pool, size_t size);

/*
ʵ���ϣ���r�п��Ի������ڴ�ض�����Щ�ڴ�صĴ�С�����弰�����ڸ�����ͬ����3���ֻ��漰����ڴ�أ�����ʹ��r->pool�ڴ�ؼ��ɡ�����ngx_pool_t�����
���Դ��ڴ���з����ڴ档
���У�ngx_palloc���������pool�ڴ���з��䵽size�ֽڵ��ڴ棬����������ڴ����ʼ��ַ���������NULL��ָ�룬���ʾ����ʧ�ܡ�
����һ����װ��ngx_palloc�ĺ���ngx_pcalloc����������һ���£����ǰ�ngx_palloc���뵽���ڴ��ȫ����Ϊ0����Ȼ����������¸��ʺ���ngx_pcalloc�������ڴ档
*/ //����ʹ��ngx_palloc���ڴ���л�ȡһ���ڴ�
//ngx_palloc��ngx_palloc�������Ƿ�ƬС���ڴ�ʱ�Ƿ���Ҫ�ڴ����
void *
ngx_palloc(ngx_pool_t *pool, size_t size)
{
    u_char      *m;
    ngx_pool_t  *p;

    // �ж� size �Ƿ���� pool ����ʹ���ڴ��С
    if (size <= pool->max) {

        p = pool->current; //��current���ڵ�pool���ݽڵ㿪ʼ�������Ѱ���Ǹ��ڵ���Է���size�ڴ�

        do {
            m = ngx_align_ptr(p->d.last, NGX_ALIGNMENT);// �� m ���䵽�ڴ�����ַ
            if ((size_t) (p->d.end - m) >= size) {// �ж� pool ��ʣ���ڴ��Ƿ���
                p->d.last = m + size;

                return m;
            }

            p = p->d.next;//�����ǰ�ڴ治����������һ���ڴ���з���ռ�

        } while (p);

        return ngx_palloc_block(pool, size);
    }

    /*
    �����������һ�����������Ҫ���ڴ����pool���ɷ����ڴ��Сʱ����ʱ�����ж�size�Ѿ�����pool->max�Ĵ�С�ˣ�����ֱ�ӵ���ngx_palloc_large���д��ڴ���䣬���ǽ�ע����ת���������
    ��ƪ������Դ�� Linux������վ(www.linuxidc.com)  ԭ�����ӣ�http://www.linuxidc.com/Linux/2011-08/41860.htm
    */
    return ngx_palloc_large(pool, size);
}

//ngx_palloc��ngx_palloc�������Ƿ�ƬС���ڴ�ʱ�Ƿ���Ҫ�ڴ����
void *
ngx_pnalloc(ngx_pool_t *pool, size_t size)
{
    u_char      *m;
    ngx_pool_t  *p;

    if (size <= pool->max) {

        p = pool->current;

        do {
            m = p->d.last;

            if ((size_t) (p->d.end - m) >= size) {
                p->d.last = m + size;

                return m;
            }

            p = p->d.next;

        } while (p);

        return ngx_palloc_block(pool, size);
    }

    return ngx_palloc_large(pool, size);
}

//���ǰ�濪�ٵ�pool�ռ��Ѿ����꣬����¿��ٿռ�ngx_pool_t
static void *
ngx_palloc_block(ngx_pool_t *pool, size_t size)
{
    u_char      *m;
    size_t       psize;
    ngx_pool_t  *p, *new;

    // ��ǰ������ pool �Ĵ�С
    psize = (size_t) (pool->d.end - (u_char *) pool);

    //// ���ڴ�����˵�ǰ���£��·���һ���ڴ�
    m = ngx_memalign(NGX_POOL_ALIGNMENT, psize, pool->log);
    if (m == NULL) {
        return NULL;
    }

    new = (ngx_pool_t *) m;

    new->d.end = m + psize;
    new->d.next = NULL;
    new->d.failed = 0;

    m += sizeof(ngx_pool_data_t);
    m = ngx_align_ptr(m, NGX_ALIGNMENT);
    new->d.last = m + size;

    // �ж��ڵ�ǰ pool �����ڴ��ʧ�ܴ������������ܸ��õ�ǰ pool �Ĵ�����
    // ������� 4 �Σ�������ڴ� pool ���ٴγ��Է����ڴ棬�����Ч��
    //���ʧ�ܴ�������4��������4���������currentָ�룬��������pool���ڴ������ʹ��
    for (p = pool->current; p->d.next; p = p->d.next) {
        if (p->d.failed++ > 4) {
            pool->current = p->d.next;// ���� current ָ�룬 ÿ�δ�pool�з����ڴ��ʱ���Ǵ�curren��ʼ����pool�ڵ��ȡ�ڴ��
        }
    }

    // �þ�ָ���������� next ָ���·���� pool
    p->d.next = new;

    return m;
}

/*
����Ҫ���ڴ����pool���ɷ����ڴ��Сʱ����ʱ�����ж�size�Ѿ�����pool->max�Ĵ�С�ˣ�����ֱ�ӵ���ngx_palloc_large���д��ڴ���䣬
��ƪ������Դ�� Linux������վ(www.linuxidc.com)  ԭ�����ӣ�http://www.linuxidc.com/Linux/2011-08/41860.htm
*/
static void *
ngx_palloc_large(ngx_pool_t *pool, size_t size)
{
    void              *p;
    ngx_uint_t         n;
    ngx_pool_large_t  *large;

    /*
    // ��������һ���СΪ size �����ڴ�
    // ע�⣺�˴���ʹ�� ngx_memalign ��ԭ���ǣ��·�����ڴ�ϴ󣬶���Ҳû̫���Ҫ
    //  ���Һ����ṩ�� ngx_pmemalign ������ר���û���������˵��ڴ�
    */
    p = ngx_alloc(size, pool->log);
    if (p == NULL) {
        return NULL;
    }

    n = 0;

    // ����largt�����Ͽ����large ָ��
    for (large = pool->large; large; large = large->next) {
        if (large->alloc == NULL) { //�������û�õ�large
            large->alloc = p;
            return p;
        }

        /*
         // �����ǰ large �󴮵� large �ڴ����Ŀ���� 3 ��������3����
        // ��ֱ��ȥ��һ���������ڴ棬���ٲ�����
        */
        if (n++ > 3) {//Ҳ����˵���pool->largeͷ��������4��large��allocָ�붼�����ˣ�����������һ���µ�pool_larg���ŵ�pool->largeͷ��
            break; //????? �о�ûɶ�ã���Ϊ����ÿ��alloc��large��Ӧ��alloc���Ǹ�ֵ�˵�
        }
    }

    large = ngx_palloc(pool, sizeof(ngx_pool_large_t));
    if (large == NULL) {
        ngx_free(p);
        return NULL;
    }

    // ���·���� large �����������
    large->alloc = p;
    large->next = pool->large;
    pool->large = large;

    return p;
}




ngx_int_t
ngx_pfree(ngx_pool_t *pool, void *p)
{
    ngx_pool_large_t  *l;

    for (l = pool->large; l; l = l->next) {
        if (p == l->alloc) {
            //ngx_log_debug1(NGX_LOG_DEBUG_ALLOC, pool->log, 0,     "free: %p", l->alloc);
            ngx_free(l->alloc);
            l->alloc = NULL;

            return NGX_OK;
        }
    }

    return NGX_DECLINED;
}


void *
ngx_pcalloc(ngx_pool_t *pool, size_t size)
{
    void *p;

    p = ngx_palloc(pool, size);
    if (p) {
        ngx_memzero(p, size);
    }

    return p;
}

/*
ngx_create_pool������pool
ngx_destory_pool������ pool
ngx_reset_pool������pool�еĲ�������
ngx_palloc/ngx_pnalloc����pool�з���һ���ڴ�
ngx_pool_cleanup_add��Ϊpool���cleanup����
*/ //��pool�з������handler���ڴ棬
/*
�Ի���fileΪ��:

���Կ�����ngx_pool_cleanup_file_t�еĶ�����ngx_buf_t��������file�ṹ���ж����ֹ��ˣ�����Ҳ����ͬ�ġ�����file�ṹ�壬�������ڴ�����Ѿ�Ϊ��������ڴ棬
ֻ�����������ʱ�Ż��ͷţ���ˣ�����򵥵�����file��ĳ�Ա���ɡ������ļ�����������������¡�

if (cln == NULL) {
 return NGX_ERROR;
}

cln->handler = ngx_pool_cleanup_file;
ngx_pool_cleanup_file_t  *clnf = cln->data;

clnf->fd = b->file->fd;
clnf->name = b->file->name.data;
clnf->log = r->pool->log;

ngx_pool_cleanup_add���ڸ���HTTP��ܣ����������ʱ����cln��handler����������Դ��
*///poll��������ngx_pool_cleanup_add, ngx_http_request_t��������ngx_http_cleanup_add








