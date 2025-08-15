/*
 * author:doop-ymc
 * date:2013-11-11
 * version:1.0
 */

#include <stdio.h>
#include "ngx_config.h"
#include "ngx_conf_file.h"
#include "nginx.h"
#include "ngx_core.h"
#include "ngx_string.h"
#include "ngx_palloc.h"

#define MY_POOL_SIZE 5000

volatile ngx_cycle_t  *ngx_cycle;
#if 0
void 
ngx_log_error_core(ngx_uint_t level, ngx_log_t *log, ngx_err_t err,
    const char *fmt, ...)
{

}
#endif
void 
echo_pool(ngx_pool_t* pool)
{
    int                  n_index;
    ngx_pool_t          *p_pool;
    ngx_pool_large_t    *p_pool_large;

    n_index = 0;
    p_pool = pool;
    p_pool_large = pool->large;

    printf("------------------------------\n");
    printf("pool begin at: 0x%p\n", pool);

    do{
        printf("->d         :0x%p\n", p_pool);
        printf("        last = 0x%p\n", p_pool->d.last);
        printf("        end  = 0x%p\n", p_pool->d.end);
        printf("        next = 0x%p\n", p_pool->d.next);
        printf("      failed = %ld\n", p_pool->d.failed);
        p_pool = p_pool->d.next;
    }while(p_pool);
    printf("->max       :%ld\n", pool->max);
    printf("->current   :0x%p\n", pool->current);
    printf("->chain     :0x%p\n", pool->chain);
    
    if(NULL == p_pool_large){
        printf("->large     :0x%p\n", p_pool_large);
    }else{
        do{
            printf("->large     :0x%p\n", p_pool_large);
            printf("        next = 0x%p\n", p_pool_large->next);
            printf("       alloc = 0x%p\n", p_pool_large->alloc);
            p_pool_large = p_pool_large->next;
        }while(p_pool_large);
    }
    
    printf("->cleanup   :0x%p\n", pool->cleanup);
    printf("->log       :0x%p\n\n\n", pool->log);
    
}

int main()
{
    
    ngx_pool_t *my_pool;
    ngx_log_t log;
    log.log_level = NGX_LOG_DEBUG;
    /*create pool size:5000*/
    my_pool = ngx_create_pool(MY_POOL_SIZE, &log);
    if(NULL == my_pool){
        printf("create nginx pool error,size %d\n.", MY_POOL_SIZE);
        return 0;
    }
    
    printf("+++++++++++CREATE NEW POOL++++++++++++\n");
    echo_pool(my_pool);

    printf("+++++++++++ALLOC 2500+++++++++++++++++\n");
    ngx_palloc(my_pool, 2500);
    echo_pool(my_pool);

    printf("+++++++++++ALLOC 2500+++++++++++++++++\n");
    ngx_palloc(my_pool, 2500);
    echo_pool(my_pool);

    printf("+++++++++++ALLOC LARGE 5000+++++++++++\n");
    ngx_palloc(my_pool, 5000);
    echo_pool(my_pool);

    printf("+++++++++++ALLOC LARGE 5000+++++++++++\n");
    ngx_palloc(my_pool, 5000);
    echo_pool(my_pool);

    ngx_destroy_pool(my_pool);
    return 0;

}
