//gcc ../src/os/unix/ngx_errno.c ../src/core/ngx_log.c  ../src/core/ngx_string.c ../src/os/unix/ngx_alloc.c ../src/core/ngx_palloc.c pool.c  -o pool  -I../src/core/  -I../src/os/unix/   -I../src/auto 
#include <stdio.h>

#include <ngx_core.h>
#include <ngx_palloc.h>
#include <ngx_string.h>

int main (int argc, char *argv[])
{
    int loop = 10;
    u_char *sp;
    ngx_pool_t *pool;
    ngx_log_t log;
    log.log_level = NGX_LOG_DEBUG;
    size_t sum = 0, alloc_size = NGX_DEFAULT_POOL_SIZE>>1;
    
    pool = ngx_create_pool(NGX_DEFAULT_POOL_SIZE, &log);
    //pool = ngx_create_pool(NGX_DEFAULT_POOL_SIZE, NULL);
    if (pool == NULL) {
        perror("ngx_create_pool() failed.");
        return 1;
    }

    while(--loop > 0)
    {
        sp = ngx_palloc(pool, alloc_size);
        if (sp == NULL) {
            printf("ngx_palloc() failed. looptimers : %d \n", loop);
            goto err1;
        }
        sum +=alloc_size;
        printf("sp        :%p, looptimers : %d, pool size %d, sum size %ld \n",  sp, loop, NGX_DEFAULT_POOL_SIZE,  sum);
    }
err1:
    ngx_destroy_pool(pool);
    ngx_log_debug0(NGX_LOG_DEBUG, pool->log, 0, "ngx pool destroy");
    return 0;
}
