#include <ngx_config.h>
#include <ngx_core.h>

static ngx_int_t
ngx_init_zone_pool(ngx_log_t    *log, ngx_shm_t   *shm)
{
    
    ngx_slab_pool_t  *sp;

    //共享内存的起始地址开始的sizeof(ngx_slab_pool_t)字节是用来存储管理共享内存的slab poll的
    sp = (ngx_slab_pool_t *) shm->addr; //共享内存起始地址
    
    if (shm->exists) {

        if (sp == sp->addr) {
            return NGX_OK;
        }
        ngx_log_error(NGX_LOG_EMERG, log, 0,"shared zone \"%V\" has no equal addresses: %p vs %p",&shm->name, sp->addr, sp);
        return NGX_ERROR;
    }

    sp->end = shm->addr + shm->size;
    sp->min_shift = 3;
    sp->addr = shm->addr;

    //创建共享内存锁
    if (ngx_shmtx_create(&sp->mutex, &sp->lock, NULL) != NGX_OK) {
        return NGX_ERROR;
    }

    ngx_slab_init(sp);

    return NGX_OK;
}


int  main(int argc,char **argv)
{
	ngx_log_t    log;
	ngx_shm_t   shm;
	ngx_slab_pool_t                *shpool;
	char *ssl_session[10];
       int n;

       memset(&log,0,sizeof(ngx_log_t));
	memset(&shm,0,sizeof(ngx_shm_t));
	shm.size=512000;
	ngx_pagesize_shift=0;
	ngx_pagesize = getpagesize() ;
	for (n = ngx_pagesize; n >>= 1; ngx_pagesize_shift++) { /* void */ }
	printf("--%d\n",1<<ngx_pagesize_shift);
      if (ngx_shm_alloc(&shm) != NGX_OK) {
	           return 1;
	}

  if (ngx_init_zone_pool(&log, &shm)) {
      return 1;
   }

  shpool = (ngx_slab_pool_t *) shm.addr;
  ssl_session[0] = (char *)ngx_slab_alloc_locked(shpool, 12);
  ssl_session[0] = (char *)ngx_slab_calloc(shpool, 56);

  ssl_session[1] = (char *)ngx_slab_alloc_locked(shpool, 14);

  ssl_session[2] = (char *)ngx_slab_alloc_locked(shpool, 11);

  ngx_slab_free(shpool,ssl_session[2]);
  ngx_slab_free(shpool,ssl_session[0]);
  ssl_session[2] = (char *)ngx_slab_alloc_locked(shpool, 65);

	return 0;
}

