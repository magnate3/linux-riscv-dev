// Copyright [2020] <Copyright Kevin, kevin.lau.gd@gmail.com>

#include "lfqueue.h"
#include "ngx_config.h"
#include "ngx_core.h"

#include <unistd.h>
#include <sys/shm.h>
#include <stddef.h>
#include <pthread.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>
#define PRIMARY_PROC 1
#define SECONDARY_PROC 2
int queue_destroy(int key);
typedef void (*test_function)(pthread_t*);
void multi_enq_deq(pthread_t *threads);
void running_test(test_function testfn);
void*  consumer(ngx_slab_pool_t  *shpool);
static ngx_int_t
ngx_init_zone_pool(ngx_log_t    *log, ngx_shm_t   *shm);
struct timeval  tv1, tv2;
#define total_put 500
//#define total_put 50
#define total_running_loop 50
int nthreads = 1;
int one_thread = 1;
int nthreads_exited = 0;
lfqueue_t *myq;
typedef struct share_lfqueue{
    lfqueue_t lf_q;
    void * shm_addr;
} share_lfqueue_t;
share_lfqueue_t* sh_q;
#define join_threads \
for (i = 0; i < nthreads; i++) {\
pthread_join(threads[i], NULL); \
}

#define detach_thread_and_loop \
for (i = 0; i < nthreads; i++)\
pthread_detach(threads[i]);\
while ( nthreads_exited < nthreads ) \
	lfqueue_sleep(10);\
if(lfqueue_size(myq) != 0){\
lfqueue_sleep(10);\
}
void dump_lfq(lfqueue_t *lfq)
{
     fprintf(stdout, "head %p, tail %p \n", lfq->head, lfq->tail);
}
void*  producer(ngx_slab_pool_t  *shpool)
{
	int i = 0;
	int *int_data;
	while (i < total_put) {
		int_data = (int*)ngx_slab_alloc_locked(shpool,sizeof(int));
		assert(int_data != NULL);
		*int_data = i++;
		printf("producer put data %d \n", *int_data);
		/*Enqueue*/
		while (lfqueue_enq(myq, int_data)) {
			printf("ENQ FULL?\n");
		}
	        dump_lfq(myq);
	}
	consumer(shpool);
	return 0;
}
void*  consumer(ngx_slab_pool_t  *shpool)
{
	int i = 0;

	int *int_data;
	dump_lfq(myq);
	while (i < total_put) {
		/*Dequeue*/
		while ((int_data = lfqueue_deq(myq)) == NULL) {
		        //printf("consumer sleep ");
			lfqueue_sleep(1);
		}
		printf("consumer get %d \n", *int_data);
		ngx_slab_free(shpool,int_data);
	}
	return 0;
}
/** Worker Send And Consume at the same time **/
void*  worker_sc(void *arg)
{
	int i = 0;
	int *int_data;
	while (i < total_put) {
		int_data = (int*)malloc(sizeof(int));
		assert(int_data != NULL);
		*int_data = i++;
		/*Enqueue*/
		while (lfqueue_enq(myq, int_data)) {
			printf("ENQ FULL?\n");
		}

		/*Dequeue*/
		while ((int_data = lfqueue_deq(myq)) == NULL) {
			lfqueue_sleep(1);
		}
		// printf("%d\n", *int_data);
		free(int_data);
	}
	__sync_add_and_fetch(&nthreads_exited, 1);
	return 0;
}
void single_enq_deq(pthread_t *threads) {
	printf("-----------%s---------------\n", "multi_enq_deq");
	int i;
	for (i = 0; i < 1; i++) {
		pthread_create(threads + i, NULL, worker_sc, NULL);
	}

	join_threads;
	// detach_thread_and_loop;
}
void multi_enq_deq(pthread_t *threads) {
	printf("-----------%s---------------\n", "multi_enq_deq");
	int i;
	for (i = 0; i < nthreads; i++) {
		pthread_create(threads + i, NULL, worker_sc, NULL);
	}

	join_threads;
	// detach_thread_and_loop;
}
static inline void* ngx_lfqueue_alloc(void *pl, size_t sz) {
	//fprintf(stderr,"sp addr %p \n",pl);
 	return ngx_slab_alloc( pl, sz);
 }
static inline void ngx_lfqueue_free(void *pl, void *ptr) {
 		ngx_slab_free( pl, ptr);
}

static ngx_int_t
ngx_init_zone_pool(ngx_log_t    *log, ngx_shm_t   *shm)
{
    
    ngx_slab_pool_t  *sp;

    //¹²ÏíÄÚ´æµÄÆðÊ¼µØÖ·¿ªÊ¼µÄsizeof(ngx_slab_pool_t)×Ö½ÚÊÇÓÃÀ´´æ´¢¹ÜÀí¹²ÏíÄÚ´æµÄslab pollµÄ
    sp = (ngx_slab_pool_t *) shm->addr; //¹²ÏíÄÚ´æÆðÊ¼µØÖ·
    
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

    //´´½¨¹²ÏíÄÚ´æËø
    if (ngx_shmtx_create(&sp->mutex, &sp->lock, NULL) != NGX_OK) {
        return NGX_ERROR;
    }

    ngx_slab_init(sp);

    return NGX_OK;
}

#if 0
lfqueue_t * queue_create(ngx_log_t    *log, ngx_slab_pool_t *shpool)
{
	lfqueue_t * queue;
        queue = (lfqueue_t*)ngx_slab_alloc_locked(shpool, sizeof(lfqueue_t));
	if(NULL == queue)
	{
	    return NULL;
	}
#if 1
        if (lfqueue_init_mf(queue, shpool, ngx_lfqueue_alloc, ngx_lfqueue_free) == -1) {
#else
        //if (lfqueue_init_mf(queue, pool, malloc, free) == -1) {
        if (lfqueue_init(queue) == -1) {
#endif
	      ngx_log_error(NGX_LOG_EMERG, log,0,  " lfqueue Initializing error... ");
	      goto err1;
	}
        return queue;
err1:
        return NULL;
}


lfqueue_t *queue_open(int key)
{
	return NULL;
}
#else

lfqueue_t * queue_create(const int key, ngx_slab_pool_t *shpool)
{
        int shmid;
	lfqueue_t * queue;
        char *m;
        int alloc_size =  sizeof(share_lfqueue_t); 
        shmid = shmget(key, 0, 0);
	if (shmid != -1)
	{
	     shmctl(shmid, IPC_RMID, NULL); 	//	删除已经存在的共享内存
	}
        if ((shmid = shmget(key, alloc_size, IPC_CREAT | IPC_EXCL | 0666)) < 0)
	{
	        fprintf(stderr,"fhmget fail %d \n",shmid);
                return NULL;
        }
        if ((m = (char *)shmat(shmid, NULL, 0)) == NULL)
	{
                return NULL;
        }
        sh_q = (share_lfqueue_t*)m;
	queue = &(sh_q->lf_q);
#if 1
        if (lfqueue_init_mf(queue, shpool, ngx_lfqueue_alloc, ngx_lfqueue_free) == -1) {
#else
        //if (lfqueue_init_mf(queue, pool, malloc, free) == -1) {
        if (lfqueue_init(queue) == -1) {
#endif
	      //ngx_log_error(NGX_LOG_EMERG, cycle->log, 0, " lfqueue Initializing error... ");
	      //return NGX_ERROR;
	      goto err1;
	}
        return queue;
err1:
        queue_destroy(key);       
        return NULL;
}


int queue_destroy(int key)
{
        int shmid = shmget(key, 0, 0);
        if (shmid < 0)
                return -1;
        return shmctl(shmid, IPC_RMID, NULL);
}

share_lfqueue_t *queue_open(int key)
{
        int shmid;
        char *m;
	share_lfqueue_t * queue;
        if ((shmid = shmget(key, 0, 0)) < 0)
                return NULL;

        if ((m = shmat(shmid, NULL, 0)) == NULL)
                return NULL;
        queue = (share_lfqueue_t*)m;
	return queue;
}
#endif
void running_test(test_function testfn) {
	int n;
	for (n = 0; n < total_running_loop; n++) {
		printf("Current running at =%d, ", n);
		nthreads_exited = 0;
		/* Spawn threads. */
		pthread_t threads[nthreads];
		printf("Using %d thread%s.\n", nthreads, nthreads == 1 ? "" : "s");
		printf("Total requests %d \n", total_put);
		gettimeofday(&tv1, NULL);

		testfn(threads);
		// one_enq_and_multi_deq(threads);

		//one_deq_and_multi_enq(threads);
		// multi_enq_deq(threads);
		// worker_s(&ri);
		// worker_c(&ri);

		gettimeofday(&tv2, NULL);
		printf ("Total time = %f seconds\n",
		        (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
		        (double) (tv2.tv_sec - tv1.tv_sec));

		lfqueue_sleep(10);
		assert ( 0 == lfqueue_size(myq) && "Error, all queue should be consumed but not");
	}
}
const int key = 99333;
#define SHM_SIZE (getpagesize() *1024)
int server() 
{
	char *str = "hello world";
	int len = strlen(str);
	ngx_log_t    log;
	ngx_shm_t   shm;
	ngx_slab_pool_t                *shpool;
	char *ssl_session[10];
       int n;

       memset(&log,0,sizeof(ngx_log_t));
       memset(&shm,0,sizeof(ngx_shm_t));
	ngx_pagesize_shift=0;
	ngx_pagesize = getpagesize() ;
	for (n = ngx_pagesize; n >>= 1; ngx_pagesize_shift++) { /* void */ }
	printf("--%d\n",1<<ngx_pagesize_shift);
      shm.size = SHM_SIZE;
      if (ngx_shm_alloc(&shm) != NGX_OK) {
	           return 1;
	}
     printf("shm addr %p \n", shm.addr);
  memcpy(shm.addr,str,len + 1);
  printf("shm content %s \n",shm.addr);
	return 0;
}
int client()
{
   ngx_slab_pool_t  *sp;
   sh_q = queue_open(key);
   if(NULL == sh_q)
  {
          fprintf(stderr,  " lfqueue create error... ");
	  return -1;
   }
    if(NGX_OK != ngx_shm_attach(sh_q->shm_addr, SHM_SIZE, 0))
    {
          fprintf(stderr,  " ngx shm attach fail... ");
	  return -1;
    }
   myq = &(sh_q->lf_q);
   printf("shm addr %p \n", sh_q->shm_addr);
   printf("shm addr content %s \n", sh_q->shm_addr);

   return 0;
}
int  main(int argc,char **argv)
{
    int c;
    int mode;
    while( (c = getopt(argc, argv, "m:h")) != EOF ) 
    {
	  if( 'm' == c  ) 
	  {
	       mode = atoi(optarg);
	  }
    }
    if(PRIMARY_PROC == mode)
    {
	 
	 printf("primary process \n");
	 server();
    }
    else if(SECONDARY_PROC == mode)
    {
	 printf("second process\n");
	 client();
    }
    getchar();
    return 0;
}
