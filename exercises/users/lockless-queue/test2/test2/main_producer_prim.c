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
#define SHARE_LFQUEUE_FILE  "/var/run/test-sh-lfq"
#define TEST_MAP 0
#if TEST_MAP
int queue_destroy(void* addr, size_t size);
#else
int queue_destroy(int key);
#endif
typedef void (*test_function)(pthread_t*);
void multi_enq_deq(pthread_t *threads);
void running_test(test_function testfn);
void*  consumer(ngx_slab_pool_t  *shpool);
inline void* ngx_lfqueue_alloc(void *pl, size_t sz);
inline void ngx_lfqueue_free(void *pl, void *ptr);
int reset_lfqueue_free(lfqueue_t *lfqueue, lfqueue_free_fn lfqueue_free);
static ngx_int_t
ngx_init_zone_pool(ngx_log_t    *log, ngx_shm_t   *shm);
struct timeval  tv1, tv2;
#define total_put 50000
//#define total_put 50
#define total_running_loop 50
#define USE_SIG_QUIT 1
int nthreads = 1;
int one_thread = 1;
int nthreads_exited = 0;
lfqueue_t *myq;
typedef struct share_lfqueue{
    lfqueue_t lf_q;
    void * shm_addr;
    void * self;
} share_lfqueue_t;
share_lfqueue_t* sh_q;
static volatile uint8_t quit_signal = 0;
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
     fprintf(stdout, "*************** dump lfq info: \n");
     fprintf(stdout, "malloc %p, free %p  \n", lfq->_malloc, lfq->_free);
     fprintf(stdout, "malloc func equal ?  %d, free func equal? %d  \n", lfq->_malloc == ngx_lfqueue_alloc, lfq->_free == ngx_lfqueue_free);
     fprintf(stdout, "head %p, tail %p , mem pool %p \n", lfq->head, lfq->tail, lfq->pl);
}
void*  producer(ngx_slab_pool_t  *shpool)
{
	int i = 0, loop;
	int *int_data;
#if 0
	while (i < total_put) {
#else
	while (!quit_signal) {
#endif
		int_data = (int*)ngx_slab_alloc_locked(shpool,sizeof(int));
		if(NULL == int_data)
		{
		    lfqueue_sleep(1000);
		    continue;
		}
		loop = 5;
		*int_data = i++;
		//printf("producer put data %d \n", *int_data);
		/*Enqueue*/
		while (lfqueue_enq(myq, int_data)) {
			printf("ENQ FULL?\n");
			lfqueue_sleep(10);
			--loop;
			if(loop <= 0)
			{
		             ngx_slab_free_locked(shpool,int_data);
			     break;
			}
		}
	        //dump_lfq(myq);
	}
	dump_lfq(myq);
	return 0;
}
void*  consumer(ngx_slab_pool_t  *shpool)
{
	int  i= 0, loop = 10;
	int *int_data;
	dump_lfq(myq);
	reset_lfqueue_free(myq,ngx_lfqueue_free);
#if 0
	while (i < total_put) {
#else
	while (!quit_signal) {
#endif
		loop = 5;
		/*Dequeue*/
		while ((int_data = lfqueue_deq(myq)) == NULL) {
		        //printf("consumer sleep ");
			sleep(1);

			if(--loop <= 0)
			{
				break;
			}
		}
		if(loop > 0)
		{
		    printf("consumer get %d \n", *int_data);
		}
		//ngx_slab_free_locked(shpool,int_data);
		++ i;
	}
	return 0;
}
/** Worker Send And Consume at the same time **/
void*  worker_sc(void *arg)
{
	int i= 0;
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
void* ngx_lfqueue_alloc(void *pl, size_t sz) {
	//fprintf(stderr,"sp addr %p \n",pl);
 	return ngx_slab_alloc_locked( pl, sz);
 }
void ngx_lfqueue_free(void *pl, void *ptr) {
	        //printf("-----------%s---------------\n", __func__);
 		ngx_slab_free_locked( pl, ptr);
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
int reset_lfqueue_malloc(lfqueue_t *lfqueue, lfqueue_malloc_fn lfqueue_malloc)
{
    lfqueue->_malloc = lfqueue_malloc;
    return 0;
}
int reset_lfqueue_free(lfqueue_t *lfqueue, lfqueue_free_fn lfqueue_free)
{
    lfqueue->_free = lfqueue_free;
    return 0;
}
#if !(TEST_MAP)
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
        //reset_lfqueue(&queue->lf_q, ngx_lfqueue_alloc, ngx_lfqueue_free);
	return queue;
}
#else

lfqueue_t * queue_create(ngx_slab_pool_t *shpool)
{
    lfqueue_t * queue;
    char *addr;
    int alloc_size =  sizeof(share_lfqueue_t); 
    int fd;
    unlink(SHARE_LFQUEUE_FILE);
    fd = open(SHARE_LFQUEUE_FILE, O_CREAT | O_RDWR, 0600);
    if (fd < 0){
    return NULL;
    }
#if 1
    if (ftruncate(fd,  alloc_size) < 0) {
	    goto err1;
    }
 #endif
    addr = (char *) mmap(NULL, alloc_size, PROT_READ|PROT_WRITE,  MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
	goto err1;
    }
    printf("queue addr %p \n",addr);
        sh_q = (share_lfqueue_t*)addr;
	queue = &(sh_q->lf_q);
#if 1
        if (lfqueue_init_mf(queue, shpool, ngx_lfqueue_alloc, ngx_lfqueue_free) == -1) {
#else
        //if (lfqueue_init_mf(queue, pool, malloc, free) == -1) {
        if (lfqueue_init(queue) == -1) {
#endif
	      //ngx_log_error(NGX_LOG_EMERG, cycle->log, 0, " lfqueue Initializing error... ");
	      //return NGX_ERROR;
	      goto err2;
	}
        close(fd);
        return queue;
err2:
	queue_destroy(addr, alloc_size);
err1:
	close(fd);
        return NULL;
}


int queue_destroy(void *addr, size_t size)
{
    if (munmap(addr, size) == -1) {
			        }
    return 0;
}

share_lfqueue_t *queue_open(void * addr, ngx_int_t len,  int offset)
{
    int fd;
    share_lfqueue_t* queue;
    fd = open(SHARE_LFQUEUE_FILE, O_RDWR, 0600);
    if (fd < 0){
    return NULL;
    }
    void * addr2 = mmap(addr, len, PROT_READ | PROT_WRITE, MAP_SHARED , fd, offset);
    if (addr2 == MAP_FAILED || addr != addr2)
    return NULL;
    queue = (share_lfqueue_t*)addr2;
    //reset_lfqueue(&queue->lf_q, ngx_lfqueue_alloc, ngx_lfqueue_free);
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
struct lfqueue_cas_node_s2 {
	void * value;
	struct lfqueue_cas_node_s *next, *nextfree;
	 time_t _deactivate_tm;
};
int init_ngx_env()
{
    int n;
    ngx_pagesize_shift=0;
    ngx_pagesize = getpagesize() ;
    for (n = ngx_pagesize; n >>= 1; ngx_pagesize_shift++) { /* void */ }
    printf("--%d\n",1<<ngx_pagesize_shift);
    return 0;
}
int server() 
{
	char *str = "hello world";
	int len = strlen(str);
	ngx_log_t    log;
	ngx_shm_t   shm;
	ngx_slab_pool_t                *shpool;
	char *ssl_session[10];

       memset(&log,0,sizeof(ngx_log_t));
       memset(&shm,0,sizeof(ngx_shm_t));
      shm.size = SHM_SIZE;
      init_ngx_env();
  printf("ngx_int_t %lu, ngx_uint_t %lu \n",sizeof(ngx_int_t),sizeof(ngx_uint_t));
      if (ngx_shm_alloc(&shm) != NGX_OK) {
	           return 1;
	}
  if(NULL == shm.addr)
  {
          fprintf(stderr,  " shm addr is null...\n ");
	  return -1;
   }
     printf("shm addr %p \n", shm.addr);
  if (ngx_init_zone_pool(&log, &shm)) {
      return 1;
   }
  
  shpool = (ngx_slab_pool_t *) shm.addr;
  shpool->log_nomem = true;
#if 0
  myq = queue_create(&log, shpool);
#else
  //myq = queue_create(shpool);
  myq = queue_create(key, shpool);
  sh_q->shm_addr = shm.addr;
  sh_q->self  = sh_q;
#endif
  if(NULL == myq)
  {
          ngx_log_error(NGX_LOG_EMERG, &log,0,  " lfqueue create error... ");
          exit(-1);
   }
  ssl_session[0] = (char *)ngx_slab_alloc_locked(shpool, len);
  memcpy(ssl_session[0],str,len + 1);
  printf("ssl_session[0] %s \n",ssl_session[0]);
  ngx_slab_free(shpool,ssl_session[0]);
  ssl_session[0] = (char *)ngx_slab_alloc_locked(shpool,  sizeof(struct lfqueue_cas_node_s2));
  ngx_slab_free(shpool,ssl_session[0]);
  producer(shpool);
  //consumer(shpool);
  //running_test(single_enq_deq);
#if 0
  ssl_session[0] = (char *)ngx_slab_calloc(shpool, 56);

  ssl_session[1] = (char *)ngx_slab_alloc_locked(shpool, 14);

  ssl_session[2] = (char *)ngx_slab_alloc_locked(shpool, 11);

  ngx_slab_free(shpool,ssl_session[2]);
  ngx_slab_free(shpool,ssl_session[0]);
  ssl_session[2] = (char *)ngx_slab_alloc_locked(shpool, 65);
#endif
	return 0;
}
int client(unsigned long addr)
{
   ngx_slab_pool_t  *sp;
   init_ngx_env();
#if TEST_MAP
   sh_q = queue_open((void*)addr,sizeof( share_lfqueue_t), 0);
#else
   sh_q = queue_open(key);
#endif
   if(NULL == sh_q)
  {
          fprintf(stderr,  " lfqueue create error...\n ");
	  return -1;
   }
   if(NULL == sh_q->shm_addr)
  {
          fprintf(stderr,  " shm addr is null... \n");
	  return -1;
   }
    if(NGX_OK != ngx_shm_attach(sh_q->shm_addr, SHM_SIZE, 0))
    {
          fprintf(stderr,  " ngx shm attach fail... \n");
	  return -1;
    }
   myq = &(sh_q->lf_q);
   printf("shm addr %p \n", sh_q->shm_addr);

   sp = (ngx_slab_pool_t *) sh_q->shm_addr; 
   consumer(sp);
   //producer(sp);
   return 0;
}
static void
signal_handler(int sig_num)
{
        if (sig_num == SIGINT) {
        printf("\n\nSignal %d received, preparing to exit...\n",
                             sig_num);
        quit_signal = 1;
        }
}

int  main(int argc,char **argv)
{
    int c;
    int mode;
    unsigned long addr = 0;
    while( (c = getopt(argc, argv, "m:a:h")) != EOF ) 
    {
	  if( 'm' == c  ) 
	  {
	       mode = atoi(optarg);
	  }
	  else if( 'a' == c  ) 
	  {
	       //  ./mycc -m 2 -a 0x7f48dd915000
	       addr =  strtoul(optarg,0,16);
	       printf("second process %s and addr 0x%lx\n", optarg, addr);
	  }
    }
    /* catch ctrl-c so we can print on exit */
    signal(SIGINT, signal_handler);
    if(PRIMARY_PROC == mode)
    {
	 
	 printf("primary process \n");
	 server();
    }
    else if(SECONDARY_PROC == mode)
    {
	 printf("second process\n");
#if TEST_MAP
	 if(0 == addr)
         {
	    printf("second process need addr\n");
	    exit(-1);
	 }
#endif
	 client(addr);
    }
    getchar();
    return 0;
}
