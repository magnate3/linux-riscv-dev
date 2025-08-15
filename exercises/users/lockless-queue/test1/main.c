// Copyright [2020] <Copyright Kevin, kevin.lau.gd@gmail.com>

#include "lfqueue.h"
#include "ncx_slab.h"

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
int queue_destroy(int key);
typedef void (*test_function)(pthread_t*);
void multi_enq_deq(pthread_t *threads);
void running_test(test_function testfn);
struct timeval  tv1, tv2;
#define total_put 50
#define total_running_loop 50
int nthreads = 1;
int one_thread = 1;
int nthreads_exited = 0;
ncx_slab_pool_t *sp;
lfqueue_t *myq;

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
 	return ncx_slab_alloc( pl, sz);
 }
static inline void ngx_lfqueue_free(void *pl, void *ptr) {
 		ncx_slab_free( pl, ptr);
}


lfqueue_t * queue_create(const int key,const  uint64_t data_size)
{
        int shmid;
	lfqueue_t * queue;
        char *m;
	u_char *pool;
        int alloc_size = data_size + sizeof(lfqueue_t); 
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
        queue = (lfqueue_t*)m;
#if 1
	pool = (u_char *)(m + sizeof(lfqueue_t));
	sp = (ncx_slab_pool_t*) pool;
#else
	pool = (u_char *)malloc(data_size);
	sp = (ncx_slab_pool_t*) pool;
#endif
	sp->addr = pool;
	sp->min_shift = 3;
	sp->end = pool + data_size;
	ncx_slab_init(sp);
	fprintf(stderr,"sp addr %p \n",sp);
#if 1
        if (lfqueue_init_mf(queue, sp, ngx_lfqueue_alloc, ngx_lfqueue_free) == -1) {
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

lfqueue_t *queue_open(int key)
{
        int shmid;
        char *m;
	lfqueue_t * queue;
        if ((shmid = shmget(key, 0, 0)) < 0)
                return NULL;

        if ((m = shmat(shmid, NULL, 0)) == NULL)
                return NULL;
        queue = (lfqueue_t*)m;
	return queue;
}

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
int main()
{
    ncx_slab_stat_t stat;
    int key = 1236;
    myq = queue_create(key,getpagesize()*1024);
    //myq = queue_create(key,1024*1024);
    if(NULL == myq)
    {
	   fprintf(stderr,"create queue fail \n");
	   exit(-1);
    }
    ncx_slab_stat(sp, &stat);
    running_test(single_enq_deq);
    //running_test(multi_enq_deq);
    queue_destroy(key);
    return 0;
}
