#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <semaphore.h>
#include <sys/mman.h>

#define MMAP_SIZE 128
static volatile int count = 0;


typedef struct {
	unsigned long  *lock;
	unsigned int   spin;
	uintptr_t     semaphore;
    sem_t          sem;
} ngx_shmtx_t;

ngx_shmtx_t  ngx_accept_mutex;

typedef struct {
	unsigned char    *addr;
    size_t             size;
    unsigned char    *name;
} ngx_shm_t;


void ngx_shm_alloc(ngx_shm_t *shm)
{
    shm->addr = (unsigned char *) mmap(NULL, shm->size, PROT_READ|PROT_WRITE, MAP_ANON|MAP_SHARED, -1, 0);

    if (shm->addr == MAP_FAILED) {
        return ;
    }
    return ;
}
void ngx_shm_free()
{

    if (ngx_accept_mutex.lock == MAP_FAILED) {
        return ;
    }
     munmap(ngx_accept_mutex.lock,  MMAP_SIZE); 
    return ;
}

void ngx_shmtx_create(ngx_shmtx_t *mtx, void *addr)
{
    mtx->lock = (unsigned long *) addr;

    if (mtx->spin == (uintptr_t) -1) {
        return ;
    }
    mtx->spin = 2048;
    return ;
}

void ngx_event_module_init()
{
	ngx_shm_t shm;
    shm.size = MMAP_SIZE;
    shm.name = (unsigned char *) "nginx_shared_zone";
    ngx_shm_alloc(&shm);

    ngx_shmtx_create(&ngx_accept_mutex, shm.addr);
    return;
}
void ngx_event_module_destroy()
{
    ngx_shm_free();
    return;
}

unsigned int ngx_shmtx_trylock(ngx_shmtx_t *mtx)
{
	unsigned long  val;

    val = *mtx->lock;

    return ((val & 0x80000000) == 0
            && __sync_bool_compare_and_swap(mtx->lock, val, val | 0x80000000));
}

int ngx_shmtx_unlock(ngx_shmtx_t *mtx)
{
	unsigned long  val, old, wait;

    for ( ;; )
    {
        old = *mtx->lock;
        wait = old & 0x7fffffff;
        val = wait ? wait - 1 : 0;

        if (__sync_bool_compare_and_swap(mtx->lock, old, val)) {
        	//对比mtx->lock 与 old是否相等,如果相等则将mtx->lock设置为val
            break;
        }
    }
}

/*
  自旋锁（__sync_bool_compare_and_swap模拟的）在冲突很高的情况下，效率会明显的低于互斥锁，因为冲突不断的发生，
  自旋锁相当于不断的竞争，相当于死锁（在多核情况下）。而互斥锁则是将冲突线程挂起，避免了继续的竞争。
 * */
void *test_func(void *arg)
{
	//printf("%zu --> %u \r\n", count, pthread_self());
	int i = 0;
	for(i = 0; i < 2000000; i++)
	{
		 while (!ngx_shmtx_trylock(&ngx_accept_mutex))
		 {
			 usleep(i%10);
		 }
		 count++;
		 ngx_shmtx_unlock(&ngx_accept_mutex);
	}
}

int main(int argc, const char *argv[])
{
	ngx_event_module_init();
    pthread_t thread_ids[10];
    int i = 0;

    for(i = 0; i < sizeof(thread_ids)/sizeof(pthread_t); i++)
    {
         pthread_create(&thread_ids[i], NULL, test_func, NULL);
    }

    for(i = 0; i < sizeof(thread_ids)/sizeof(pthread_t); i++)
    {
         pthread_join(thread_ids[i], NULL);
    }

    printf("结果:count = %d\n",count);
    ngx_event_module_destroy();

    return 0;
}
