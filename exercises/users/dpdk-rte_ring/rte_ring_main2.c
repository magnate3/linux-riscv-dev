#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "rte_ring.h"

#define RING_SIZE 16<<20

typedef struct cc_queue_node {
    int data;
} cc_queue_node_t;

static struct rte_ring *r;

typedef unsigned long long ticks;
#define X86_PLAT 1
#ifdef X86_PLAT
static __inline__ ticks getticks(void)
{
    u_int32_t a, d;

    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    return (((ticks)a) | (((ticks)d) << 32));
}
#else
static __inline__ ticks getticks(void)
{
        uint64_t tsc;

        asm volatile("mrs %0, pmccntr_el0" : "=r"(tsc));
        return tsc;
}
#endif

void *enqueue_fun(void *data)
{
    int n = *(int*)data;
    int i = 0;
    int ret;
    cc_queue_node_t *p;
    int sum = 0;
    for (; i < n; i++) {
        p = (cc_queue_node_t *)malloc(sizeof(cc_queue_node_t));
        p->data = i;
	sum +=i;
        ret = rte_ring_mp_enqueue(r, p);
        if (ret != 0) {
            printf("enqueue failed: %d\n", i);
        }
    }
    return NULL;
}

void *dequeue_func(void *data)
{
    int ret;
    int i = 0;
    uint64_t sum = 0;
    int n = *(int*)data;
    cc_queue_node_t *p;
    ticks t1, t2, diff;
    //return;

    t1 = getticks();
    while (1) {
        p = NULL;
        ret = rte_ring_sc_dequeue(r, (void **)&p);
        if (ret != 0) {
            //do something
        }
        if (p != NULL) {
            i++;
            sum += p->data;
            free(p);
            if (i == n) {
                break;
            }
        }
    }

    t2 = getticks();
    diff = t2 - t1;
    printf("time diff: %llu\n", diff);
    printf("dequeue total: %d, sum: %lu\n", i, sum);
    return NULL;
}


int main(int argc, char *argv[])
{
    int ret = 0;
    int num1 = 2000, num2=num1*5;
    uint64_t sum = 0;
    pthread_t pid1, pid2, pid3, pid4, pid5, pid6;
    pthread_attr_t pthread_attr;

    r = rte_ring_create("test", RING_SIZE, 0);

    if (r == NULL) {
        return -1;
    }

    printf("start enqueue, 5 producer threads, echo thread enqueue %d numbers.\n",num1);

    pthread_attr_init(&pthread_attr);
    if ((ret = pthread_create(&pid1, &pthread_attr, enqueue_fun, (void *)&num1)) == 0) {
        pthread_detach(pid1);
    }

    if ((ret = pthread_create(&pid2, &pthread_attr, enqueue_fun, (void *)&num1)) == 0) {
        pthread_detach(pid2);
    }

    if ((ret = pthread_create(&pid3, &pthread_attr, enqueue_fun, (void *)&num1)) == 0) {
        pthread_detach(pid3);
    }
    
    if ((ret = pthread_create(&pid4, &pthread_attr, enqueue_fun, (void *)&num1)) == 0) {
        pthread_detach(pid4);
    }

    if ((ret = pthread_create(&pid5, &pthread_attr, enqueue_fun, (void *)&num1)) == 0) {
        pthread_detach(pid5);
    }
    sum = (0 + num1 -1 )*num1;
    sum = (sum>>1)*5;
    printf("expected sum: %lu\n", sum);
    //printf("expected sum: %d\n", (((0 + num1 -1 )*num1) >>1)*5);

    printf("start dequeue, 1 consumer thread.\n");

    if ((ret = pthread_create(&pid6, &pthread_attr, dequeue_func, (void *)&num2)) == 0) {
        //pthread_detach(pid6);
    }
    
    pthread_join(pid6, NULL);

    rte_ring_free(r);

    return 0;
}
