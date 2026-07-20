#include <stdlib.h>
#include <stdio.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
//#include <infiniband/verbs_exp.h>

#define TRUE  1
#define FALSE	0

/* a link in the queue, holds the info and point to the next Node*/
typedef struct Node_t {
    struct ibv_exp_wc *wc;
    struct Node_t *next;
} NODE;

/* the HEAD of the Queue, hold the amount of node's that are in the queue*/
typedef struct Queue {
    NODE *head;
    NODE *tail;
    int size;
    int limit;
} Queue;

Queue *ConstructQueue(int limit);
void DestructQueue(Queue *queue);
int Enqueue(Queue *pQueue, NODE *item);
NODE *Dequeue(Queue *pQueue);
int isEmpty(Queue* pQueue);

