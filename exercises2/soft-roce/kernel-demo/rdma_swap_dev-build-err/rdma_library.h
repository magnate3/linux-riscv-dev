/*
 * RDMA Client Library (Kernel Space)
 * Not concurrent
 */

#ifndef _RDMA_LIB_H_
#define _RDMA_LIB_H_

#include <rdma/ib_verbs.h>
#include <rdma/rdma_cm.h>
#include <linux/blkdev.h>
#include "conf.h"

typedef struct rdma_ctx* rdma_ctx_t;
typedef struct rdma_request* rdma_req_t;

typedef enum {RDMA_READ,RDMA_WRITE} RDMA_OP;

typedef struct rdma_request
{
    RDMA_OP rw;
    u64 dma_addr;
    uint64_t remote_offset;
    uint32_t length;
    struct batch_request* batch_req;
} rdma_request;

typedef struct batch_request
{
    int id;
#if CUSTOM_MAKE_REQ_FN
    struct bio *bio;
#else
    volatile struct request * req;
#endif
    int nsec;
    volatile int outstanding_reqs;
    volatile struct batch_request* next;
#if MODE == MODE_ONE
    volatile bool all_request_sent;
    volatile int comp_reqs;
#endif
#if MEASURE_LATENCY
    unsigned long long start_time;
    bool first;
#endif
} batch_request;

static inline uint64_t get_cycle(void){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

typedef struct batch_request_pool
{
    struct request** io_req;

    struct batch_request** all;
    struct batch_request** data;
    int size;
    int head;
    int tail;
    spinlock_t lock;
#if MEASURE_LATENCY
    int latency_dist[LATENCY_BUCKET];
#endif
} batch_request_pool;

struct rdma_ctx {
    struct socket *sock;
    char server_addr[100];
    int server_port;
   
    struct ib_cq* send_cq, *recv_cq;
    struct ib_pd* pd;
    struct ib_qp* qp;
    struct ib_qp_init_attr qp_attr;
    struct ib_mr *mr;
    int rkey;

    int lid;
    int qpn;
    int psn;

    char* rdma_recv_buffer;
    u64 dma_addr;
    unsigned long long int rem_mem_size;

    int rem_qpn;
    int rem_psn;
    int rem_lid;
   
    unsigned long long int rem_vaddr;
    uint32_t rem_rkey;

    atomic64_t wr_count;

    volatile unsigned long outstanding_requests;
    //atomic_t outstanding_requests;
    wait_queue_head_t queue;
    atomic_t operation_count;
    wait_queue_head_t queue2;
    atomic_t comp_handler_count;

    struct batch_request_pool* pool;
};

batch_request_pool* get_batch_request_pool(int size);
void destroy_batch_request_pool(batch_request_pool* pool);
batch_request* get_batch_request(batch_request_pool* pool);
void return_batch_request(batch_request_pool* pool, batch_request* req);

void debug_pool_insert(struct batch_request_pool* pool, struct request* req);
void debug_pool_remove(struct batch_request_pool* pool, struct request* req);


u64 rdma_map_address(void* addr, int length);
void rdma_unmap_address(u64 addr, int length);
int rdma_library_ready(void);

int rdma_library_init(void);
int rdma_library_exit(void);

rdma_ctx_t rdma_init(int npages, char* ip_addr, int port, int mem_pool_size);
int rdma_exit(rdma_ctx_t);

int rdma_op(rdma_ctx_t ctx, rdma_req_t req, int n_requests);
void make_wr(rdma_ctx_t ctx, struct ib_send_wr* wr, struct ib_sge *sg, RDMA_OP op, u64 dma_addr, uint64_t remote_offset, uint length, struct batch_request* batch_req);
void simple_make_wr(rdma_ctx_t ctx, struct ib_send_wr* wr, struct ib_sge *sg, RDMA_OP op, u64 dma_addr, uint64_t remote_offset, uint length, struct batch_request* batch_req);
bool merge_wr(struct ib_send_wr* old_wr, struct ib_sge *old_sg, struct ib_send_wr* new_wr, struct ib_sge *new_sg);
void poll_cq(rdma_ctx_t ctx);
#endif // _RDMA_LIB_H_


