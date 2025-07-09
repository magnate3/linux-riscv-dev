/*
 * Copyright (c) 2005 Mellanox Technologies. All rights reserved.
 */

#ifndef SHRSRC_TEST_H
#define SHRSRC_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <limits.h>
#include <arpa/inet.h>
#include <poll.h>
#include <ctype.h>
#include <pthread.h>
#include <signal.h> 
#include <unistd.h> 
#include <assert.h>
#include <infiniband/verbs.h>
#include "rdma_utils.h"

#define RDMA_DEBUG
#define UNUSED __attribute__ ((unused))
#define ATTR_PACKED __attribute__ ((packed))

#define MAX_QP_NUM				512
#define DEFAULT_CQ_SIZE			1024;

#define DEF_MR_SIZE				(1 << 22)
#define DEF_SG_SIZE				4096
#define DEF_BUF_SIZE			(1 << 10)

#define RDMA_BUF_UD_ADDITION    40
#define RDMA_BUF_LOCAL_INFO_SZ  24
#define RDMA_BUF_HDR_SIZE		128

#define DEF_PKEY_IX				0
#define DEF_RNR_NAK_TIMER		0x12
#define DEF_SL					0
#define DEF_STATIC_RATE			0
#define MAX_ASYNC_EVENT_VALUE	32
#define LAT_LEVEL				14

enum post_enum_t {
	POST_MODE_ONE_BY_ONE,
	POST_MODE_LIST
};

struct thread_context_t {
	struct rdma_resource_t    *rdma_resource;
	pthread_t          thread;
	struct sock_t      sock;
	uint32_t           thread_id;
	uint64_t           imm_qp_ctr_snd;
	uint32_t           is_requestor;
	uint32_t           num_of_iter;

#if 0
	uint32_t           lat[LAT_LEVEL]; /// error when LAT_LEVEL > 12, why
	cycles_t           *t_a;
	cycles_t           *t_b;
	cycles_t           *t_c;
#endif

#if 0
	double             max_lat;
	uint32_t           max_lat_iter_num;
	double             min_lat;
	uint32_t           min_lat_iter_num;
#endif

	struct ibv_cq           *send_cq;
	struct ibv_cq           *recv_cq;
	struct ibv_comp_channel *send_comp_channel;
	struct ibv_comp_channel *recv_comp_channel;
	struct ibv_qp		    *qp;	
	enum ibv_qp_type	    qp_type;

	struct ibv_qp_attr  qp_attr;
	struct ibv_ah       *ud_av_hdl;

	uint32_t qkey;
	uint32_t psn;
	uint32_t remote_qpn;
	uint32_t remote_qkey;
	uint32_t remote_psn;
	uint16_t remote_lid;

	// memory pool
	void* buff;
	void* buff_aligned;
	struct ibv_mr *local_mr;
	pthread_mutex_t mr_mutex;
	pthread_mutex_t pending_mr_mutex;
	struct rdma_buf_t* head;
	struct rdma_buf_t* tail;
	struct rdma_buf_t* pending_head;
	struct rdma_buf_t* pending_tail;
	double             max_lat;
	uint32_t           max_lat_iter_num;
	double             min_lat;
	uint32_t           min_lat_iter_num;

	uint32_t           lat[LAT_LEVEL];
	cycles_t           *t_a;
	cycles_t           *t_b;
	cycles_t           *t_c;
};


struct rdma_req_t {
	struct rdma_buf_t *rdma_buf;
	struct thread_context_t *t_ctx;
	uint32_t num_of_oust;
	uint32_t post_mode;

	uint32_t thread_id;
	uint32_t is_requestor;
	uint32_t data_size;
	enum ibv_wr_opcode	opcode;

	union {
		struct ibv_mr	   *local_mr;
		struct ibv_mr	   *remote_mr;
		struct ibv_mr      peer_mr;
	};

	/// Reserved for RDMA RD/WR
	struct ibv_mr	   *remote_mr;
	void			   *peer_buf;
};


typedef void* (*thread_func_t)(void *);


struct async_event_thread_context_t {
	pthread_t thread;
	struct ibv_context *ibv_ctx;
	int expected_event_arr[MAX_ASYNC_EVENT_VALUE];
	int thread_id;
};


struct user_param_t {
	char*    hca_id;
	uint8_t  ib_port;
	char*    server_ip;

	const char         *ip;
	unsigned short int tcp_port;
	uint32_t           ts_mask;
	enum ibv_wr_opcode opcode;
	uint32_t           sl;
	uint32_t           interval;

	uint32_t num_of_iter;
	uint32_t num_of_thread;
	uint32_t num_of_cq;
	uint32_t num_of_srq;
	uint32_t num_of_qp;
	uint32_t num_of_qp_min;
	uint32_t num_of_oust;
	uint32_t buffer_size;
	uint32_t cq_size;
	enum ibv_qp_type qp_type;

	uint32_t max_send_wr;
	uint32_t max_recv_wr;
	uint32_t sq_max_inline;
	uint32_t test_flags;
	uint32_t sq_sig_all;
	uint32_t debug_level;

	uint32_t max_recv_size;
	uint32_t max_send_size;
	uint32_t max_send_sge;
	uint32_t max_recv_sge;
	uint32_t size_per_sg;

	uint8_t path_mtu;
	uint8_t qp_timeout;
	uint8_t qp_retry_count;
	uint8_t qp_rnr_timer;
	uint8_t qp_rnr_retry;
	uint32_t comp_timeout;
	uint32_t rr_post_delay;

	uint8_t direction; 
	uint32_t num_of_transaction;
	uint32_t use_event;
};


struct rdma_buf_t {
	int32_t buf_idx;
	uint32_t thread_idx; ///
	struct rdma_buf_t* cur;
	struct rdma_buf_t* next;
	char ud_addition[RDMA_BUF_UD_ADDITION];
	uint32_t status;
	uint32_t slid;
	uint32_t dlid;
	uint32_t sqpn;
	uint32_t dqpn;
	uint64_t sqn;
	char pad[32];
};


struct rdma_resource_t {
	struct user_param_t user_param;
	double              freq_mhz;

	uint16_t             local_lid;
	struct ibv_device    **dev_list;
	struct ibv_context   *ib_ctx;
	struct ibv_port_attr port_attr;
	struct ibv_pd        *pd;
	struct qp_context_t  *qp_ctx_arr;
	struct cq_context_t  *cq_ctx_arr;
	struct ibv_srq       **srq_hdl_arr;

	struct thread_context_t *client_ctx;
	struct async_event_thread_context_t async_event_thread;

	uint32_t           lat[LAT_LEVEL];
	double             max_lat;
	uint32_t           max_lat_iter_num;
	double             min_lat;
	uint32_t           min_lat_iter_num;
};

extern uint32_t volatile stop_all;
extern uint32_t Debug;
#define DEBUG(...) do { if (Debug > 3) { printf(__VA_ARGS__); }} while (0)
#define TRACE(...) do { if (Debug > 2) { printf(__VA_ARGS__); }} while (0)
#define INFO(...)  do { if (Debug > 1) { printf(__VA_ARGS__); }} while (0)
#define ERROR(...) do { { printf(__VA_ARGS__); }} while (0)

void print_completion(int thread_id, int ts_type, const struct ibv_wc *compl_data);
int rdma_resource_init(struct sock_t* sock, struct rdma_resource_t*);
int rdma_resource_destroy(struct rdma_resource_t*);
int post_receive(struct thread_context_t *t_ctx, struct rdma_req_t *rdma_req);
int post_send(struct thread_context_t *t_ctx, struct rdma_req_t *rdma_req);
int wait_completion_and_check_validation(struct thread_context_t *t_ctx,
	struct rdma_req_t *rdma_req);
int do_validation_check(struct rdma_req_t *rdma_req, uint32_t oust_idx);
void *async_event_thread(void *ptr);
void *cq_poll_thread(void *ptr);
int start_rdma_threads(struct sock_t*, struct rdma_resource_t*, struct sock_bind_t*);
int create_rdma_buf_pool(struct thread_context_t *);
struct rdma_buf_t* get_rdma_buf(struct thread_context_t*);
void* get_rdma_payload_buf(struct rdma_buf_t *);
void put_rdma_buf(struct thread_context_t*, struct rdma_buf_t*);

int create_qp(struct thread_context_t *t_ctx);
int connect_qp(struct thread_context_t *t_ctx);
void destroy_qp(struct thread_context_t *t_ctx);

#endif /* SHRSRC_TEST_H */

