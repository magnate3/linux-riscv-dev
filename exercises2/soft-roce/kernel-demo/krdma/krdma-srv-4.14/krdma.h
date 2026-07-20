#ifndef __KVM_X86_KRDMA_H
#define __KVM_X86_KRDMA_H
/*
 * Copyright (C) 2019, Trusted Cloud Group, Shanghai Jiao Tong University.
 *
 * Authors:
 *   Yubin Chen <binsschen@sjtu.edu.cn>
 *   Jin Zhang <jzhang3002@sjtu.edu.cn>
 *
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the COPYING file in the top-level directory.
 */
#include <linux/pci.h>
#include <linux/list.h>

#include <rdma/ib_verbs.h>
#include <rdma/rdma_cm.h>

#include "ktcp.h"

#define RDMA_RESOLVE_TIMEOUT 2000
#define RDMA_CONNECT_RETRY_MAX 3

#define RDMA_SEND_QUEUE_DEPTH 1
#define RDMA_RECV_QUEUE_DEPTH 32
#define RDMA_CQ_QUEUE_DEPTH (RDMA_SEND_QUEUE_DEPTH + RDMA_RECV_QUEUE_DEPTH)

#define RDMA_SEND_BUF_SIZE RDMA_SEND_QUEUE_DEPTH
#define RDMA_RECV_BUF_SIZE RDMA_RECV_QUEUE_DEPTH

#define RDMA_SEND_BUF_LEN (PAGE_SIZE * 1024)
#define RDMA_RECV_BUF_LEN (PAGE_SIZE * 1024)
#define RDMA_RDWR_BUF_LEN (PAGE_SIZE * 1024)

typedef uint32_t imm_t;


enum krdma_role {
	KRDMA_CLIENT_CONN = 0,
	KRDMA_LISTEN_CONN = 1,
	KRDMA_ACCEPT_CONN = 2,
};

enum krdma_code {
	SERVER_EXIT = 1000,
	CLIENT_EXIT,
	CLIENT_RETRY,
	STATE_ERROR,
};

typedef enum { 
	KRDMA_SEND,
	KRDMA_RECV,
	KRDMA_READ,
	KRDMA_WRITE
} krdma_poll_type_t;

/*
 * Invalid: the slot has not been used.
 * Posted: the request has been posted into the sq/rq.
 * Polled: the request has been polled from the cq (but not been completed yet).
 */
enum krdma_trans_state { INVALID = 0, POSTED, POLLED };

typedef struct krdma_rw_info {
	void *buf;
	size_t length;
	uint64_t addr;
	uint32_t rkey;
	uint32_t qp_num;
	uint16_t lid;
} __attribute__((packed)) krdma_rw_info_t;

typedef struct krdma_send_trans {
	/* For DMA */
	void *send_buf;
	dma_addr_t send_dma_addr;
	uint16_t txid;
	enum krdma_trans_state state;

	struct ib_sge send_sge;
	struct ib_send_wr sq_wr;
} krdma_send_trans_t;

typedef struct krdma_recv_trans {
	/* For DMA */
	void *recv_buf;
	imm_t imm;
	dma_addr_t recv_dma_addr;
	size_t length;
	uint16_t txid;
	enum krdma_trans_state state;

	struct ib_sge recv_sge;
	struct ib_recv_wr rq_wr;
} krdma_recv_trans_t;

/* control block that supports both RDMA send/recv and read/write */
struct krdma_cb {
	struct mutex slock;
	struct mutex rlock;

	enum krdma_role role;

	enum {
		KRDMA_INIT = 0,
		KRDMA_ADDR_RESOLVED = 1,
		KRDMA_ROUTE_RESOLVED = 2,
		KRDMA_CONNECTED = 3,
		KRDMA_FLUSHING = 4,
		KRDMA_CLOSING = 5,
		KRDMA_CLOSED = 6,
		KRDMA_SEND_DONE = 7,
		KRDMA_RECV_DONE = 8,
		KRDMA_ERROR = 9,
		KRDMA_CONNECT_REJECTED = 10,
		KRDMA_DISCONNECTED = 11,
	} state;

	/* Communication Manager id */
	struct rdma_cm_id *cm_id;

	/* Completion Queue */
	struct ib_cq *send_cq;
	struct ib_cq *recv_cq;
	/* Protection Domain */
	struct ib_pd *pd;
	/* Queue Pair */
	struct ib_qp *qp;

	/*
	 * The buffers to buffer async requests.
	 */
	// Set to false in all send/recv APIs
	bool read_write; // which mr?
	union {
		struct {
			krdma_send_trans_t *send_trans_buf;
			krdma_recv_trans_t *recv_trans_buf;
		} sr_mr;
		struct {
			krdma_rw_info_t *local_info;
			krdma_rw_info_t *remote_info;
		} rw_mr;
	} mr;

	struct completion cm_done;

	struct list_head list;

	struct list_head ready_conn;
	struct list_head active_conn;

	int retry_count;
};

#define DYNAMIC_POLLING_INTERVAL

void krdma_config(size_t max_buf_size);

/* RDMA SEND/RECV APIs */
int krdma_send(struct krdma_cb *cb, const char *buffer, size_t length);

int krdma_receive(struct krdma_cb *cb, char *buffer);

/* Called with remote host & port */
int krdma_connect(const char *host, const char *port, struct krdma_cb **conn_cb);

int krdma_listen(const char *host, const char *port, struct krdma_cb **listen_cb);

int krdma_accept(struct krdma_cb *listen_cb, struct krdma_cb **accept_cb);

/* RDMA READ/WRITE APIs */
/* Called with remote host & port */
int krdma_rw_init_client(const char *host, const char *port, struct krdma_cb **cbp);

int krdma_rw_init_server(const char *host, const char *port, struct krdma_cb **cbp);

int krdma_read(struct krdma_cb *cb, char *buffer, size_t length);

int krdma_write(struct krdma_cb *cb, const char *buffer, size_t length);

/* RDMA release API */
int krdma_release_cb(struct krdma_cb *cb);

#endif /* __KVM_X86_KRDMA_H */
