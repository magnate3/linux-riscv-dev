#ifndef MY_RAD_HEADER
#define MY_RAD_HEADER
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>
#include <infiniband/arch.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>

#define MAX_SEND_WR		4
#define MAX_SEND_SGE	1
#define MAX_RECV_WR		4
#define MAX_RECV_SGE	1
#define MAX_INLINE_DATA	256
#define REGION_LENGTH	512
#define SERVER_MR_SIZE	4096

enum completion_type {
	RECV,
	SEND
};

enum client_opcodes {
	DISCONNECT = 1,
	WRITE,
	READ
};

uint32_t get_completion(struct rdma_cm_id *, enum completion_type, uint8_t);
struct rdma_cm_id *cm_event(struct rdma_event_channel *, enum rdma_cm_event_type);
int swap_info(struct rdma_cm_id *, struct ibv_mr *, uint32_t *, uint64_t *, size_t *);
int obliterate(struct rdma_cm_id *,struct rdma_cm_id *, struct ibv_mr *,
	struct rdma_event_channel *);
void stop_it(char *, int);
void rdma_recv(struct rdma_cm_id *, struct ibv_mr *);
void rdma_send_op(struct rdma_cm_id *, uint8_t);
void rdma_write_inline(struct rdma_cm_id *, void *, uint64_t, uint32_t);
#endif