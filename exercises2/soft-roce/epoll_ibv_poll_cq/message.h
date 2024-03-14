#ifndef __MESSAGE__
#define __MESSAGE__

/*
 * These messages allow client to tell server:
 * Client RDMA Read
 * Client RDMA Write
 * Server RDMA Read
 * Server RDMA Write
 * Done ...
 *
 * The protocol is:
 * after Server ops it sends done, and after client ops, it sends done.
 */
struct message {
	enum {
		MSG_SR,    /* RDMA read by server from client */
		MSG_CR,    /* RDMA read by the client from the server */
		MSG_SW,    /* RDMA write by the server to client */
		MSG_CW,    /* RDMA write by the client to the server */
		MSG_OK,    /* We have done what you asked */
		MSG_DONE,
	} type;

	union {
		struct ibv_mr mr;
	} data;
};
#endif
