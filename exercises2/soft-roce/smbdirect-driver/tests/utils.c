#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <rdma/rdma_cma.h>

#include "utils.h"
#include "message.h"

void fill_mem(char *buffer, uint size, char *msg, uint port)
{
	unsigned int msg_len = strlen(msg);
	unsigned int i = 0;
	unsigned int blks = size >> 8;

	for (i = 0; i < blks; i++) {
		snprintf(buffer + (i * 256), 256, "Blk: %04u, %s\n", i, msg);
	}
}

void build_negotiate_resp(struct connection_struct *connection)
{
	struct smbd_negotiate_resp *resp;

	resp = (struct smbd_negotiate_resp *)connection->send_buffer;

	connection->send_length = sizeof(struct smbd_negotiate_resp);
	resp->min_version = 0x0100;
	resp->max_version = 0x0100;
	resp->negotiated_version = 0x0100;
	resp->reserved = 0;
	resp->credits_requested = 255;
	resp->credits_granted = 255;  /* Should be based on requested */
	resp->status = 0;
	resp->max_read_write_size = 1024 * 1024;
}

void build_negotiate_req(struct connection_struct *connection)
{
	struct smbd_negotiate_req *neg;

	neg = (struct smbd_negotiate_req *)connection->send_buffer;

	connection->send_length = sizeof(struct smbd_negotiate_req);
	neg->min_version = 0x0100;
	neg->max_version = 0x0100;
	neg->reserved = 0;
	neg->credits_requested = 255;
	neg->preferred_send_size = 1364;
	neg->max_receive_size = 8192;
	neg->max_fragmented_size = 1024 * 1024;
}

/*
 * Send a message by posting the message that is in the send_buffer and
 * start the polling for completion events ...
 *
 * It is fine to have the work request and sge on the stack here.
 */
int send_message(struct connection_struct *connection)
{
	struct ibv_send_wr work_req, *bad_work_req = NULL;
	struct ibv_sge sge;

	fprintf(stderr, "%s: called ...\n", __func__);

	memset(&work_req, 0, sizeof(work_req));

	work_req.wr_id = (uintptr_t)connection;
	work_req.opcode = IBV_WR_SEND;
	work_req.sg_list = &sge;
	work_req.num_sge = 1;
	work_req.send_flags = IBV_SEND_SIGNALED;

	sge.addr = (uintptr_t)connection->send_buffer;
	sge.length = connection->send_length;
	sge.lkey = connection->send_mr->lkey;

	if (ibv_post_send(connection->id->qp, &work_req, &bad_work_req)) {
		fprintf(stderr, "%s: Unable to post message via RDMA: %s\n",
			__func__,
			strerror(errno));
		return 1;
	}

	return 0;
}

/*
 * Post a receive against the recv buffer in the connection struct.
 */
int post_receive(struct connection_struct *connection)
{
	struct ibv_recv_wr work_req, *bad_work_req = NULL;
	struct ibv_sge sge;

	memset(&work_req, 0, sizeof(work_req));
	work_req.wr_id = (uintptr_t)connection; /* Context */
	work_req.next = NULL;
	work_req.sg_list = &sge;
	work_req.num_sge = 1;

	sge.addr = (uintptr_t)connection->recv_buffer;
	sge.length = RECV_BUFFER_SIZE;
	sge.lkey = connection->recv_mr->lkey;

	if (ibv_post_recv(connection->id->qp, &work_req, &bad_work_req)) {
		fprintf(stderr, "%s: Unable to post receive: %s\n",
			__func__,
			strerror(errno));
		return 1;
	}

	return 0;
}

/*
 * Post an RDMA Read operation ...
 */
int post_rdma_req(struct connection_struct *connection, 
		  struct ibv_mr *mr,
		  int read,
		  size_t buffer_size)
{
	int res = 0;
	struct ibv_send_wr work_req, *bad_work_req = NULL;
	struct ibv_sge sge;

	memset(&work_req, 0, sizeof(work_req));
	work_req.wr_id = (uintptr_t)connection;
	work_req.opcode = read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
	work_req.sg_list = &sge;
	work_req.num_sge = 1;
	work_req.send_flags = IBV_SEND_SIGNALED;
	work_req.wr.rdma.remote_addr = mr->addr;
	work_req.wr.rdma.rkey = mr->rkey;
	sge.addr = (uintptr_t)connection->rdma_buffer;
	sge.length = buffer_size;
	sge.lkey = connection->rdma_buffer_mr->lkey;

	/* Now, post it ... */
	if ((res = ibv_post_send(connection->id->qp, 
				  &work_req,
				  &bad_work_req))) {
		fprintf(stderr, "Error posting RDMA op: %s\n",
			strerror(res));
		return 1;
	} 

	return 0;
}
