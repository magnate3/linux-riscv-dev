/*
 * A simple IB example ... 
 *
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>

#include <rdma/rdma_cma.h>

#include "event_loop.h"
#include "message.h"
#include "utils.h"

#define RDMA_BUFFER_SIZE (1024*1024)

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

/*
 * Here we are handling the completion of a Verbs event. We bundle them all
 * in together ... but could split them out to different completion types.
 */
int handle_completion(struct connection_struct *connection, 
		      struct ibv_wc *work_completion)
{
	struct message *msg = (struct message *)connection->recv_buffer;
	struct message *snd = (struct message *)connection->send_buffer;

	fprintf(stderr, "Handle completion called: %p\n", msg);

	if (work_completion->status != IBV_WC_SUCCESS) {
		fprintf(stderr, "Bad completion status: %s\n",
			ibv_wc_status_str(work_completion->status));
		exit(199); /* Should be a cleaner way to clean up! */
	}

	/* Is it a receive? */
	if (work_completion->opcode & IBV_WC_RECV) {
		struct ibv_send_wr work_req, *bad_work_req = NULL;
		struct ibv_sge sge;
		int res = 0;

		/*
		 * Now, process the message ... 
		 */
		if (connection->state != SMBD_CONNECTED) {
			struct smbd_negotiate_req *req = (struct smbd_negotiate_req *) connection->recv_buffer;

			printf("Negotiate req: Min Vers: 0x%04x, "
				"Max Vers: 0x%04x, Reserved: 0x%04x, \n"
				"\tCredits Requested: 0x%04x, "
				"Preferred Send Size: %d, "
				"Max receive_size = %d, "
				"Max fragmented size = %d\n",
				req->min_version,
				req->max_version,
				req->reserved,
				req->credits_requested,
				req->preferred_send_size,
				req->max_receive_size,
				req->max_fragmented_size);

			build_negotiate_resp(connection);
			if (send_message(connection)) {
				return 1;
			}
			connection->state = SMBD_CONNECTED;
		}
		else {
			/*
			 * Just turn it around and send it back ...
			 */

		}
	}
	else if (work_completion->opcode == IBV_WC_SEND) {
		printf("Send completed ...\n");

	}
	else if (work_completion->opcode == IBV_WC_RDMA_READ) {
		fprintf(stderr, "RDMA READ from client completed ...\n");
		fprintf(stderr, "===================================\n");
		fprintf(stderr, connection->rdma_buffer);

	}
	else if (work_completion->opcode == IBV_WC_RDMA_WRITE) {
		fprintf(stderr, "RDMA WRITE to client completed ...\n");
		fprintf(stderr, "==================================\n");

	}
	else {
		fprintf(stderr, "%s: Opcode is: %0X\n", __func__,
			work_completion->opcode);
	}

	return 0;
}

/*
 * This is for IBV completions ... might split them later into separate ones
 * for send, recv, rdma_write and rdma_read.
 */
uint32_t handle_completion_event(int32_t fd, uint32_t events, void *ctx)
{
	struct connection_struct *connection = ctx;
	struct ibv_cq *comp_queue = NULL;
	struct ibv_wc work_completion;
	void *ev_ctx = NULL;
	int res = 0;

	fprintf(stderr, "%s called\n", __func__);

	/*
	 * Retrieve the completion event and deal with it. We did not pass 
	 * a context when we created the completion queue, but we need a place
	 * for get_cq_events to place that NULL pointer anyway.
	 */
	if (ibv_get_cq_event(connection->comp_chan, &comp_queue, &ev_ctx)) {
		fprintf(stderr, "Could not get completion queue events: %s",
			strerror(errno));
		return 1;
	}

	ibv_ack_cq_events(comp_queue, 1);  /* Ack it */

	/* Re-arm ... we need the subsequent events, if any */
	if (ibv_req_notify_cq(comp_queue, 0)) {
		fprintf(stderr, "Could not request notifies on "
			"comp queue: %s\n",
			strerror(errno));
		return 1;
	}

	while ((res = ibv_poll_cq(comp_queue, 1, &work_completion)) > 0) {
		handle_completion(connection, &work_completion);
	}

	/* Did an error occur? */
	if (res < 0) {

	}

	return 0;
}

/*
 * Handle a connection request event ...
 */
int handle_connect_request_event(struct rdma_cm_event *cm_event)
{
	struct rdma_cm_id *connect_id = cm_event->id;
	struct rdma_conn_param cm_params;  /* Note, on stack */
	struct ibv_qp_init_attr qp_attrs;
	struct connection_struct *connection = NULL;
	struct ibv_recv_wr work_req, *bad_work_req = NULL;
	struct ibv_sge wr_sge;

	/* 
	 * We have everything we need from the event now, so ack it. After we
	 * ack it, it is gone, so we can't access it again.
	 */
	rdma_ack_cm_event(cm_event);

	fprintf(stderr, "Received connect request from remote port %u\n",
		rdma_get_dst_port(connect_id));
	/*
	 * Do important stuff here, like create QPs etc ...
	 */
	connection = malloc(sizeof(struct connection_struct));
	if (!connection) {
		fprintf(stderr, "Could not allocate space for connection: %s\n",
			strerror(errno));
		return 1;
	}

	memset(connection, 0, sizeof(connection));

	/*
	 * Stash this here so we can get it next time around
	 */
	connect_id->context = connection;
	connection->id = connect_id;

	connection->ctx = connect_id->verbs;
	connection->pd = ibv_alloc_pd(connect_id->verbs);
	if (!connection->pd) {
		fprintf(stderr, "Unable to allocate IB Protection Domain: %s\n",
			strerror(errno));
		goto error;
	}

	/* Create a completion channel for event notification */
	connection->comp_chan = ibv_create_comp_channel(connect_id->verbs);
	if (!connection->comp_chan) {
		fprintf(stderr, "Unable to create completion channel: %s\n",
			strerror(errno));
		goto error;
	}

	/* Create a completion queue ... */
	connection->cq = ibv_create_cq(connect_id->verbs, 
					5, 
					NULL,
					connection->comp_chan,
					0); 
	if (!connection->cq) {
		fprintf(stderr, "Unable to create completion queue: %s\n",
			strerror(errno));
		goto error;
	}

	/*
	 * Request notificiation before we set the queue pair up
	 */
	if (ibv_req_notify_cq(connection->cq, 0)) {
		fprintf(stderr, "Unable to request notify: %s\n",
			strerror(errno));
		goto error;
	}

	/*
	 * Now, set up the queue pair attributes. We point both the send and
	 * recv completion queues to the one completion queue we just created.
	 * If we need separate completion routines, then use separate
	 * completion queues. Queue type is Reliable Connection oriented queue.
	 */
	memset(&qp_attrs, 0, sizeof(qp_attrs)); /* zero it so no surprises */
	qp_attrs.send_cq = connection->cq;
	qp_attrs.recv_cq = connection->cq;
	qp_attrs.qp_type = IBV_QPT_RC;
	qp_attrs.cap.max_send_wr = 10;
	qp_attrs.cap.max_recv_wr = 10;
	qp_attrs.cap.max_send_sge = 1;
	qp_attrs.cap.max_recv_sge = 1;

	if (rdma_create_qp(connect_id, connection->pd, &qp_attrs)) {
		fprintf(stderr, "Unable to create queue pair: %s\n",
			strerror(errno));
		goto error;
	}

	/*
	 * Register memory areas ... we will probably want to send a message
	 * in response ... and an area to copy to.
	 */
	connection->recv_mr = ibv_reg_mr(connection->pd,
					 connection->recv_buffer,
					 RECV_BUFFER_SIZE,
					 IBV_ACCESS_LOCAL_WRITE | 
						IBV_ACCESS_REMOTE_WRITE);
	if (!connection->recv_mr) {
		fprintf(stderr, "Unable to register recv memory region: %s\n",
			strerror(errno));
		goto error;
	}

	connection->send_mr = ibv_reg_mr(connection->pd,
					 connection->send_buffer,
					 SEND_BUFFER_SIZE,
					 IBV_ACCESS_LOCAL_WRITE |
						IBV_ACCESS_REMOTE_WRITE);
	if (!connection->send_mr) {
		fprintf(stderr, "Unable to register send memory region: %s\n",
			strerror(errno));
		goto error;
	}

	connection->rdma_buffer = malloc(RDMA_BUFFER_SIZE);
	if (!connection->rdma_buffer) {
		fprintf(stderr, "Unable to allocate RDMA buffer: %s\n",
			strerror(errno));
		goto error;
	}

	connection->rdma_buffer_mr = ibv_reg_mr(connection->pd,
						connection->rdma_buffer,
						RDMA_BUFFER_SIZE,
						IBV_ACCESS_LOCAL_WRITE | 
						IBV_ACCESS_REMOTE_WRITE);
	if (!connection->rdma_buffer_mr) {
		fprintf(stderr, "Unable to register RDMA buffer: %s\n",
			strerror(errno));
		goto error;
	}

	/*
	 * Post receives before we accept the connection. That way, the client
	 * will not have to wait if it starts sending messages as soon as the
	 * connection is done. It is OK that these things are on the stack
	 * because they are copied.
	 */
	if (post_receive(connection)) {
		fprintf(stderr, "Unable to post receive: %s\n",
			strerror(errno));
		goto error;
	}

	/*
	 * Now, set up the polling as well
	 */
	if (poll_add_event_source(connection->comp_chan->fd,
				  EPOLLIN,
				  connection,
				  handle_completion_event)) {
		fprintf(stderr, "Unable to add completion event: %s\n",
			strerror(errno));
		goto error;
	}

	/* Now, accept it ... */
	memset(&cm_params, 0, sizeof(cm_params));
	cm_params.initiator_depth = cm_params.responder_resources = 1;
	cm_params.rnr_retry_count = 7; /* Keep retrying */
	if (rdma_accept(connect_id, &cm_params)) {
		fprintf(stderr, "Unable to accept RDMA connection: %s\n",
			strerror(errno));
		goto error;
	}

	return 0;

error:
	/* Clean up ... do we need to destroy the connect_id? */
	free(connection->rdma_buffer);
	free(connection);
	return 1;
}

int handle_connected_event(struct rdma_cm_event *cm_event)
{
	struct rdma_cm_id *connect_id = cm_event->id;
	struct connection *connection = connect_id->context;

	rdma_ack_cm_event(cm_event);

	fprintf(stderr, "We are now connected on remote port %u\n",
		rdma_get_dst_port(connect_id));

	return 0;
}

int handle_disconnect_event(struct rdma_cm_event *cm_event)
{
	struct rdma_cm_id *connect_id = cm_event->id;

	rdma_ack_cm_event(cm_event);

	fprintf(stderr, "Received a disconnect event on remote port %u\n",
		rdma_get_dst_port(connect_id));

	if (rdma_destroy_id(connect_id)) {
		fprintf(stderr, "Problem destroying connect_id: %s\n",
			strerror(errno));
	}

	return 0;
}

/*
 * Handle a CM event. It is assumed that we are passed the channel in ctx
 * We leave acking of the event up to the actual handling routine ... except
 * for the default case ... this avoids the awkward ness of copying the event
 * and acking it before processing it.
 *
 * We need to do it this way because handle_disconnect_event is going to
 * destroy the connect_id ... so it must ack the event first.
 *
 * We step through the states for the server side.
 */
uint32_t handle_cm_event_server(int32_t fd, uint32_t events, void *ctx)
{
	struct rdma_event_channel *ev_chan = ctx;
	struct rdma_cm_event *cm_event = NULL;

	if (rdma_get_cm_event(ev_chan, &cm_event)) {
		fprintf(stderr, "%s: Unable to get CM event: %s\n", __func__,
			strerror(errno));
		return 1;  /* Bad, drop out of the event loop */
	}

	/*
	 * Now, handle the event ... 
	 */
	switch (cm_event->event) {
	case RDMA_CM_EVENT_CONNECT_REQUEST: /* someone wants to connect */
		/*
		 * Don't do anything bad on errors, yet
		 */
		handle_connect_request_event(cm_event);
		break;

	case RDMA_CM_EVENT_ESTABLISHED: /* Everything is up */
		handle_connected_event(cm_event);
		break;

	case RDMA_CM_EVENT_DISCONNECTED: /* Connection gone */
		handle_disconnect_event(cm_event);
		break;

	default:
		fprintf(stderr, "Unknown event \"%s\", ignored\n",
			rdma_event_str(cm_event->event));
		/*
		 * We must ack this one here ...
		 */
		rdma_ack_cm_event(cm_event);
		break;
	}

	return 0;
}

/*
 * Now a main routine that uses IB etc ... we are the server here ... so we
 * listen for a connection. We are using the connection oriented port space,
 * RDMA_PS_TCP, not the UDP space (for unicast and multicast).
 */
int main(int argc, char *argv[])
{
	struct rdma_cm_event *cm_event = NULL;
	struct rdma_cm_id *cm_listen_id = NULL;
	struct rdma_event_channel *rdma_ev_chan = NULL;
	struct sockaddr_in l_addr;
	uint16_t listen_port = 0;

	memset(&l_addr, 0, sizeof(l_addr));
	l_addr.sin_family = AF_INET;

	/* Get an RDMA event channel first */
	rdma_ev_chan = rdma_create_event_channel();
	if (!rdma_ev_chan) {
		fprintf(stderr, "Unable to create event channel: %s\n",
			strerror(errno));
		return 1;
	}

	/* Create an RDMA Comms ID ..., no context needed yet */
	if (rdma_create_id(rdma_ev_chan, &cm_listen_id, NULL, RDMA_PS_TCP)) {
		fprintf(stderr, "Unable to create RDMA Comms ID: %s\n",
			strerror(errno));
		return 2;
	}

	if (rdma_bind_addr(cm_listen_id, (struct sockaddr *)&l_addr)) {
		fprintf(stderr, "Unable to bind to an RDMA address: %s\n",
			strerror(errno));
		return 3;
	}

	listen_port = ntohs(rdma_get_src_port(cm_listen_id));

	fprintf(stderr, "Listening on port %u\n", listen_port);

	/* We only allow one outstanding connection request */
	if (rdma_listen(cm_listen_id, 1)) {
		fprintf(stderr, "Unable to listen on port %u (%s)\n",
			listen_port, 
			strerror(errno));
		return 4;
	}

	/*
	 * Now, register the event stuff and then poll for it. It does not
	 * matter that we register with our polling loop here, I think.
	 */
	if (poll_add_event_source(rdma_ev_chan->fd,
				  EPOLLIN,
				  rdma_ev_chan,
				  handle_cm_event_server)) {

	}

	/*
	 * Now dispatch events ...
	 */
	for (;;) {
		if (event_loop_and_dispatch()) {
			fprintf(stderr, "Exiting!\n");
			return 99;
		}
	}

}
