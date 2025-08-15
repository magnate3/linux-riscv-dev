/*
 * A simple SMB Direct client ... 
 *
 */

#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <rdma/rdma_cma.h>

#include "event_loop.h"
#include "message.h"
#include "utils.h"

#define RESOLVE_TIMEOUT 500 /* in mS */

#define RDMA_BUFFER_SIZE 65536

/*
 * We need a context that allows the event handlers to know what the main line
 * wanted .../
 */
struct command_context {
	struct rdma_event_channel *chan;
	int rdma_write;
	int local_op;
};

/*
 * Here we are handling the completion of a Verbs event.
 * We expect to get a response to our send ... 
 *
 * First up is negotiation, and then data transfer ...
 *
 */
int handle_completion(struct connection_struct *connection, 
		      struct ibv_wc *work_completion)
{
	struct command_context *cmd = connection->private;
	int res = 0;

	fprintf(stderr, "Handle completion called:\n");

	if (work_completion->status != IBV_WC_SUCCESS) {
		fprintf(stderr, "Bad completion status: %s\n",
			ibv_wc_status_str(work_completion->status));
		exit(199); /* Should be a cleaner way to clean up! */
	}

	/* Is it a receive? */
	if (work_completion->opcode & IBV_WC_RECV) {
		struct ibv_send_wr work_req, *bad_work_req = NULL;
		struct ibv_sge sge;
		struct smbd_negotiate_resp *resp;

		resp = (struct smbd_negotiate_resp *)connection->recv_buffer;

		printf("Received Negotiate Resp\n\tMin version: 0x%04x,"
			"Max version: 0x%04x, Negotiated Version: 0x%04x, "
			"Reserved: 0x%04x\n\tCredits requested: %d "
			"Credits granted: %d, Status: %d, "
			"Max Read/Write Size: %d\n",
			resp->min_version, resp->max_version,
			resp->negotiated_version, resp->reserved,
			resp->credits_requested, resp->credits_granted,
			resp->status, resp->max_read_write_size);

	}
	else if (work_completion->opcode == IBV_WC_SEND) {
		struct rdma_cm_id *connect_id = connection->id;
		struct ibv_recv_wr work_req, *bad_work_req = NULL;
		struct ibv_sge sge;

		printf("SEND completed\n");

		/*
		 * Post a receive for the response that should come back
		 */
		if (post_receive(connection)) {
			return 1; /* Message already sent ... */
		}

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
		if (handle_completion(connection, &work_completion))
			return 1;
	}

	/* Did an error occur? */
	if (res < 0) {

	}

	return 0;
}

/*
 * Handle an address resolved event ...
 */
int handle_address_resolved_event(struct rdma_cm_event *cm_event,
				  struct command_context *cmd)
{
	struct rdma_cm_id *connect_id = cm_event->id;
	struct ibv_qp_init_attr qp_attrs;
	struct connection_struct *connection = NULL;
	struct ibv_recv_wr work_req, *bad_work_req = NULL;
	struct ibv_sge wr_sge;

	/* 
	 * We have everything we need from the event now, so ack it. After we
	 * ack it, it is gone, so we can't access it again.
	 */
	rdma_ack_cm_event(cm_event);

	fprintf(stderr, "%s: Received address resolved event\n", __func__);
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
	connection->private = cmd;  /* Used in the completion handling code */

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

/*	connection->rdma_buffer_mr = ibv_reg_mr(connection->pd,
						connection->rdma_buffer,
						RDMA_BUFFER_SIZE,
						IBV_ACCESS_LOCAL_WRITE | 
						  IBV_ACCESS_REMOTE_READ |
						  IBV_ACCESS_REMOTE_WRITE);
	if (!connection->rdma_buffer_mr) {
		fprintf(stderr, "Unable to register RDMA buffer: %s\n",
			strerror(errno));
		goto error;
	}*/

	/*
	 * Post a recv before we accept the connection. That way, the client
	 * will not have to wait if it starts sending messages as soon as the
	 * connection is done. It is OK that these things are on the stack
	 * because they are copied.
	 */
	if (post_receive(connection)) {
		goto error; /* Message already sent ... */
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

	/*
	 * Now, resolve the route ...
	 */
	if (rdma_resolve_route(connect_id, RESOLVE_TIMEOUT)) {
		fprintf(stderr, "Unable to resolve route: %s\n",
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

int handle_route_resolved_event(struct rdma_cm_event *cm_event)
{
	struct rdma_cm_id *connect_id = cm_event->id;
	struct rdma_conn_param cm_params;  /* Note, on stack */

	fprintf(stderr, "%s called\n", __func__);

	rdma_ack_cm_event(cm_event);

	/* Now, accept it ... */
	memset(&cm_params, 0, sizeof(cm_params));
	cm_params.initiator_depth = cm_params.responder_resources = 1;
	cm_params.rnr_retry_count = 7; /* Keep retrying */
	if (rdma_connect(connect_id, &cm_params)) {
		fprintf(stderr, "Unable to connect to server: %s\n",
			strerror(errno));
		goto error;
	}

error:
	return 0;
}

int handle_connected_event(struct rdma_cm_event *cm_event)
{
	struct rdma_cm_id *connect_id = cm_event->id;
	struct connection_struct *connection = connect_id->context;
	struct message *msg = (struct message *)connection->send_buffer;
	struct command_context *cmd = connection->private; 

	rdma_ack_cm_event(cm_event);

	connection->connected = 1;

	fprintf(stderr, "We are now connected on remote port %u\n",
		rdma_get_dst_port(connect_id));

	/*
	 * Send the Negotiate request. The rest will be
	 * handled as completions ...
	 */
	build_negotiate_req(connection);

	if (send_message(connection)) {
		fprintf(stderr, "%s: Unable to send message to server.\n");
		return 1;
	}

	return 0;
}

int handle_disconnect_event(struct rdma_cm_event *cm_event)
{
	struct rdma_cm_id *connect_id = cm_event->id;

	rdma_ack_cm_event(cm_event);

	fprintf(stderr, "Received a disconnect event on remote port %u\n",
		rdma_get_dst_port(connect_id));

	rdma_disconnect(connect_id);

	if (rdma_destroy_id(connect_id)) {
		fprintf(stderr, "Problem destroying connect_id: %s\n",
			strerror(errno));
		return 1;
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
 * We have to step through the states required for the client side.
 */
uint32_t handle_cm_event_client(int32_t fd, uint32_t events, void *ctx)
{
	struct command_context *cmd = ctx;
	struct rdma_event_channel *ev_chan = cmd->chan;
	struct rdma_cm_event *cm_event = NULL;
	uint32_t res = 0;

	if (rdma_get_cm_event(ev_chan, &cm_event)) {
		fprintf(stderr, "%s: Unable to get CM event: %s\n", __func__,
			strerror(errno));
		return 1;  /* Bad, drop out of the event loop */
	}

	/*
	 * Now, handle the event ... we should only get address resolved and
	 * connected etc.
	 */
	switch (cm_event->event) {
	case RDMA_CM_EVENT_ADDR_RESOLVED: /* Address resolved */
		res = handle_address_resolved_event(cm_event, cmd);
		break;

	case RDMA_CM_EVENT_ROUTE_RESOLVED:
		res = handle_route_resolved_event(cm_event);
		break;

	case RDMA_CM_EVENT_ESTABLISHED: /* Everything is up */
		res = handle_connected_event(cm_event);
		break;

	case RDMA_CM_EVENT_DISCONNECTED: /* Connection gone */
		res = handle_disconnect_event(cm_event);
		if (res) {

		}
		res = 1;
		break;

	case RDMA_CM_EVENT_REJECTED: /* Something is wrong */
		fprintf(stderr, "\nRDMA_CM_EVENT_REJECTED event received\n");
		fprintf(stderr, "Did you give me the correct info?\n");
		res = 1;
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

	return res;
}

void usage(char *program)
{
	fprintf(stderr, "Usage: %s host port\n", program);
}

/*
 * Now a main routine that uses IB etc ... we are the server here ... so we
 * listen for a connection. We are using the connection oriented port space,
 * RDMA_PS_TCP, not the UDP space (for unicast and multicast).
 */
int main(int argc, char *argv[])
{
	struct rdma_cm_event *cm_event = NULL;
	struct rdma_cm_id *cm_connect_id = NULL;
	struct rdma_event_channel *rdma_ev_chan = NULL;
	uint16_t connect_port = 0;
	struct addrinfo *remote_addr = NULL;
	int rdma_write = 0; /* the op to do is an RDMA READ by default */
	int local_op = 0;  /* the op is local, ie, we do it, default to rem */
	int opt;
	int res = 0;
	struct command_context cmd;

	/*
	 * Parse some flags first ...
	 */ 
	while ((opt = getopt(argc, argv, "wc")) != -1) {
		switch (opt) {
		case 'w':
			rdma_write = 1;
			printf("We are going to ask for an RDMA WRITE\n"); 
			break;

		case 'c':
			local_op = 1;
			printf("We will tell the client we will do it\n");
			break;

		case '?':
			usage(argv[0]);
			return 255;
			break;
		}
	}

	/* There should be three args plus options argv[0], host and port */
	/* Have to account for difference between position and count      */
	if (((optind + 3 - 1) > argc) || (argc < 3)) {
		printf("optind = %u, argc = %u\n", optind, argc);
		usage(argv[0]);
		return 255;
	}

	/* Grab the address info */
	if (res = getaddrinfo(argv[optind], 
			      argv[optind+1], 
			      NULL, 
			      &remote_addr)) {
		fprintf(stderr, "Unable to get address info: %s\n",
			gai_strerror(res));
		return 255;
	}

	/* Get an RDMA event channel first */
	rdma_ev_chan = rdma_create_event_channel();
	if (!rdma_ev_chan) {
		fprintf(stderr, "Unable to create event channel: %s\n",
			strerror(errno));
		return 1;
	}

	/* Create an RDMA Comms ID ..., no context needed yet */
	if (rdma_create_id(rdma_ev_chan, &cm_connect_id, NULL, RDMA_PS_TCP)) {
		fprintf(stderr, "Unable to create RDMA Comms ID: %s\n",
			strerror(errno));
		return 2;
	}

	if (rdma_resolve_addr(cm_connect_id, 
			      NULL,
			      remote_addr->ai_addr,
			      RESOLVE_TIMEOUT)) {
		fprintf(stderr, "Unable to bind to an RDMA address: %s\n",
			strerror(errno));
		return 3;
	}

	connect_port = ntohs(rdma_get_src_port(cm_connect_id));

	fprintf(stderr, "Connecting on port %u\n", connect_port);

	freeaddrinfo(remote_addr);

	/*
	 * Now, register the event stuff and then poll for it. It does not
	 * matter that we register with our polling loop here, I think.
	 */
	cmd.chan = rdma_ev_chan;
	cmd.rdma_write = rdma_write;
	cmd.local_op = local_op;

	if (poll_add_event_source(rdma_ev_chan->fd,
				  EPOLLIN,
				  &cmd,
				  handle_cm_event_client)) {

	}

	/*
	 * Now dispatch events ... Hmmm, no real need to test
	 */
	if (event_loop_and_dispatch()) {
		fprintf(stderr, "Exiting!\n");
		return 99;
	}

}
