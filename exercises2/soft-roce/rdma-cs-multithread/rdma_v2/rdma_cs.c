/**
 * @file rdma_cs.c
 * @author Austin Pohlmann
 * @brief File containing the definitions of the functions listed in rdma_cs.h
 *
 */
#include "rdma_cs.h"

 /**
  * @brief Print an error message and exit the application
  *
  * @return @c NULL
  * @param reason a string representation of what the error came from
  * @param error the error number, typically just errno
  * @param file the file to output the error message to
  */
void stop_it(char *reason, int error, FILE *file){
	fprintf(file, "Error for %s: '%s'\n", reason, strerror(error));
	exit(-1);
}

/**
 * @brief Process a communication manager event
 *
 * The function will call exit(-1) if the event found does not match the expected event
 * @return the @c struct @c rdma_cm_id of the new connection if the event was @c RDMA_CM_EVENT_CONNECT_REQUEST
 * @param ec the event channel to check
 * @param expected the expected event
 * @param file the file to output the connection info of a new connection if the event was @c RDMA_CM_EVENT_CONNECT_REQUEST
 */
struct rdma_cm_id *cm_event(struct rdma_event_channel *ec,
 enum rdma_cm_event_type expected, FILE *file){
	struct rdma_cm_event *event;
	struct rdma_cm_id *id;
	if(rdma_get_cm_event(ec, &event))
		stop_it("rdma_get_cm_event()", errno, file);
	if(event->event != expected){
		fprintf(file, "Error: expected \"%s\" but got \"%s\"\n", 
			rdma_event_str(expected), rdma_event_str(event->event));
		exit(-1);
	}
	if(event->event == RDMA_CM_EVENT_CONNECT_REQUEST){
		id=event->id;
		struct ibv_qp_init_attr init_attr;
		memset(&init_attr, 0, sizeof(init_attr));
		init_attr.qp_type = IBV_QPT_RC;
		init_attr.cap.max_send_wr  = MAX_SEND_WR;
		init_attr.cap.max_recv_wr  = MAX_RECV_WR;
		init_attr.cap.max_send_sge = MAX_SEND_SGE;
		init_attr.cap.max_recv_sge = MAX_RECV_SGE;
		init_attr.cap.max_inline_data = MAX_INLINE_DATA;
		if(rdma_create_qp(id, NULL, &init_attr))
			stop_it("rdma_create_qp()", errno, file);
		struct rdma_conn_param *conn_params = malloc(sizeof(*conn_params));
		memset(conn_params, 0, sizeof(*conn_params));
		conn_params->retry_count = 8;
		conn_params->rnr_retry_count = 8;
		conn_params->responder_resources = 10;
		conn_params->initiator_depth = 10;
		if(rdma_accept(id, conn_params))
			stop_it("rdma_accept()", errno, file);
		fprintf(file, "Accepted connection request from remote QP 0x%x.\n",
			(unsigned int)event->param.conn.qp_num);
	}
	if(rdma_ack_cm_event(event))
		stop_it("rdma_ack_cm_event()", errno, file);
	return id;
}

/**
 * @brief Exchange the information needed to perform rdma read/write operations
 *
 * The address, rkey, and size of a memory region is sent to and recieved from a remote host.
 * WARNING: this erases the contents of the memory region used
 * @return @c NULL
 * @param cm_id the id associated with the connection to the remote host
 * @param mr the memory region to send information about
 * @param rkey the location to store the remote host's memory region's rkey
 * @param remote_addr the location to store the remote host's memory region's address
 * @param size the location to store the remote host's memory region's size
 * @param file the file to print the sent and received information to
 */
void swap_info(struct rdma_cm_id *cm_id, struct ibv_mr *mr, uint32_t *rkey, uint64_t *remote_addr,
	size_t *size, FILE *file){
	if(rdma_post_recv(cm_id, "qwerty", mr->addr, 30, mr))
		stop_it("rdma_post_recv()", errno, file);
	memcpy(mr->addr+30,&mr->addr,sizeof(mr->addr));
	memcpy(mr->addr+30+sizeof(mr->addr),&mr->rkey,sizeof(mr->rkey));
	memcpy(mr->addr+30+sizeof(mr->addr)+sizeof(mr->rkey),&mr->length,sizeof(mr->length));
	if(rdma_post_send(cm_id, "qwerty", mr->addr+30, 30, mr, IBV_SEND_SIGNALED))
		stop_it("rdma_post_send()", errno, file);
	get_completion(cm_id, SEND, 1, file);
	fprintf(file, "Sent local address: 0x%0llx\nSent local rkey: 0x%0x\n", (unsigned long long)mr->addr, (unsigned int)mr->rkey);
	get_completion(cm_id, RECV, 1, file);
	memcpy(remote_addr, mr->addr, sizeof(*remote_addr));
	memcpy(rkey, mr->addr+sizeof(*remote_addr), sizeof(*rkey));
	fprintf(file, "Received remote address: 0x%0llx\nReceived remote rkey: 0x%0x\n", (unsigned long long)*remote_addr, (unsigned int)*rkey);
	if(size != NULL){
		memcpy(size, mr->addr+sizeof(*remote_addr)+sizeof(*rkey), sizeof(*size));
		fprintf(file, "Received remote memory region length: %u bytes\n", (unsigned int)*size);
	}
	memset(mr->addr, 0, mr->length);
}

/**
 * @brief Wait for and pull a work completion
 *
 * This function will block until a work completion of the specified type is pulled from the completion queue.
 * If the completion contains immediate data, it will be returned.
 * Operations included in SEND: send (read and write if IBV_SEND_SIGNALED is set)
 * Operations included in RECV: receive
 * @return the immediate data received, if present
 * @param cm_id the id associated with the connection to the remote host
 * @param type the type of completion to pull, either SEND of RECV
 * @param print 1 if this should print anything, 0 if not
 * @param file the file to print to
 */
uint32_t get_completion(struct rdma_cm_id *cm_id, enum completion_type type, uint8_t print, FILE *file){
	struct ibv_wc wc;
	uint32_t data = 0;
	switch(type){
		case SEND:
			if(-1 == rdma_get_send_comp(cm_id, &wc))
				stop_it("rdma_get_send_comp()", errno, file);
			break;
		case RECV:
			if(-1 == rdma_get_recv_comp(cm_id, &wc))
				stop_it("rdma_get_recv_comp()", errno, file);
			break;
		default:
			fprintf(file, "Error: unknown completion type\n");
			exit(-1);
	}
	if(print){
		switch(wc.opcode){
			case IBV_WC_SEND:
				fprintf(file, "Send ");
				break;
			case IBV_WC_RECV:
				fprintf(file, "Receive ");
				break;
			case IBV_WC_RECV_RDMA_WITH_IMM:
				fprintf(file, "Receive with immediate data ");
				break;
			case IBV_WC_RDMA_WRITE:
				fprintf(file, "Write ");
				break;
			case IBV_WC_RDMA_READ:
				fprintf(file, "Read ");
				break;
			default:
				fprintf(file, "Operation %d", wc.opcode);
				break;
		}
	}
	if(!wc.status){
		if(print)
			fprintf(file, "completed successfully!\n");
		if(wc.wc_flags & IBV_WC_WITH_IMM && wc.opcode != IBV_WC_SEND){
			if(print)
				fprintf(file, "Immediate data: 0x%x\n", ntohl(wc.imm_data));
			data = ntohl(wc.imm_data);
		}
		if(wc.opcode == IBV_WC_RECV && print){
			fprintf(file, "%u bytes received.\n", wc.byte_len);
		}
	} else if(print){
		fprintf(file, "failed with error value %d.\n", wc.status);
	}
	return data;
}

/**
 * @brief Disconnect from a remote host and free the resources used
 *
 * @param mine the id of the local host
 * @param client the id of the remote host
 * @param mr the memory region to free
 * @param ec the event channel to use
 * @param file the file to print to
 */
int obliterate(struct rdma_cm_id *mine,struct rdma_cm_id *client, struct ibv_mr *mr,
	struct rdma_event_channel *ec, FILE *file){
	fprintf(file, "Disconnecting...\n");
	struct rdma_cm_id *id = client != NULL ? client : mine;
	if(rdma_dereg_mr(mr))
		stop_it("rdma_dereg_mr()", errno, file);
	if(client != NULL)
		cm_event(ec, RDMA_CM_EVENT_DISCONNECTED, file);
	if(rdma_disconnect(id))
		stop_it("rdma_disconnect()", errno, file);
	if(client == NULL)
		cm_event(ec, RDMA_CM_EVENT_DISCONNECTED, file);
	rdma_destroy_qp(id);
	if(client != NULL){
		if(rdma_destroy_id(client))
			stop_it("rdma_destroy_id()", errno, file);
	}
	if(mine != NULL){
		if(rdma_destroy_id(mine))
			stop_it("rdma_destroy_id()", errno, file);
	}
	rdma_destroy_event_channel(ec);
	return 0;
}

/**
 * @brief A simple wrapper for rdma_post_recv()
 *
 * @param id the id associated with the connection to the remote host
 * @param mr the memory region to receive the data in
 * @param file the file to print to in the event of an error
 */
void rdma_recv(struct rdma_cm_id *id, struct ibv_mr *mr, FILE *file){
	if(rdma_post_recv(id, "qwerty", mr->addr, mr->length, mr))
		stop_it("rdma_post_recv()", errno, file);
}

/**
 * @brief A 0 byte send with immediate data
 *
 * This is used to send opcodes between hosts using the immediate data.
 * @return @c NULL
 * @param id the id associated with the connection to the remote host
 * @param op the opcode (immediate data) to send
 * @param file the file to print to in the event of an error
 */
void rdma_send_op(struct rdma_cm_id *id, uint8_t op, FILE *file){
	struct ibv_send_wr wr, *bad;
	wr.next = NULL;
	wr.sg_list = NULL;
	wr.num_sge = 0;
	wr.opcode = IBV_WR_SEND_WITH_IMM;
	wr.send_flags = IBV_SEND_SIGNALED;
	wr.imm_data = htonl(op);
	if(rdma_seterrno(ibv_post_send(id->qp, &wr, &bad)))
		stop_it("ibv_post_send()", errno, file);
}

/**
 * @brief A simple wrapper for an inline write using rdma_post_write().
 *
 * @return @c NULL
 * @param id the id associated with the connection to the remote host
 * @param buffer the buffer containing the data to be written
 * @param address the remote address to write to
 * @param key the key associated with the remote address
 * @param file the file to print to in the event of an error
 */
void rdma_write_inline(struct rdma_cm_id *id, void *buffer, uint64_t address, uint32_t key, FILE *file){
	if(rdma_post_write(id, "qwerty", buffer, strlen(buffer), NULL,
		IBV_SEND_INLINE | IBV_SEND_SIGNALED, address, key))
		stop_it("rdma_post_write()", errno, file);
}
