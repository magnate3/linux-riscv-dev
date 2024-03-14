#include "rdma_cs.h"



int connect_four(struct rdma_cm_id *, struct rdma_event_channel *, char *, short int);

int main(int argc, char **argv){
	// Get server address and port from arguments
	if(argc != 3){
		printf("Invalid arguements: %s <address> <port>\n", argv[0]);
		return -1;
	}
	char *ip = argv[1];
	short port = atoi(argv[2]);
	// Create the event channel
	struct rdma_event_channel *event_channel = rdma_create_event_channel();
	if(event_channel == NULL)
		stop_it("rdma_create_event_channel()", errno);
	// Create the ID
	struct rdma_cm_id *cm_id;
	if(rdma_create_id(event_channel, &cm_id, "qwerty", RDMA_PS_TCP))
		stop_it("rdma_create_id()", errno);
	// Connect to the server
	connect_four(cm_id, event_channel, ip, port);
	// Register memory region
	struct ibv_mr *mr = ibv_reg_mr(cm_id->qp->pd, malloc(REGION_LENGTH), REGION_LENGTH,
	 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
	if(mr == NULL)
		stop_it("ibv_reg_mr()", errno);
	// Exchange addresses and rkeys with the server
	uint32_t rkey;
	uint64_t remote_addr;
	size_t server_mr_length;
	swap_info(cm_id, mr, &rkey, &remote_addr, &server_mr_length);
	// The real good
	void *buffer = malloc(MAX_INLINE_DATA);
	uint8_t opcode;
	unsigned long long length;
	unsigned long long offset;
	int i;
	unsigned char *byte;
	while(1){
		printf("---------------\n"
			"| RDMA client |\n"
			"---------------\n"
			"1) Disconnect\n"
			"2) Write\n"
			"3) Read\n"
			"> ");
		opcode = fgetc(stdin)%48;
		while(fgetc(stdin) != '\n')
			fgetc(stdin);
		if(opcode == DISCONNECT){
			rdma_send_op(cm_id, opcode);
			get_completion(cm_id, SEND, 0);
			break;
		} else if(opcode == WRITE){
			printf("Server memory region is %u bytes long. "
				"Choosing a relative point (0 - %u) to start writing to.\n> ",
				(unsigned int)server_mr_length, (unsigned int)server_mr_length-MAX_INLINE_DATA);
			scanf("%llu", &offset);fgetc(stdin);
			if(offset+MAX_INLINE_DATA> server_mr_length){
				printf("Invalid offset.\n");
				continue;
			}
			printf("Enter the data to be sent to the server. Max length is %u bytes.\n",
				MAX_INLINE_DATA);
			memset(buffer, 0, MAX_INLINE_DATA);
			if(fgets(buffer, MAX_INLINE_DATA, stdin) == NULL){
				printf("Unknow error occured.");
				rdma_send_op(cm_id, DISCONNECT);
				get_completion(cm_id, SEND, 0);
				break;
			}
			rdma_write_inline(cm_id, buffer, remote_addr+offset, rkey);
			get_completion(cm_id, SEND, 1);
		} else if(opcode == READ){
			printf("Server memory region is %u bytes long. "
				"Choosing a relative point (0 - %u) to start reading from, followed by "
				"how many bytes you wish to read (0-%u).\n> ", (unsigned int)server_mr_length,
				(unsigned int)server_mr_length-1, REGION_LENGTH);
			scanf("%llu", &offset);fgetc(stdin);
			printf("> ");
			scanf("%llu", &length);fgetc(stdin);
			if(offset+length > server_mr_length || length > REGION_LENGTH){
				printf("Invalid offset and/or length.\n");
				continue;
			}
			if(rdma_post_read(cm_id, "qwerty", mr->addr, length, mr, IBV_SEND_SIGNALED,
				remote_addr + offset, rkey))
				stop_it("rdma_post_read()", errno);
			get_completion(cm_id, SEND, 1);
			// Print data in hex 1 byte at a time
			printf("Data: ");
			byte = (unsigned char *)mr->addr;
			for(i=0;i<length;i++)
				printf("%02x ", byte[i]);
			printf("\n");
		} else {
			printf("Unknown operation, try again buddy.\n");
		}

	}
	// Disconnect
	obliterate(cm_id, NULL, mr, event_channel);
	return 0;
}

int connect_four(struct rdma_cm_id *cm_id, struct rdma_event_channel *ec, char *ip, 
	short int port){
	struct sockaddr_in sin;
	memset(&sin, 0, sizeof(struct sockaddr_in));
	sin.sin_family = AF_INET;
	sin.sin_port = htons(port);
	inet_aton(ip, &(sin.sin_addr));
	// Resolve the server's address
	if(rdma_resolve_addr(cm_id, NULL, (struct sockaddr *)&sin, 10000))
		stop_it("rdma_resolve_addr()", errno);
	// Wait for the address to resolve
	cm_event(ec, RDMA_CM_EVENT_ADDR_RESOLVED);
	// Create queue pair
	struct ibv_qp_init_attr *init_attr = malloc(sizeof(*init_attr));
	memset(init_attr, 0, sizeof(*init_attr));
	init_attr->qp_type = IBV_QPT_RC;
	init_attr->cap.max_send_wr  = MAX_SEND_WR;
	init_attr->cap.max_recv_wr  = MAX_RECV_WR;
	init_attr->cap.max_send_sge = MAX_SEND_SGE;
	init_attr->cap.max_recv_sge = MAX_RECV_SGE;
	init_attr->cap.max_inline_data = MAX_INLINE_DATA;
	if(rdma_create_qp(cm_id, NULL, init_attr))
		stop_it("rdma_create_qp()", errno);
	// Resolve the route to the server
	if(rdma_resolve_route(cm_id, 10000))
		stop_it("rdma_resolve_route()", errno);
	// Wait for the route to resolve
	cm_event(ec, RDMA_CM_EVENT_ROUTE_RESOLVED);
	// Send a connection request to the server
	struct rdma_conn_param *conn_params = malloc(sizeof(*conn_params));
	printf("Conencting...\n");
	memset(conn_params, 0, sizeof(*conn_params));
	conn_params->retry_count = 8;
	conn_params->rnr_retry_count = 8;
	conn_params->responder_resources = 10;
	conn_params->initiator_depth = 10;
	if(rdma_connect(cm_id, conn_params))
		stop_it("rdma_connect()", errno);
	// Wait for the server to accept the connection
	cm_event(ec, RDMA_CM_EVENT_ESTABLISHED);
	return 0;
}