#include "rdma_cs.h"

void binding_of_isaac(struct rdma_cm_id *, short);

int main(int argc, char **argv){
	// Get port from arguments
	short port;
	if(argc == 2)
		port = atoi(argv[1]);
	else
		port = 0;
	// Create event channel
	struct rdma_event_channel *event_channel = rdma_create_event_channel();
	if(event_channel == NULL)
		stop_it("rdma_create_event_channel()", errno);
	// Create the ID
	struct rdma_cm_id *cm_id;
	if(rdma_create_id(event_channel, &cm_id, "qwerty", RDMA_PS_TCP))
		stop_it("rdma_create_id()", errno);
	// Bind to the port
	binding_of_isaac(cm_id, port);
	// Listen for connection requests
	if(rdma_listen(cm_id, 1))
		stop_it("rdma_listen()", errno);
	printf("Listening for connection requests...\n");
	// Make an ID specific to the client that connected
	struct rdma_cm_id *client_id;
	client_id = cm_event(event_channel, RDMA_CM_EVENT_CONNECT_REQUEST);
	cm_event(event_channel, RDMA_CM_EVENT_ESTABLISHED);
	// Register memory region
	struct ibv_mr *mr = ibv_reg_mr(client_id->qp->pd, malloc(SERVER_MR_SIZE),SERVER_MR_SIZE,
	 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
	if(mr == NULL)
		stop_it("ibv_reg_mr()", errno);
	// Exchange addresses and rkeys with the server
	uint32_t rkey;
	uint64_t remote_addr;
	swap_info(client_id, mr, &rkey, &remote_addr, NULL);
	// The real good
	uint32_t opcode;
	while(1){
		rdma_recv(client_id, mr);
		opcode = get_completion(client_id, RECV, 1);
		if(opcode == DISCONNECT){
			printf("Client issued a disconnect.\n");
			break;
		} 
	}
	// Disconnect
	obliterate(cm_id, client_id, mr, event_channel);
	return 0;
}

// Bind to the specified port. If zero, bind to a random open port.
void binding_of_isaac(struct rdma_cm_id *cm_id, short port){
	struct sockaddr_in sin;
	memset(&sin, 0, sizeof(struct sockaddr_in));
	sin.sin_family = AF_INET;
	sin.sin_port = htons(port);
	if(rdma_bind_addr(cm_id, (struct sockaddr *)&sin))
		stop_it("rdma_bind_addr()", errno);
	printf("RDMA device bound to port %u.\n", ntohs(rdma_get_src_port(cm_id)));
}