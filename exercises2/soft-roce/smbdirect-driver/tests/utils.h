#ifndef __UTILS_H__
#define __UTILS_H__

#define RECV_BUFFER_SIZE 8192
#define SEND_BUFFER_SIZE 8192
#define MAX_RDMA_READ (1024 * 1024)
#define MAX_RDMA_WRITE (1024 * 1024)

#define RDMA_READ 1
#define RDMA_WRITE 0

#define SMBD_UNCONNECTED 0
#define SMBD_CONNECTED 1

struct connection_struct {
	int connected;
	void *private;
	int state;
	struct rdma_cm_id *id;
	struct ibv_context *ctx;
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_comp_channel *comp_chan;
	struct ibv_qp *qp;
	struct ibv_mr *recv_mr;
	struct ibv_mr *send_mr;
	struct ibv_mr *rdma_buffer_mr;
	char recv_buffer[RECV_BUFFER_SIZE];
	int recv_length;
	char send_buffer[SEND_BUFFER_SIZE];
	int send_length;
	char *rdma_buffer;
};

struct smbd_negotiate_req {
	uint16_t min_version;
	uint16_t max_version;
	uint16_t reserved;
	uint16_t credits_requested;
	uint32_t preferred_send_size;
	uint32_t max_receive_size;
	uint32_t max_fragmented_size;
} __attribute__((packed));

struct smbd_negotiate_resp {
	uint16_t min_version;
	uint16_t max_version;
	uint16_t negotiated_version;
	uint16_t reserved;
	uint16_t credits_requested;
	uint16_t credits_granted;
	uint32_t status;
	uint32_t max_read_write_size;
} __attribute__((packed));

void fill_mem(char *buffer, uint size, char *msg, uint port);
int send_message(struct connection_struct *connection);
int post_receive(struct connection_struct *connection);
int post_rdma_read(struct connection_struct *connection, 
		   struct ibv_mr *mr,
		   int read,
		   size_t buffer_size);
void build_negotiate_req(struct connection_struct *connection);
void build_negotiate_resp(struct connection_struct *connection);
#endif
