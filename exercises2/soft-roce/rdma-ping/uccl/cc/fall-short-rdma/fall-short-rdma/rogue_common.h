
#include <sched.h>


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>

#include <netdb.h> 
#include <rdma/rdma_cma.h>

#include "rogue.h"
//#define BILLION 1000000000L

#define IB_PHYS_PORT 2
//#define GID_INDEX 2 
#define SL 0

#define MAX_TIMESTAMP_SIZE	10000 
#define REQ_AREA_SHM_KEY 3185
#define RESP_AREA_SHM_KEY 3186


#define MAX_INLINE_SIZE 256
//#define MAX_SEND_WR 4096 
#define ADD_SIZE 0
#define COMMON_ROCE 0
#define COMMON_RONE 1

// Compare, print, and exit
#define CPE(val, msg, err_code) \
	if(val) { fprintf(stderr, msg); fprintf(stderr, " Error %d \n", err_code); \
	exit(err_code);}


struct connection {
	struct qp_attr *local_qp_attr;
	struct qp_attr *remote_qp_attr;
	struct ibv_ah *ah;
	struct ibv_qp *qp;
	int isSignal;
	int num_completions;
	int num_sendcount;
	int num_requests;

	int isClient;
	struct ibv_mr *recv_mr;
	struct ibv_mr *send_mr;
	char *recv_region;
	char *send_region;

	int send_message_size;
	int send_transport;
	int window_size;
	int signal_all;
	int actual_completions;

 	int server_socketfd;	
	struct ibv_mr *peer_mr;
};

struct context {
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_cq *send_cq;

	struct Queue *usr_recv_cq;
	struct Queue *usr_send_cq;
	
	struct ibv_comp_channel *comp_channel;
	struct ibv_comp_channel *send_comp_channel;
	struct ibv_context *ctx;
	struct connection *conn;
	rudp_srv_state_t *srv_state;
	rudp_cli_state_t *cli_state;
	//struct qp_attr *local_qp_attr;
	//struct qp_attr *remote_qp_attr;
	//struct ibv_qp *qp;
	//struct ibv_mr *recv_mr;
	//struct ibv_mr *send_mr;
	//char *recv_region;
	//char *send_region;
	//int num_completions;
	//int num_sendcount;

	uint64_t  recv_bytes;
	pthread_t cq_poller_thread;
	pthread_t send_cq_poller_thread;
	pthread_t write_poller_thread;
	pthread_t throughput_timer_thread;
};


//void die(const char *reason);
void bindingCPU(int num);
struct context * init_ctx(struct ibv_device *dev, struct context *s_ctx);

void modify_qp_to_init(struct connection *conn);
void ud_modify_qp_to_init(struct connection *conn);
struct context * build_context(struct ibv_context *verbs,struct context *s_ctx);
void build_qp_attr(struct ibv_qp_init_attr *qp_attr, struct context *s_ctx);
/*static void * poll_cq(void *);
  static void * poll_send_cq(void *);
  static void * write_cq(void *);
 */
void record_timestamp(int count,  long long unsigned * start_timestamp);
void check_timestamp_enable(struct ibv_context *ctx);
void * throughput_timer(void *);
void get_qp_info(struct context *s_ctx);
uint16_t get_local_lid(struct ibv_context *ctx);
union ibv_gid  get_gid(struct ibv_context *ctx);
void set_memory(struct connection *ctx, struct context *s_ctx, int use_write, int key);
void print_qp_attr(struct qp_attr *dest);
int connect_ctx(struct connection *conn, int my_psn, struct qp_attr *dest, int priority, int isRone);
int modify_dgram_qp_to_rts(struct connection *ctx, int localpsn);
void per_request_latency(struct connection *conn, long long unsigned *start, long long unsigned *end );
void avg_latency(struct connection *conn);
void create_ah(struct connection *conn, struct ibv_pd *pd); 
void post_receives_comm(struct connection *conn);


int on_send(void *context, int sig);
uint64_t on_write_read(struct context *ctx, struct connection *conn, int sig, int op, int isRone);
int on_disconnect(struct context *ctx);
void fillzero(char *region, char c, int send_message_size);
//const  char *ibv_wc_opcode_str(enum ibv_wc_opcode opcode);



