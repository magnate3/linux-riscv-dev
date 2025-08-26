/*
 * rudp.h - Private header for the protocol layer
 *
 * October 2012
 */

#ifndef RUDP_H
#define RUDP_H

#include <strings.h>
#include <string.h>
#include <sys/socket.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs_exp.h>


//#include "unp.h"
//#include "unprtt.h"
#include "timely.h"
//#include "pri_dcbnetlink.h"
/* Defines */
#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)


#define IFACE "eno1d1"

#define RUDP_PAYLOAD_SIZE              sizeof(struct hdr) 
#define RUDP_TIME_WAIT_INTERVAL         10
#define RUDP_PERSIST_TIMER_INTERVAL     5
#define RUDP_CLIENT_TIMEOUT             60

#define CONGESTION_STATE_SLOW_START     0
#define CONGESTION_STATE_AVOIDANCE      1

#define MSG_TYPE_FILENAME               1
#define MSG_TYPE_CONNECTION_PORT        2
#define MSG_TYPE_FILE_DATA              3
#define MSG_TYPE_ACKNOWLEDGEMENT        4
#define MSG_TYPE_FIN                    5
#define MSG_TYPE_WINDOW_PROBE           6
#define MSG_TYPE_ERROR_INVALID_FILE     7

#define RECV_BUFFER_SIZE (1024 * 1024 * 4)
#define USEC_IN_SEC                     1000000

#define	MAXLINE		4096	/* max text line length */
#define MAX_SEND_WR     1024	
#define BILLION 	1000000000L
#define NIC_CLOCK_MHZ 	317
#define MAX_32BIT_NUM   4294967295
#define LINE_RATE 	10000  /*we take the microsecods*/
/*typedef struct rudp_cli_state_s {
    uint32_t    advw_size;
    uint32_t    advw_start;
    uint32_t    advw_end;
    uint32_t    advw_free;
    uint32_t    expected_seq;
    int         random_seed;
    double      data_loss_prob;
    int         recv_rate;
    uint8_t     fin_received;
    rudp_payload_t  *advw;
    pthread_mutex_t advw_lock;
} rudp_cli_state_t;
*/
typedef struct rudp_cli_state_s {
	struct qp_attr *local_qp_attribute;
	struct qp_attr *remote_qp_attribute;
	struct ibv_context *ib_ctx;
	struct ibv_qp *qp;
	
	struct ibv_cq *send_cq;
	struct ibv_comp_channel *recv_channel;
	struct ibv_comp_channel *send_channel;

	int opcode;
	int send_transport;
	uint64_t basic_clock;
	struct ibv_mr *peer_mr;
		char *recv_region;
	struct ibv_mr *recv_mr;
char *send_region;
	struct ibv_mr *send_mr;
	int inter_port_no;
	struct dcbnetlink_state dcbstate; 
} rudp_cli_state_t;

/* Function prototypes */
//void die(const char *reason);

const char *ibv_wc_opcode_str(enum ibv_exp_wc_opcode opcode);

ssize_t readline(int fd, void *vptr, size_t maxlen);
int rudp_srv_init (rudp_srv_state_t *state, rudp_srv_state_t *srv_state);
int rudp_srv_destroy (rudp_srv_state_t *srv_state);
int rudp_srv_conn_send (int fd1, int fd2, struct sockaddr_in *peer_addr, 
                        void *data, int size);
int rudp_srv_conn_recv (int fd, void *buf, int size, 
                        struct sockaddr *src_addr, 
                        int *src_len);

int init_connect(int portno, char *hostname);
uint64_t polling_write(char *region, struct ibv_cq *cq, int index); 
int rudp_send_read_write (
struct ibv_send_wr *wr, struct ibv_comp_channel *comp_channel, 
struct ibv_qp *qp, int seq, struct ibv_cq *cq, rudp_srv_state_t *srv_state);
int wait_for_acks(struct ibv_qp *qp, struct ibv_comp_channel *send_comp_channel, uint32_t bytes_remaining, struct ibv_cq *cq, rudp_srv_state_t *srv_state);
void init_rtt();

uint64_t query_hardware_time(struct ibv_context *ctx);	
void *watchdog_server(void *ctx);

int ibv_poll_cq_y(Queue *q, int num, struct ibv_exp_wc *wc);

int rudp_close (int fd);
int rudp_send_ctrl_packet (int fd, int msg_type);

int rudp_cli_init (rudp_cli_state_t *state, rudp_cli_state_t *cli_state);
int rudp_cli_destroy (rudp_cli_state_t *state);
int rudp_cli_conn_send (int fd, struct sockaddr_in *peer_addr, 
                        void *data, int size, void *recv_data, int recv_size,
                        int (*reconnect_fn)(int, void *, void *));
int rudp_cli_get_sleep_interval (void);
int rudp_cli_transfer_complete (void);
int rudp_read (char *buf);
int rudp_recv (int fd, struct sockaddr *src_addr, int *src_len);

#endif /* RUDP_H */
