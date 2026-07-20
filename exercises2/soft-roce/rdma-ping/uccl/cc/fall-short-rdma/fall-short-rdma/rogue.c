/*
 * rudp.c
 *
 *
 * October 2016
 */

/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <assert.h>
#include <setjmp.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <inttypes.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs_exp.h>

#include "rogue.h"

//#define ENABLE_RL (1)
#define RL_PROBE (1.0)

/* Globals */
//rudp_srv_state_t **srv_state;
//rudp_cli_state_t **cli_state;
//static struct rtt_info rttinfo;
//static int rttinit = 0;
//static struct msghdr msgsend, msgrecv;
static sigjmp_buf jmpbuf;
//struct hdr sendhdr; //recvhdr;
/* Defines */
//#define FILE_PAYLOAD_SIZE       (RUDP_PAYLOAD_SIZE - sizeof(struct hdr))
#define SEG_MESSAGE_SIZE        (int)pow(2, 7+IBV_MTU_SIZE) 
#define FILE_PAYLOAD_SIZE       sizeof(struct hdr)
/****************************************************************************/
/*           Common routines related to both Server and Client              */
/****************************************************************************/

uint64_t query_hardware_time(struct ibv_context *ctx);
uint32_t query_hardware_elapsed_time(struct ibv_context *ctx, uint64_t basic_clock);

#define WRITE_CHECK_BYTES  1024
#define LOOK_AHEAD         16
uint64_t polling_write(char *region, struct ibv_cq *recv_cq, int index){
	uint32_t offset=0;
	uint64_t recv_bytes=0;
	while(1){
		offset = offset%RECV_BUFFER_SIZE;
		char c=(char)region[offset];
		if(c!='q'){
			memset(&region[offset],'q', WRITE_CHECK_BYTES );
			recv_bytes += WRITE_CHECK_BYTES;
			offset = offset + WRITE_CHECK_BYTES;
		}else{  
			int i=0;
			int cnt=0;
			for(i=0;i< LOOK_AHEAD;i++){
				if((char)region[(offset+ i*WRITE_CHECK_BYTES)%RECV_BUFFER_SIZE] !='q'){
					cnt++;
					if(cnt==3)
						break;
				}
			}
			if(cnt == 3){	
				memset(&region[offset],'q', (i+1)*WRITE_CHECK_BYTES );
				//	printf("index %d, offset %"PRIu32"\n",index, offset);
				recv_bytes += 3*WRITE_CHECK_BYTES;
				offset= offset+(i+1)*WRITE_CHECK_BYTES;
			}else{
				struct ibv_exp_wc wc;
				int k = ibv_exp_poll_cq(recv_cq, 1, &wc, sizeof(wc));
				if(k > 0){
					printf("recv_bytes %"PRIu64"\n", recv_bytes);
					return recv_bytes;
				}

			}
		}
	}
}







const char *ibv_wc_opcode_str(enum ibv_exp_wc_opcode opcode)
{
	switch (opcode) {
		case IBV_EXP_WC_SEND:               return "IBV_WC_SEND";
		case IBV_EXP_WC_RDMA_WRITE:         return "IBV_WC_RDMA_WRITE";
		case IBV_EXP_WC_RDMA_READ:          return "IBV_WC_RDMA_READ";
		case IBV_WC_COMP_SWAP:          return "IBV_WC_COMP_SWAP";
		case IBV_WC_FETCH_ADD:          return "IBV_WC_FETCH_ADD";
		case IBV_WC_BIND_MW:            return "IBV_WC_BIND_MW";
						/* recv-side: inbound completion */
		case IBV_EXP_WC_RECV:               return "IBV_WC_RECV";
		case IBV_EXP_WC_RECV_RDMA_WITH_IMM: return "IBV_WC_RECV_RDMA_WITH_IMM";
		default:                        return "IBV_WC_UNKNOWN";
	};
}



void err_sys(const char* x) 
{ 

	perror(x); 
	exit(1); 
}
/*	ssize_t
	readline(int fd, void *vptr, size_t maxlen)
	{
	ssize_t	n, rc;
	char	c, *ptr;

	ptr = vptr;
	for (n = 1; n < maxlen; n++) {
	if ( (rc = read(fd, &c, 1)) == 1) {
 *ptr++ = c;
 if (c == '\n')
 break;
 } else if (rc == 0) {
 if (n == 1)
 return(0);	// EOF, no data read 
 else
 break;		// EOF, some data was read 
 } else
 return(-1);	//error 
 }

 *ptr = 0;
 return(n);
 }

 ssize_t
 Readline(int fd, void *ptr, size_t maxlen)
 {
 ssize_t		n;

 if ( (n = readline(fd, ptr, maxlen)) == -1)
 err_sys("readline error");
 return(n);
 }
 */
int post_rate(uint32_t imm_data, rudp_srv_state_t *srv_state, int use_imm){
	//printf("post rate\n");
	struct ibv_sge sg;
	struct ibv_send_wr wr;
	struct ibv_send_wr *bad_wr;

	memset(&sg, 0, sizeof(sg));
	sg.addr	  = (uintptr_t)srv_state->send_region;
	sg.length = 0;
	sg.lkey	  = srv_state->send_mr->lkey;

	memset(&wr, 0, sizeof(wr));
	wr.wr_id      = 0;
	wr.sg_list    = &sg;
	wr.num_sge    = 1;
	if(use_imm){
		wr.opcode     = IBV_WR_RDMA_WRITE_WITH_IMM;
		wr.imm_data   = htonl(imm_data);
	}else{
		wr.wr_id      = query_hardware_time(srv_state->ib_ctx);
		wr.opcode     = IBV_WR_RDMA_WRITE;
		wr.send_flags |= IBV_SEND_SIGNALED;
	}
	wr.wr.rdma.remote_addr = (uintptr_t )srv_state->peer_mr->addr;		
	wr.wr.rdma.rkey        = srv_state->peer_mr->rkey;

	if (ibv_post_send(srv_state->qp, &wr, &bad_wr)) {
		fprintf(stderr, "Error, ibv_post_send() failed\n");
		return -1;
	}	
}
int run_dcbnetlink(struct dcbnetlink_state dcbstate, uint64_t rate){
	int i;


	/* Set the rate-limits */
	for (i = 0; i < NUMPRI; i++) {
		dcbstate.rates[i] = rate ; //7 Gbps
		//printf("dcbstate i %d, %"PRIu64"\n", i, dcbstate.rates[i]);
	}

	if (send_dcbnetlink_msg(&dcbstate) < 0) {
		fprintf(stderr, "Error sending DCB NETLINK message\n");
		return -1;
	}

}

void post_receive(struct ibv_qp *qp, char *recv_region, struct ibv_mr *recv_mr)
{
	struct ibv_recv_wr wr, *bad_wr = NULL;

	struct ibv_sge sge;
	memset(&wr, 0, sizeof(wr));

	//wr.wr_id = (uintptr_t)id;
	wr.sg_list = &sge;
	wr.num_sge = 1;

	sge.addr = (uintptr_t)recv_region;
	sge.length = RECV_BUFFER_SIZE;
	sge.lkey = recv_mr->lkey;


	TEST_NZ(ibv_post_recv(qp, &wr, &bad_wr));
}


uint64_t query_hardware_time(struct ibv_context *ctx){
	/*	struct ibv_exp_device_attr attr;
		memset(&attr, 0, sizeof(attr));
		attr.comp_mask = IBV_EXP_DEVICE_ATTR_WITH_HCA_CORE_CLOCK | IBV_EXP_DEVICE_ATTR_WITH_TIMESTAMP_MASK;
		int err= ibv_exp_query_device(ctx, &attr);


		if (attr.comp_mask & IBV_EXP_DEVICE_ATTR_WITH_TIMESTAMP_MASK){
		printf("com_mask is on\n");
		if(attr.timestamp_mask){
		printf("timestamp_mask  %lld\n", (long long)attr.timestamp_mask);
		}
		}

		if(attr.comp_mask & IBV_EXP_DEVICE_ATTR_WITH_HCA_CORE_CLOCK){
		printf("com_mask with HCA_CORE_CLOCK is on\n");
		if(attr.hca_core_clock){
		printf("hca_core_clock  %lld\n", (long long)attr.hca_core_clock);
		}
		}

		printf("query hardware time\n");*/
	struct ibv_exp_values queried_values;
	memset(&queried_values, 0, sizeof(queried_values));
	queried_values.comp_mask = IBV_EXP_VALUES_HW_CLOCK | IBV_EXP_VALUES_HW_CLOCK_NS;
	int ret = ibv_exp_query_values(ctx, IBV_EXP_VALUES_HW_CLOCK | IBV_EXP_VALUES_HW_CLOCK_NS, &queried_values);
	if (!ret && (queried_values.comp_mask & IBV_EXP_VALUES_HW_CLOCK)){ 
		//printf("hw_clock = %llu\n", (long long unsigned)queried_values.hwclock);
		return queried_values.hwclock;
	}else
		return 0;
}

uint32_t query_hardware_elapsed_time(struct ibv_context *ctx, uint64_t basic_clock){
	uint64_t curr = query_hardware_time(ctx);
	uint32_t elapsed = (curr-basic_clock)%MAX_32BIT_NUM;
	return elapsed;
}
/*
 * sigalarm_handler
 *
 * Handler for SIGALRM. We just set the jmpbuf here inorder to avoid race
 * conditions.
 */
	static void 
sigalarm_handler (int signo)
{
	siglongjmp(jmpbuf, 1);
}

/*
 * rudp_start_timer
 *
 * Start a timer with the specified interval
 */
	static void
rudp_start_timer (uint32_t ivl)
{
	struct itimerval timer;

	timer.it_interval.tv_sec = 0;
	timer.it_interval.tv_usec = 0;
	timer.it_value.tv_sec = (ivl / USEC_IN_SEC);
	timer.it_value.tv_usec = (ivl % USEC_IN_SEC);
	//	printf("ivl [ %"PRIu32" ]\n", ivl);	
	//printf("tv_sec %d, tv_usec %d\n", (ivl/USEC_IN_SEC), (ivl % USEC_IN_SEC));
	setitimer(ITIMER_REAL, &timer, 0);
}

static void rudp_update_timer(){

}
/*
 * rudp_stop_timer
 *
 * Stop a currently running timer
 */
	static void
rudp_stop_timer (void)
{
	struct itimerval timer;

	timer.it_interval.tv_sec = 0;
	timer.it_interval.tv_usec = 0;
	timer.it_value.tv_sec = 0;
	timer.it_value.tv_usec = 0;
	setitimer(ITIMER_REAL, &timer, 0);
}

/****************************************************************************/
/*                       Server related routines                            */
/****************************************************************************/
int init_connect(int portno, char *hostname){
	int sockfd,  n;
	struct sockaddr_in serv_addr;
	struct hostent *server;

	//	printf("internal connection host name %s, portno %d\n", hostname, portno);
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0) 
		die("ERROR opening socket");
	server = gethostbyname(hostname);
	if (server == NULL) {
		fprintf(stderr,"ERROR, no such host\n");
		exit(0);
	}
	bzero((char *) &serv_addr, sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	bcopy((char *)server->h_addr, 
			(char *)&serv_addr.sin_addr.s_addr,
			server->h_length);
	serv_addr.sin_port = htons(portno);
	if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
		die("ERROR connecting @internal tcp ");

	/*char v[64];
	  sprintf(v,"%llu" , basic_clock);

	  printf("own clock [ %"PRIu64" ]\n", basic_clock);	
	  n = send(sockfd,v,sizeof(v),0);
	  if (n < 0) 
	  die("ERROR writing to socket");
	  memset(v, 0, sizeof(v));
	  n = recv(sockfd,v,sizeof(v),0);
	  if (n < 0) die("ERROR reading from socket");
	  printf("received value %s\n", v);
	//basic_clock = GET_MIN(strtoll(v, NULL, 10), basic_clock);
	 */	
	//	printf("final clock %"PRIu64"\n", basic_clock);


	return sockfd;
}

uint32_t ctrl_message_send(int sockfd, char * buffer, int need_recv, int size){
	if(size == 0)
		size = sizeof(buffer);
	int n = send(sockfd,buffer,size,0);
	if (n < 0) 
		die("ERROR writing to socket");

	uint32_t psn;
	if(need_recv){
		n = recv(sockfd,&psn, sizeof(psn),0);
		return ntohl(psn);
		printf("psn %"PRIu32"\n", ntohl(psn));
	}


	return 0;
}

/*void * srv_poll_cq(void *ev_ctx){
  struct ibv_exp_wc wc[MAX_SLOTS_NUM];
  int i=0;
  struct ibv_cq *cq;
//for(i=0;i<MAX_SLOTS_NUM;i++)
// memset(&wc[i], 0, sizeof(wc[0]));

int ack_i=0;
int k=0;
while(1){
TEST_NZ(ibv_get_cq_event(srv_state->send_channel, &cq, &ev_ctx));
ibv_ack_cq_events(cq, MAX_SLOTS_NUM);
TEST_NZ(ibv_req_notify_cq(cq, 0));
k = ibv_exp_poll_cq(cq, MAX_SLOTS_NUM, wc, sizeof(wc[0]));
while(k){
srv_state->cwnd_free += k;	
//	printf("k = %d ***************************************\n",k);
k = ibv_exp_poll_cq(cq, MAX_SLOTS_NUM, wc, sizeof(wc[0]));

}
}

}*/
/*
 * rudp_srv_init
 *
 * Initializes the RUDP layer for the server
 */
	int
rudp_srv_init (rudp_srv_state_t *state, rudp_srv_state_t *srv_state)
{
	int i;

	/* Initialize the state parameters */
	//	srv_state->max_cwnd_size = MAX_CWND_SIZE;//state->max_cwnd_size;
	srv_state->local_qp_attribute = state->local_qp_attribute;
	srv_state->remote_qp_attribute = state->remote_qp_attribute;
	srv_state->send_transport = state->send_transport;
	srv_state->opcode = state->opcode;
	srv_state->ib_ctx = state->ib_ctx;
	srv_state->qp = state->qp;	
	srv_state->recv_cq = state->recv_cq;
	srv_state->send_cq = state->send_cq;
	srv_state->send_channel = state->send_channel;
	srv_state->recv_channel = state->recv_channel;

	srv_state->send_mr = state->send_mr;
	srv_state->send_region = state->send_region;
	srv_state->recv_region = state->recv_region;
	srv_state->recv_mr = state->recv_mr;
	srv_state->peer_mr = state->peer_mr;


	srv_state->usr_send_cq = state->usr_send_cq;
	srv_state->usr_recv_cq = state->usr_recv_cq;
	srv_state->hostname = state->hostname;
	srv_state->inter_port_no = state->inter_port_no;

	srv_state->event = state->event;
	srv_state->rate_limit_mode = state->rate_limit_mode;
	srv_state->vegas_previous_ack=0;
	srv_state->current_seq = 0;
	srv_state->pending_acks = 0;
	srv_state->cwnd_size_cnt = 0;
	tcp_timely_init(srv_state);

	srv_state->t_enqueue_point=0;
	srv_state->cwnd_update_cnt = 0;
	srv_state->hw_timeout = 8;
	srv_state->send_bytes = 0;
	srv_state->cp_bytes = 0;
	srv_state->cwnd_bytes = INIT_CWND; /* CW starts with size 1 in slow start phase */
	srv_state->rate_limiter = 10.0;  //start with line rate
	srv_state->conf_rate_limiter = 10.0;  //start with line rate
	srv_state->max_rate_limiter = 0.1;  //start with line rate
	srv_state->free_bytes = srv_state->cwnd_bytes;
	srv_state->free_slots = MAX_SLOTS_NUM;
	srv_state->ss_thresh = RUDP_DEFAULT_SSTHRESH;
	srv_state->cwnd_start = 0;
	srv_state->cwnd_end = 0;
	srv_state->expected_ack = 0; /* First packet sent has sequence number 0 */
	//srv_state->num_acks = 0;
	//srv_state->num_dup_acks = 0;
	//srv_state->last_dup_ack = 0;
	srv_state->rudp_state = CONGESTION_STATE_SLOW_START;

	//TODO
	// Just make it run now.
	srv_state->advw_size = MAX_SEND_WR;
	/* Initialize the congestion window */
	srv_state->cwnd = 
		(rudp_payload_t *)malloc(MAX_SLOTS_NUM * 
				sizeof(rudp_payload_t));
	if (!srv_state->cwnd) {
		return -1;
	}
	bzero(srv_state->cwnd, MAX_SLOTS_NUM * sizeof(rudp_payload_t));

	for (i = 0; i < MAX_SLOTS_NUM; i++) {
		srv_state->cwnd[i].data = (char *)malloc(RUDP_PAYLOAD_SIZE);
		if (!srv_state->cwnd[i].data) {
			return -1;
		}
		bzero(srv_state->cwnd[i].data, RUDP_PAYLOAD_SIZE);
	}

	/* initilize the basic timestamp*/
	srv_state->basic_clock = query_hardware_time(srv_state->ib_ctx);
	srv_state->prev_timestamp = srv_state->basic_clock;
	if(srv_state->send_transport == RC_TRANSPORT)
		srv_state->sockfd = init_connect(srv_state->inter_port_no,srv_state->hostname);

	if(srv_state->opcode == WRITE_OPERATOR || srv_state->opcode == SEND_OPERATOR){

		if (init_dcbnetlink(&srv_state->dcbstate,IFACE) < 0) {
			fprintf(stderr, "Error initializing DCB NETLINK state\n");
			return -1; 
		}
		run_dcbnetlink(srv_state->dcbstate, 0);
		//printf("init_dcbnetlink\n");
	}

	for(i=0;i<MAX_SEND_WR/2;i++){
		post_receive(srv_state->qp, srv_state->recv_region, srv_state->recv_mr);

	}
	/* Register for SIGALRM to handle timeouts */
	if(srv_state->send_transport == UC_TRANSPORT)
		signal(SIGALRM, sigalarm_handler);

	//pthread_create(&cq_poller_thread, NULL, srv_poll_cq,NULL);
	pthread_t cq_poller_thread;
	//	printf("init server done\n");
	return 0;
}

/*
 * rudp_srv_destroy
 *
 * Cleans up the resources allocated to the RUDP layer on the server
 */
	int
rudp_srv_destroy (rudp_srv_state_t *srv_state)
{
	int i;

	/* Free the allocated resources */
	if (srv_state) {
		if (srv_state->cwnd) {
			for (i = 0; i < MAX_SLOTS_NUM; i++) {
				if (srv_state->cwnd[i].data) {
					free(srv_state->cwnd[i].data);
				}
			}
		}
		free(srv_state->cwnd);
		free(srv_state);
	}

	return 0;
}
/*
 * add_packet_to_cwnd
 *
 * Form the packet from the given parameters and add it to the congestion
 * window
 */
	static int 
add_packet_to_cwnd (struct hdr *old_packet_hdr, uint32_t len, struct ibv_send_wr new_wr, rudp_srv_state_t *srv_state)
{
	char *ptr = NULL;

	struct hdr *packet_hdr = (struct hdr *)malloc(sizeof(struct hdr));
	packet_hdr->msg_type = htonl(MSG_TYPE_FILE_DATA);
	packet_hdr->wr = old_packet_hdr->wr;
	packet_hdr->end = old_packet_hdr->end;
	//packet_hdr->ts = old_packet_hdr->ts;
	packet_hdr->seq = old_packet_hdr->seq;
	packet_hdr->retransmit = 0;
	packet_hdr->window_size = 0;
	packet_hdr->nic_ts = old_packet_hdr->nic_ts;
	/* Sanity check */
	//	printf("seq %d, cwnd_end %d\n", old_packet_hdr->seq, srv_state->cwnd_end);
	//	assert(srv_state->cwnd[srv_state->cwnd_end].valid == 0);

	/* New packet always goes to where cwnd_end is pointing */
	srv_state->cwnd[srv_state->cwnd_end].valid = 1;
	srv_state->cwnd[srv_state->cwnd_end].data_size = len;

	struct ibv_sge sge, sge_p;
	struct ibv_send_wr new_wr_p; 
	sge = *new_wr.sg_list;	
	memset(&sge_p, 0, sizeof(sge_p));
	sge_p.addr = sge.addr ;
	sge_p.length = sge.length ;
	sge_p.lkey = sge.lkey ;

	memset(&new_wr_p, 0, sizeof(new_wr_p));
	new_wr_p.wr_id = (uintptr_t)packet_hdr;
	new_wr_p.opcode = new_wr.opcode;
	if(new_wr.opcode == IBV_WR_RDMA_WRITE_WITH_IMM)
		new_wr_p.imm_data = new_wr.imm_data;
	new_wr_p.sg_list = &sge_p;
	new_wr_p.num_sge = new_wr.num_sge ;
	new_wr_p.send_flags = new_wr.send_flags ;

	new_wr_p.wr.rdma.remote_addr =  new_wr.wr.rdma.remote_addr ; 
	new_wr_p.wr.rdma.rkey = new_wr.wr.rdma.rkey;


	srv_state->cwnd[srv_state->cwnd_end].wr=new_wr_p;

	ptr = (char*)srv_state->cwnd[srv_state->cwnd_end].data;
	memcpy(ptr, packet_hdr, sizeof(struct hdr));


	srv_state->free_bytes = (srv_state->free_bytes >= len)?srv_state->free_bytes-len: 0;
	srv_state->free_slots--;
	//srv_state->cwnd_free--;
	srv_state->cwnd_end = (srv_state->cwnd_end + 1) % MAX_SLOTS_NUM;
	return 0;
}
/*
 * print_cwnd
 *
 * Print the congestion window 
 */
	static void
print_cwnd (rudp_srv_state_t *srv_state)
{
	int i,idx;

	int start = srv_state->cwnd_start;	
	//int end = srv_state->cwnd_end;	
	/*	printf("start %d end %d size %d\n", 
		srv_state->cwnd_start, srv_state->cwnd_end, srv_state->cwnd_size);

		idx = (srv_state->cwnd_size+start-1)% MAX_SLOTS_NUM;
		printf("start seq: %d, end seq: %d\n", ((struct hdr *)(srv_state->cwnd[start].data))->seq,
		((struct hdr *)(srv_state->cwnd[idx].data))->seq); */
	//for (i = start; i < srv_state->cwnd_size+start; i++) {
	//	idx = i% MAX_SLOTS_NUM;	
	//	printf("seq number %d ", ((struct hdr *)(srv_state->cwnd[idx].data))->seq);
	//}
	//printf("\n");
}
/*
 * find_packet_count
 *
 * Determine the number of packets that can be sent based on the congestion
 * window and advertised window state.
 */
/*	static inline int 
	find_packet_count (rudp_srv_state_t *srv_state)
	{
	int num = GET_MIN(srv_state->advw_size, srv_state->cwnd_free);
	return num;
	}*/

/*
 * update_cwnd_after_timeout
 *
 * This is called when a sent packet is lost or the corresponding
 * acknowledgement is lost. Here, we shrink the congestion window to the new
 * size passed and update the window parameters accordingly.
 */
	static void
update_cwnd_after_timeout (int new_cwnd_bytes, int new_ss_thresh, 
		uint64_t retransmited, rudp_srv_state_t *srv_state)
{
	int i, idx, bytes, start, end, valid;
	rudp_payload_t *new_cwnd;
	uint64_t  old_window_bytes;

	//FIXME here we assume that the number of requests is always less < MAX_SLOTS_NUM- extra_slots, thus, we do not allocate new cwnd
	/*	if(valid >= (MAX_SLOTS_NUM-extra_slots)){
	// Allocate a new window copy 
	int allocated_slots = valid + extra_slots;
	new_cwnd = (rudp_payload_t *)malloc((allocated_slots) *
	sizeof(rudp_payload_t));
	if (!new_cwnd) {
	fprintf(stderr,"update_cwnd_after_timeout: no memory\n");
	assert(0);
	}
	bzero(new_cwnd, allocated_slots * sizeof(rudp_payload_t));

	for (i = 0; i < allocated_slots; i++) {
	new_cwnd[i].data = (char *)malloc(RUDP_PAYLOAD_SIZE);
	if (!new_cwnd[i].data) {
	fprintf(stderr,"update_cwnd_after_timeout: no memory\n");
	assert(0);
	}
	bzero(new_cwnd[i].data, RUDP_PAYLOAD_SIZE);
	}

	// Copy relevant packets from the old copy 
	idx = srv_state->cwnd_start;
	valid = 0;
	// compute how many messages/packets whose valid == 1, i.e., packets that has not got the ack back in the new congestion window 
	for (i = 0; i < allocated_slots; i++) {
	if (srv_state->cwnd[idx].valid == 0) {
	break;
	}

	if(idx == srv_state->cwnd_bytes){
	new_cwnd[i].valid = srv_state->cwnd[idx].valid;
	new_cwnd[i].data_size = srv_state->cwnd[idx].data_size;

	new_cwnd[i].wr = srv_state->cwnd[idx].wr;
	memcpy(new_cwnd[i].data, srv_state->cwnd[idx].data, 
	RUDP_PAYLOAD_SIZE);
	idx = (idx + 1)% MAX_SLOTS_NUM;
	valid++;
	}
	}
	start = srv_state->cwnd_start;
	// current_seq = ((struct hdr *)(srv_state->cwnd[start].data))->seq + valid;
	srv_state->free_bytes = GET_MAX((new_cwnd_size - valid)*SEG_MESSAGE_SIZE, 0);

	// Free the old window and replace it with the new one 
	for (i = 0; i < MAX_SLOTS_NUM; i++) {
	if (srv_state->cwnd[i].data) {
	free(srv_state->cwnd[i].data);
	}
	}
	free(srv_state->cwnd);
	srv_state->cwnd = new_cwnd;

	}*/

	srv_state->free_slots = MAX_SLOTS_NUM- 1;
	srv_state->free_bytes = (new_cwnd_bytes>= retransmited)? (new_cwnd_bytes-retransmited): 0; 

	/* Update the window parameters */
	old_window_bytes = srv_state->cwnd_bytes;
	srv_state->ss_thresh = new_ss_thresh;
	srv_state->cwnd_bytes = new_cwnd_bytes;
	//	srv_state->cwnd_start = 0;
	//	srv_state->cwnd_end = 0;
	//	srv_state->num_acks = 0;
	//	srv_state->num_dup_acks = 0;

	/*
	 * Update pending acks count. TODO see if we can update it intelligently
	 * instead of walking the whole window.
	 */
	/*	idx = 0;
	 *pending_acks = 0;
	 for (i = 0; i < MAX_SLOTS_NUM; i++) {
	 if (srv_state->cwnd[i].valid == 0) {
	 idx = i;
	 break;
	 }
	 *pending_acks = *pending_acks + 1;
	 }
	 srv_state->cwnd_end = idx;
	 */
	/* Debug */
	fprintf(stderr,"shrinking congestion window from %d to %d. new ss_thresh: %d\n", 
			srv_state->cwnd_start, srv_state->cwnd_end, srv_state->ss_thresh);
}

/*
 * update_cwnd_after_valid_ack
 *
 * This is called to update the congestion window parameters whenever we
 * receive a valid ack.
 */
	static void
update_cwnd_after_valid_ack (rudp_srv_state_t *srv_state, uint32_t start, int num_rcvd_acks)
{
	int i, src_idx;
	//	printf("start %d, end %d\n", srv_state->cwnd_start, srv_state->cwnd_end);
	/* Update the free window slots and update the window parameters */

	srv_state->free_bytes = srv_state->cwnd_bytes;
	//	printf("update_cwnd_after_valid_ack free_bytes %d, cwnd_size %d\n", srv_state->free_bytes, srv_state->cwnd_size);
	src_idx= srv_state->cwnd_start;	
	long re = 0;
	while(srv_state->cwnd[src_idx].valid){
		//printf("i %d, src_idx %d\n", i, src_idx);
		//if (srv_state->cwnd[src_idx].valid == 1) {

		re = srv_state->free_bytes -srv_state->cwnd[src_idx].data_size;		
		srv_state->free_bytes =re < 0 ?0:re;
		if(srv_state->free_bytes ==0)
			break;
		src_idx = (src_idx+1)%MAX_SLOTS_NUM;	
		//}
	}
	//srv_state->free_bytes = GET_MAX(srv_state->free_bytes, 0);
	//printf("end update_cwnd_after_valid_ack\n");
	//srv_state->cwnd_start = 0;
	//srv_state->cwnd_end = dest_idx;
}


/*
 * rudp_handle_ack
 *
 * This is called when we receive acks for multiple packets. This can happen
 * during network congestion when packets arrive out of order. Depending on
 * what state our congestion window is in and how many acks are received, we
 * need to update our window state carefully.
 * 
 * Handle the acknowledgement. We have 3 cases here:
 * 1. We receive only one expected ack
 * 2. We receive a cumulative ack for a set of packets in the cwnd
 * 3. We receive a cumulative ack for a packet that is not there in
 *    our cwnd (Happens if we have shrunk our congestion window)
 */

	static uint32_t
rudp_handle_ack (uint32_t received_ack, uint32_t num_acks,
		int *pending_acks, rudp_srv_state_t *srv_state)
{
	int i, idx;
	uint32_t start, end, exp_ack, start_seq, last_seq;
	uint32_t orig_start;

	orig_start = start =  srv_state->cwnd_start;
	end = srv_state->cwnd_end;
	//free = srv_state->cwnd_free;
	//size = srv_state->cwnd_size;
	exp_ack = srv_state->expected_ack;

	if (end == 0) {
		end = MAX_SLOTS_NUM - 1;
	} else {
		end--;
	}
	last_seq = ((struct hdr *)(srv_state->cwnd[end].data))->seq;
	start_seq = ((struct hdr *)(srv_state->cwnd[start].data))->seq;
	//printf("start %d, end %d, start_seq, %d, last_seq %d\n", start, end, start_seq, last_seq);
	/* Case 1: We have received a single ack */
	if (num_acks == 1) {
		srv_state->cwnd[srv_state->cwnd_start].valid = 0;
		//		srv_state->cwnd_free++;
		//	bzero(srv_state->cwnd[srv_state->cwnd_start].data, RUDP_PAYLOAD_SIZE);
		//	memset(&srv_state->cwnd[srv_state->cwnd_start].wr, 0, sizeof(srv_state->cwnd[srv_state->cwnd_start].wr));
		if(srv_state->opcode == WRITE_OPERATOR)
			srv_state->cp_bytes += srv_state->cwnd[srv_state->cwnd_start].data_size;

		srv_state->free_slots++;
		//		srv_state->num_acks++;
		srv_state->cwnd_start =  
			(srv_state->cwnd_start + 1) % MAX_SLOTS_NUM;
		*pending_acks = *pending_acks - 1;
		srv_state->expected_ack += 1;

		return orig_start;
	}

	/* Case 2: All the acked packets are in the congestion window */
	if ((start_seq + num_acks) <= (last_seq + 1)) {
		//printf("cum_ack: case 1\n");
		for (i = 0; i < num_acks; i++) {
			srv_state->cwnd[srv_state->cwnd_start].valid = 0;
			//		bzero(srv_state->cwnd[srv_state->cwnd_start].data, RUDP_PAYLOAD_SIZE);
			//		memset(&srv_state->cwnd[srv_state->cwnd_start].wr, 0, sizeof(srv_state->cwnd[srv_state->cwnd_start].wr));
			if(srv_state->opcode == WRITE_OPERATOR)
				srv_state->cp_bytes += srv_state->cwnd[srv_state->cwnd_start].data_size;

			srv_state->free_slots++;
			//			srv_state->num_acks++;
			srv_state->cwnd_start = 
				(srv_state->cwnd_start + 1) % MAX_SLOTS_NUM;
			*pending_acks = *pending_acks - 1;
			srv_state->expected_ack += 1;
		}
		//printf("free_slots %d\n", srv_state->free_slots);

		return orig_start;
	}

	/* 
	 * Case 3: The packet corresponding to the received ack is not in the
	 * congestion window. This can happen during congestion when we have
	 * shrunk the congestion window. In this case, all the packets in our
	 * congestion window has been acked.
	 */
	if (received_ack >= (last_seq + 1)) {
		//	fprintf(stderr,"cum_ack: case 2\n");
		idx = start;
		while (1) {
			if (srv_state->cwnd[idx].valid == 0) {
				break;
			}
			srv_state->cwnd[idx].valid = 0;
			//		bzero(srv_state->cwnd[idx].data, RUDP_PAYLOAD_SIZE);
			if(srv_state->opcode == WRITE_OPERATOR)
				srv_state->cp_bytes += srv_state->cwnd[idx].data_size;
			//		memset(&srv_state->cwnd[idx].wr, 0, sizeof(srv_state->cwnd[idx].wr));
			srv_state->free_slots++;
			//srv_state->cwnd_free++;
			//			srv_state->num_acks++;
			*pending_acks = *pending_acks - 1;
			srv_state->expected_ack += 1;
			idx = (idx + 1) % MAX_SLOTS_NUM;
		}

		return orig_start;
	}
}

/*
 * retransmit_first_unacked_packet
 *
 * This routine retransmits the packet pointed to by cwnd_start. This is
 * invoked in case of a timeout or fast retransmit.
 */
	static uint32_t
retransmit_first_unacked_packet (struct ibv_qp *qp, rudp_srv_state_t *srv_state)
{
	struct ibv_send_wr *bad_wr=NULL;
	rudp_payload_t *payload;

	payload = &srv_state->cwnd[srv_state->cwnd_start];
	struct hdr * header =  (struct hdr *)payload->data;
	fprintf(stderr,"retransmit_first_unacked_packet start %d, seq: %d\n", srv_state->cwnd_start, header->seq);


	struct ibv_send_wr wr = payload->wr;
	wr.send_flags |= IBV_SEND_SIGNALED;
	struct hdr *retrheader = (struct hdr * )(uintptr_t)payload->wr.wr_id;
	retrheader->retransmit = 1;
	//retrheader->nic_ts = query_hardware_elapsed_time(srv_state->ib_ctx);
	//on_write_read(conn, sig, op, header);
	TEST_NZ(ibv_post_send(qp, &wr, &bad_wr));
	return payload->data_size;
}

/*
 * retransmit_unacked_packets
 * This function retransmit the first free slots packets in the congestion window.
 * This is invoked when the (end indices - start indices) > congestion window size, which
 * happens when timeout happens
 */
	static void
retransmit_unacked_packets (struct ibv_qp *qp, int *retransmit_in_flight, rudp_srv_state_t *srv_state)
{
	int i, idx;
	uint32_t start, end, start_seq, end_seq, num_packets;
	rudp_payload_t *payload;
	struct ibv_send_wr *bad_wr=NULL;

	start = srv_state->cwnd_start;
	end = srv_state->cwnd_end; /* end points to the first free slot */
	if (end == 0) {
		end = MAX_SLOTS_NUM - 1;
	} else {
		end -= 1;
	}

	/*start_seq = ((struct hdr *)(srv_state->cwnd[start].data))->seq;
	  end_seq = ((struct hdr *)(srv_state->cwnd[end].data))->seq;

	  int free_slots = srv_state->cwnd_size - *retransmit_in_flight;
	  num_packets = end_seq - start_seq + 1- *retransmit_in_flight;
	  num_packets = GET_MIN(num_packets, free_slots);
	 */

	/*	printf("start %d, end %d, num_packets %d, retransmit_in_flight %d \n", 
		start, end, num_packets, *retransmit_in_flight);*/
	idx = srv_state->cwnd_start;
	while((srv_state->free_bytes >= srv_state->cwnd[idx].data_size  || *retransmit_in_flight <= SEG_MESSAGE_SIZE 
				|| srv_state->free_slots >= MAX_SLOTS_NUM-1)
			&& srv_state->free_slots > 0 && idx <= end	) {
		assert(srv_state->cwnd[idx].valid == 1);
		payload = &srv_state->cwnd[idx];
		//	struct hdr * header =  (struct hdr *)payload->data;
		struct hdr *retrheader = (struct hdr * )(uintptr_t)payload->wr.wr_id;
		fprintf(stderr,"seq :%d, expected_ack %d\n", retrheader->seq, srv_state->expected_ack);
		if(retrheader->seq < srv_state->expected_ack)
			continue;
		retrheader->retransmit = 1;
		//retrheader->nic_ts = query_hardware_elapsed_time(srv_state->ib_ctx);
		TEST_NZ(ibv_post_send(qp, &payload->wr, &bad_wr));
		idx = (idx + 1) % MAX_SLOTS_NUM;
		*retransmit_in_flight = *retransmit_in_flight + srv_state->cwnd[idx].data_size;
		srv_state->free_bytes = (srv_state->free_bytes > srv_state->cwnd[idx].data_size)? srv_state->free_bytes-srv_state->cwnd[idx].data_size:0;
		srv_state->free_slots --;
		//idx = idx + 1;
	}
}

/*
 * rudp_send
 *
 * This functions sends a chunk of data to the remote end reliably by
 * implementing TCP like flow-control and congestion-control techniques. We
 * return from this function only when we have successfully transmitted all
 * the 'size' bytes of the passed buffer.
 */

/*void init_rtt(){
// Initialize the RTT library if we are coming here for the first time 
if (rttinit == 0) {
rtt_init(&rttinfo);
rttinit = 1;
rtt_d_flag = 1;
}

}*/

int ibv_poll_cq_y(Queue *q, int num, struct ibv_exp_wc *wc){
	int i=0;
	for(i=0;i<num;i++){
		NODE *pN=Dequeue(q);
		if(pN == NULL){
			return i;
		}
		wc[i] = *pN->wc;
	}
	return num;
}

	int
rudp_send_read_write(struct ibv_send_wr *wr, struct ibv_comp_channel *send_comp_channel, 
		struct ibv_qp *qp, int seq, struct ibv_cq *send_cq, rudp_srv_state_t *srv_state)
{
	struct hdr *packet_hdr;
	uint32_t num_packets = 0;
	//	char buffer[FILE_PAYLOAD_SIZE];
	int num_slots,ret;
	uint32_t len, offset=0;
	struct ibv_send_wr *bad_wr=NULL;

	uint32_t bytes_remaining = wr->sg_list->length;//conn->send_message_size* conn->num_requests;
	//struct iovec iovrecv[1];
	int i; //j,n

	int wait_for_acks_ret=0;
	/* 
	 * Keep sending in chunks of RUDP_PAYLOAD_SIZE bytes till we finish
	 * sending everything reliably.
	 */

	if(seq == 0){
		if(srv_state->send_transport == RC_TRANSPORT)
			post_rate(0, srv_state, 0);
	}
	while (bytes_remaining ) { //|| pending_acks
		if(srv_state->current_seq > 0){
			srv_state->message_size = GET_MAX(srv_state->cwnd_bytes/2, 64*SEG_MESSAGE_SIZE);

		}else{
			srv_state->message_size = INIT_CWND;
		}
		srv_state->message_size = GET_MIN(srv_state->message_size, bytes_remaining);
		/* Figure out how many packets to send */
		//num_slots = find_packet_count(srv_state);
		num_slots   = (int)(srv_state->free_bytes/srv_state->message_size);
		if(srv_state->free_slots > MAX_SLOTS_NUM-2)
			num_slots = srv_state->free_slots-(MAX_SLOTS_NUM-2);
		else if( srv_state->expected_ack >= srv_state->current_seq){
			srv_state->free_slots = MAX_SLOTS_NUM;
			num_slots = srv_state->free_slots-(MAX_SLOTS_NUM-2);
		}
		num_packets = (int)ceil((double)bytes_remaining / srv_state->message_size);
		num_packets = GET_MIN(num_packets, num_slots);
		num_packets = GET_MIN(num_packets, srv_state->free_slots);

		if(wait_for_acks_ret  == -1){	
			fprintf(stderr,"srv_state->free_bytes %d, free_slots %d, num_packets %d\n", srv_state->free_bytes, srv_state->free_slots, num_packets);
			fprintf(stderr,"index %d, message_size: %d cwnd_free %d, cwnd_size %d, slots %d, start %d, end %d\n", 
					srv_state->index, srv_state->message_size, srv_state->free_bytes, srv_state->cwnd_bytes, num_slots, 
					srv_state->cwnd_start, srv_state->cwnd_end);
		}

		/* Send all the packets */
		for (i = 0; i < num_packets; i++) {
			srv_state->current_seq ++;
			/* Construct the header */
			packet_hdr = (struct hdr *)malloc(sizeof(struct hdr));
			packet_hdr->msg_type = htonl(MSG_TYPE_FILE_DATA);
			packet_hdr->wr = wr;
			//packet_hdr->ts = rtt_ts(&rttinfo);
			packet_hdr->seq = srv_state->current_seq;
			//printf("packet_hdr->seq @rudp_send_read_write%d\n", packet_hdr->seq);
			packet_hdr->retransmit = 0;
			packet_hdr->window_size = 0;

			if(bytes_remaining <=srv_state->message_size)	
				packet_hdr->end = 1;
			else
				packet_hdr->end = 0;
			len = (bytes_remaining >= srv_state->message_size) ? srv_state->message_size: 
				bytes_remaining;

			//len = GET_MIN(len, srv_state->free_bytes);	


			struct ibv_send_wr new_wr;
			struct ibv_sge sge, old_sge;
			if(len == wr->sg_list->length){
				new_wr = *wr;
			}else{
				old_sge = *wr->sg_list;	
				memset(&sge, 0, sizeof(sge));
				sge.addr = (uintptr_t)old_sge.addr+offset;
				sge.length = len;
				sge.lkey = old_sge.lkey;

				//printf("offset %d, seq %d\n",offset, packet_hdr->seq);
				memset(&new_wr, 0, sizeof(new_wr));
				new_wr.sg_list = &sge;
				new_wr.num_sge = 1;

				new_wr.send_flags = wr->send_flags;

				new_wr.wr.rdma.remote_addr = (uintptr_t )wr->wr.rdma.remote_addr+ offset; 
				new_wr.wr.rdma.rkey = wr->wr.rdma.rkey;
			}
			new_wr.wr_id = (uintptr_t)packet_hdr;
			/** Copy appropriate number of bytes from user buffer to send
			 * buffer 
			 */
			/*memcpy(buffer, filebuf + offset, len);*/ 
			offset = offset+ len;

			bytes_remaining -= len;

			srv_state->send_bytes += len;
			/* 
			 * Now add this packet to cwnd.
			 * Note : this function should not return error.
			 */

			//struct timely *timely = (struct timely *)srv_state->cong;
			//			printf("thread %d, seq %d, len %d, num_slots %d, cwnd_bytes %d\n", srv_state->index, packet_hdr->seq,len, num_slots, srv_state->cwnd_bytes);

			if( srv_state->send_bytes % (16*1024) == 0 
					||	i== num_packets-1 )//1 || || 				
				packet_hdr->nic_ts = query_hardware_elapsed_time(srv_state->ib_ctx, srv_state->basic_clock);
			else{
				packet_hdr->nic_ts = 0x7fff;
				//	printf("beg_snd_nxt %d, current_seq %d\n", timely->beg_snd_nxt, srv_state->current_seq);
			}
			if(srv_state->send_transport == UC_TRANSPORT 
					&& wr->opcode == IBV_WR_RDMA_WRITE 
					&& packet_hdr->nic_ts != 0x7fff){
				new_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
				//printf("imm clock [ %"PRIu32" ]\n", packet_hdr->nic_ts);
				new_wr.imm_data = htonl(srv_state->t_enqueue_point);
				srv_state->t_enqueue[srv_state->t_enqueue_point%16]=packet_hdr->nic_ts;	
				srv_state->t_enqueue_point ++;
			}else{
				new_wr.opcode = wr->opcode;
			}
			ret = add_packet_to_cwnd(packet_hdr, len, new_wr, srv_state);
			assert(ret == 0);

			/* Now send the packet */
			//wr->wr_id = (uintptr_t)packet_hdr;
			TEST_NZ(ibv_post_send(qp, &new_wr, &bad_wr));
			srv_state->pending_acks = srv_state->pending_acks + 1;
			//srv_state->cwnd_free--;			
		}

		/*	if (num_packets > 0){
			TEST_NZ(ibv_post_send(qp, wr, &bad_wr));
			bytes_remaining -= wr->sg_list->length;
			srv_state->cwnd_free--;
			}*/
		/* 
		 * If we don't have anything to send and we have pending
		 * acknowledgements, just retransmit the first unacknowledged packet
		 * instead of waiting for the timeout and shrinking the congestion
		 * window. Not doing this optimization for now.
		 */
		//		printf("bytes_remaining %d, seq %d\n", bytes_remaining,seq);
		if(bytes_remaining){
			wait_for_acks_ret = wait_for_acks(qp, send_comp_channel, bytes_remaining, send_cq, srv_state);
			//if(wait_for_acks_ret)
			//	return srv_state->current_psn;
		}
		//printf("congestion window size %d\n", srv_state->cwnd_size);
	}

	return wait_for_acks_ret;

}
int wait_for_acks(struct ibv_qp *qp, struct ibv_comp_channel *send_comp_channel, uint32_t bytes_remaining, struct ibv_cq *send_cq, rudp_srv_state_t *srv_state)
{
	uint8_t timeout, persist_mode; // num_extra_free;
	uint64_t retransmit = 0;
	/* Reset state parameters */
	timeout = 0;
	persist_mode = 0;
	/* Initialize the retransmission count to 0 */
	//rtt_newpack(&rttinfo);

	int  num_rcvd_acks, ss_thresh; // idx

send_again:

	/* Did we timeout */
	if (timeout == 1) {
		/* Are we in persist mode */
		if (persist_mode) {
			/* Send a probe message */
			printf("rudp_send: sending window probe message\n");
			//TODO 
			//rudp_send_ctrl_packet(fd, MSG_TYPE_WINDOW_PROBE);
		} else {
			/* Retransmit the last packet sent */

			retransmit = retransmit_first_unacked_packet(qp, srv_state);

			//	retransmit_unacked_packets(qp, &retransmit,srv_state);
			/* 
			 * We have to reduce congestion window to ss_thresh. Compute
			 * the new ss_thresh. Also update the window parameters and
			 * buffer pointer.
			 */
			//print_cwnd();
			if (srv_state->cwnd_bytes > SEG_MESSAGE_SIZE) {
				ss_thresh = srv_state->cwnd_bytes / 2;
				update_cwnd_after_timeout(1024, ss_thresh, 
						retransmit, srv_state);
			}

		}

		timeout = 0;
	}

	/* Start the timer */
	if (persist_mode) {
		rudp_start_timer(RUDP_PERSIST_TIMER_INTERVAL * USEC_IN_SEC);
	} else {
		//rudp_start_timer(rtt_start(&rttinfo));
	}

	/*	if (sigsetjmp(jmpbuf, 1) != 0) {
		if (rtt_timeout(&rttinfo) < 0) {
		printf("rudp_send: no response from peer. giving up.\n");
		rttinit = 0;
		errno = ETIMEDOUT;
		return -1;
		}
		printf("rudp_send: request timed out. retransmitting..\n");
		timeout = 1;
		goto send_again;
		}*/

	/* Initialize the receive parameters for getting the acknowledgement */
	struct ibv_cq *cq;
	void *ev_ctx;

	//print_cwnd();
	//		int psn=get_current_my_psn(qp, 1);
	//		if(psn != -1)
	//		srv_state->current_psn = psn;
	int polling_size = MAX_SLOTS_NUM;
	struct ibv_exp_wc wc[polling_size];
	int i=0;
	/*for(i=0;i<MAX_SLOTS_NUM;i++)
	  memset(&wc[i], 0, sizeof(wc[0]));
	 */
	int ack_i=0;
	int k=0;
	//	do{
	if(srv_state->event == 1){
		TEST_NZ(ibv_get_cq_event(send_comp_channel, &cq, &ev_ctx));
		//	printf("***************************************\n");
		ibv_ack_cq_events(cq, polling_size);
		TEST_NZ(ibv_req_notify_cq(cq, 0));
		k = ibv_exp_poll_cq(cq, polling_size, wc, sizeof(wc[0]));
	}else{
		do{
			k = ibv_exp_poll_cq(send_cq, polling_size, wc, sizeof(wc[0]));

		}while(k<=0);	

	}
	//srv_state->cwnd_free += k;	

	//	printf("k= %d, cwnd_size %"PRIu64", current_seq %d, thread index %d\n", k, srv_state->cwnd_bytes, srv_state->current_seq,srv_state->index);

	if(retransmit){
		fprintf(stderr,"k= %d, cwnd_bytes %d, current_seq %d\n", k, srv_state->cwnd_bytes, srv_state->current_seq);
	}
	for(ack_i=0; ack_i< k; ack_i++){
		if (wc[ack_i].status != IBV_WC_SUCCESS){
			fprintf(stderr,"index %d, wrong wc->status: %s, opcode %s\n", 
					srv_state->index, ibv_wc_status_str(wc[ack_i].status), ibv_wc_opcode_str(wc[ack_i].exp_opcode));
			struct hdr *recvheader = (struct hdr * )(uintptr_t)wc[ack_i].wr_id;
			fprintf(stderr,"seq %d, k = %d, %d\n", recvheader->seq,k, srv_state->free_slots);
			//		if(retransmit)


			uint32_t psn =  ctrl_message_send(srv_state->sockfd, "RECON",1, 0);
			reconnect_qp(qp,psn,
					srv_state->remote_qp_attribute, srv_state->send_transport, srv_state->hw_timeout, srv_state->sl);
			/*reconnect_qp(qp,srv_state->current_psn,
			  srv_state->remote_qp_attribute, srv_state->send_transport);
			 */
			timeout = 1;
			if(ack_i > 0)
				ack_i = ack_i -1;
			else
				goto send_again;	
		}
		if(srv_state->expected_ack == 0 && srv_state->send_transport == RC_TRANSPORT){
			uint32_t base_rtt;
			if(wc[ack_i].exp_wc_flags & IBV_EXP_WC_WITH_TIMESTAMP){
				base_rtt = wc[ack_i].timestamp- wc[ack_i].wr_id;
			}
			tcp_timely_pkts_acked(srv_state,base_rtt/NIC_CLOCK_MHZ);
			srv_state->prev_timestamp = wc[ack_i].timestamp;
			srv_state->prev_rtt = base_rtt/NIC_CLOCK_MHZ;
			//	printf("opcode = %s, basertt %"PRIu32"\n", ibv_wc_opcode_str(wc[ack_i].exp_opcode), base_rtt/NIC_CLOCK_MHZ);
			srv_state->expected_ack = 1;
			continue;
		}else if(srv_state->expected_ack == 0){

			srv_state->expected_ack = 1;
		}
		//printf("byte_len %"PRIu32"\n", wc[ack_i].byte_len);
		struct hdr *recvheader = (struct hdr * )(uintptr_t)wc[ack_i].wr_id;
		/*if( recvheader->end ){
		  NODE *pN = (NODE*) malloc(sizeof (NODE));
		  pN ->wc = &wc[ack_i];
		//if(!Enqueue(srv_state->usr_send_cq, pN))
		//	die("queue is full | item or queue is Null");
		}*/
		uint32_t acked_count= recvheader->seq;
		uint32_t byte_len = wc[ack_i].byte_len;
		//only for READ
		if(srv_state->opcode == READ_OPERATOR)
			srv_state->cp_bytes += byte_len;
		if(recvheader->nic_ts != 0x7fff || ack_i == k-1 || timeout || retransmit){

			/* Check for duplicate acknowledgements */
			if(  acked_count >= srv_state->expected_ack){
				/* Stop the timer */
				//rudp_stop_timer();
				/*
				 * We got a valid acknowledgement. Update the window parameters. Be
				 * sure to block SIGALRM so that we don't end up in an inconsistent
				 * state.
				 */
				if(retransmit)
					fprintf(stdout,"acked_count %d, expected_ack %d, ack_i %d, %"PRIu64",%"PRIu64"\n", 
							acked_count, srv_state->expected_ack, ack_i, srv_state->cp_bytes,srv_state->send_bytes);
				num_rcvd_acks = acked_count - srv_state->expected_ack + 1;
				int orig_start = rudp_handle_ack(acked_count, num_rcvd_acks, 
						&srv_state->pending_acks, srv_state);
				//		fprintf(stderr,"acked_count %d, expected_ack %d, ack_i %d, %"PRIu64",%"PRIu64"\n", 
				//				acked_count, srv_state->expected_ack, ack_i, srv_state->cp_bytes,srv_state->send_bytes);

				//printf("index %d, acked_count %d, acked_i %d, current_seq %d\n", srv_state->index, acked_count, ack_i, srv_state->current_seq);

				if(recvheader->nic_ts != 0x7fff ){
					if(srv_state->send_transport == RC_TRANSPORT){
						uint64_t cur_nic_ts	= 0;
						if(wc[ack_i].exp_wc_flags & IBV_EXP_WC_WITH_TIMESTAMP){
							cur_nic_ts = (wc[ack_i].timestamp- srv_state->basic_clock)%MAX_32BIT_NUM;
						}
						if(cur_nic_ts < recvheader->nic_ts)
							cur_nic_ts = cur_nic_ts+ MAX_32BIT_NUM;
						uint64_t sample_rtt1 = (cur_nic_ts-recvheader->nic_ts)/NIC_CLOCK_MHZ;
						uint64_t sample_rtt2 = (wc[ack_i].timestamp - srv_state->prev_timestamp + srv_state->prev_rtt)/NIC_CLOCK_MHZ; 		
						uint64_t sample_rtt = GET_MIN(sample_rtt1, sample_rtt2);
						srv_state->prev_timestamp = wc[ack_i].timestamp;
						srv_state->hw_timeout = sample_rtt1;
						uint64_t current_rate = (uint64_t)(srv_state->conf_rate_limiter*1000);
						//if((srv_state->cp_bytes%(1024*1024))==0 && srv_state->cp_bytes <= 10*1024*1024){
						//printf("index %d, sample_rtt, %"PRIu32", bytes*8/current_rate %"PRIu64", \n", srv_state->index, sample_rtt, byte_len*8/current_rate);
						//printf("byte_len %"PRIu32",current_rate %"PRIu64"\n", byte_len, current_rate);
						//}
						if(sample_rtt < byte_len*8/current_rate){
							srv_state->prev_rtt = ((struct Timely *)srv_state->cong)->baseRTT;
							sample_rtt = srv_state->prev_rtt;
						}else{
							sample_rtt = sample_rtt - byte_len*8/current_rate;
							if(sample_rtt < MINIMUM_BASIC_RTT){
								srv_state->prev_rtt = ((struct Timely *)srv_state->cong)->baseRTT;
								sample_rtt = srv_state->prev_rtt;
							}else{
								srv_state->prev_rtt = sample_rtt;
							}	
						}
						if(sample_rtt == 0x7fffffff){
							sample_rtt = MINIMUM_BASIC_RTT;
							srv_state->prev_rtt = sample_rtt;
						}	
						tcp_timely_pkts_acked(srv_state,sample_rtt);
						uint64_t rcvd_acked_bytes = srv_state->cp_bytes - srv_state->vegas_previous_ack;
						//	printf("acked %d, rcvd_acked_bytes %d\n", srv_state->cp_bytes, rcvd_acked_bytes);
						tcp_timely_cong_avoid(srv_state, srv_state->cp_bytes, rcvd_acked_bytes);
						srv_state->vegas_previous_ack = srv_state->cp_bytes;
						//#endif

					}else if(srv_state->send_transport == UC_TRANSPORT){
						struct ibv_exp_wc imm_wc;
						uint64_t uc_nic_ts = 0;
						int immack=0;
						struct ibv_cq *rv_cq;
						void *ev_ctx2;
						//struct timespec start, end;  
						//long long unsigned diff;
						//clock_gettime(CLOCK_MONOTONIC, &start);
							
						immack = ibv_exp_poll_cq(srv_state->recv_cq, 1, &imm_wc, sizeof(imm_wc));
						//printf("immack %d\n", immack);
						if(immack == 0){
							rudp_start_timer(100);
							if (sigsetjmp(jmpbuf, 1) != 0) {
								fprintf(stderr, "timeout of the event mode\n");
								immack = ibv_exp_poll_cq(srv_state->recv_cq, 1, &imm_wc, sizeof(imm_wc));
								rudp_stop_timer();  
								goto skip_polling ;
							}

							if(srv_state->event == 1){
								TEST_NZ(ibv_get_cq_event(srv_state->recv_channel, &rv_cq, &ev_ctx2));
								ibv_ack_cq_events(rv_cq, 1);
								TEST_NZ(ibv_req_notify_cq(rv_cq, 0));
								immack = ibv_exp_poll_cq(srv_state->recv_cq, 1, &imm_wc, sizeof(imm_wc));
							}else{
								do{
									immack = ibv_exp_poll_cq(srv_state->recv_cq, 1, &imm_wc, sizeof(imm_wc));

								}while(immack<=0);	

							}

							rudp_stop_timer();
						}	
skip_polling:
						if(immack && imm_wc.byte_len==0 ){
							if(imm_wc.exp_wc_flags & IBV_EXP_WC_WITH_TIMESTAMP){
								uc_nic_ts = (imm_wc.timestamp - srv_state->basic_clock)%MAX_32BIT_NUM;
							}
							post_receive(qp, srv_state->recv_region, srv_state->recv_mr);
							uint64_t immdata= ntohl(imm_wc.imm_data);
							int t_queue_index = immdata&0xf;
							uint32_t t_extra = (immdata&0xfffffff0)>>4;
							if(uc_nic_ts < srv_state->t_enqueue[t_queue_index])
								uc_nic_ts = uc_nic_ts + MAX_32BIT_NUM;

							uint64_t sample_rtt1 = (uc_nic_ts-srv_state->t_enqueue[t_queue_index])/NIC_CLOCK_MHZ;
							srv_state->prev_rtt = sample_rtt1;
							uint64_t sample_rtt2 = (imm_wc.timestamp - srv_state->prev_timestamp)/NIC_CLOCK_MHZ; 		
							uint64_t imm_sample_rtt = GET_MIN(sample_rtt1, sample_rtt2);
							//printf("sample_rtt1, %"PRIu64", sample_rtt2 %"PRIu64", %"PRIu32" \n",  sample_rtt1, sample_rtt2, t_extra);	   

							uint64_t rcvd_acked_bytes = srv_state->cp_bytes - srv_state->vegas_previous_ack;
							uint64_t current_rate = (uint64_t)(srv_state->conf_rate_limiter*1000);
							t_extra = t_extra + rcvd_acked_bytes*8/current_rate;	
							if(imm_sample_rtt < t_extra){

								imm_sample_rtt = ((struct Timely *)srv_state->cong)->baseRTT;
							}else{
								imm_sample_rtt = imm_sample_rtt - t_extra;
								if(imm_sample_rtt < MINIMUM_BASIC_RTT)
									imm_sample_rtt = ((struct Timely *)srv_state->cong)->baseRTT;
							}
							if(imm_sample_rtt == 0x7fffffff){
								imm_sample_rtt = MINIMUM_BASIC_RTT;
							}

							srv_state->prev_timestamp = imm_wc.timestamp;
							srv_state->rtt_samples[srv_state->cwnd_update_cnt] = imm_sample_rtt;	
							//uint32_t imm_sample_rtt = (ntohl(imm_wc.imm_data))/NIC_CLOCK_MHZ;
							//	printf("imm_sample_rtt, %"PRIu32", baseRTT %"PRIu32" \n",  imm_sample_rtt, ((struct Timely *)srv_state->cong)->baseRTT);	
							//printf("byte_len %d, immack %d, free_slots %d\n", imm_wc.byte_len, immack, srv_state->free_slots);
							tcp_timely_pkts_acked(srv_state, imm_sample_rtt);

							//printf("byte_len %d, immack %d, free_slots %d\n", imm_wc.byte_len, immack, srv_state->free_slots);									tcp_timely_cong_avoid(srv_state, srv_state->cp_bytes, rcvd_acked_bytes);
							srv_state->vegas_previous_ack = srv_state->cp_bytes;
						}
					}
					srv_state->cwnd_update_cnt ++;
					//#if ENABLE_RL
					if(srv_state->rate_limit_mode == 1){
						if(srv_state->cwnd_update_cnt%32 == 0
								|| srv_state->rate_limiter < (srv_state->conf_rate_limiter)){										/*printf("index %d, rate %f, prev_rate %f\n", srv_state->index,
																									  srv_state->max_rate_limiter*1.0, srv_state->conf_rate_limiter*1.0);
																									 */
							//srv_state->conf_rate_limiter = srv_state->rate_limiter;
							double target_rate = GET_MIN(srv_state->conf_rate_limiter + RL_PROBE, srv_state->rate_limiter);
							if(srv_state->conf_rate_limiter == target_rate || 
									(srv_state->conf_rate_limiter >=10.0 && target_rate >=10.0) ){	
								// if the configured rate is the same as target rate 
								// or both rate_limiter are larger than 10.0.
							}else{		
								srv_state->conf_rate_limiter = target_rate;
								if(srv_state->conf_rate_limiter*1.0 >= 10.0){
									if(srv_state->opcode == READ_OPERATOR)
										post_rate(0, srv_state, 1);
									else
										run_dcbnetlink(srv_state->dcbstate, 0);
								}else{
									//double rate = (int)ceil(srv_state->conf_rate_limiter*1.0);
									double rate = srv_state->conf_rate_limiter*1.0;
									uint64_t r_rate = (uint64_t)(rate*1000*1000);
									//printf("testing3 %"PRIu64"\n", r_rate);
									if(srv_state->opcode == READ_OPERATOR) 
										post_rate(r_rate, srv_state, 1);
									else{
										run_dcbnetlink(srv_state->dcbstate, r_rate);	
									}
								}
								//		printf("index %d, conf_rate %f\n", srv_state->index,
								//				srv_state->conf_rate_limiter);
								//srv_state->max_rate_limiter = 0.1;
							}
						}
					}

				}
				//	printf("index %d, acked_count %d, acked_i %d, current_seq %d\n", srv_state->index, acked_count, ack_i, srv_state->current_seq);
				/* Update the congestion window parameters */

				update_cwnd_after_valid_ack(srv_state, orig_start, num_rcvd_acks);
				//printf("cwnd_size %"PRIu64",free_byts %d, free_slots %d\n", srv_state->cwnd_bytes, srv_state->free_bytes, srv_state->free_slots);
				if( retransmit ){
					/*	
					 * fill the congestion window
					 */
					retransmit -= byte_len;
					if(srv_state->send_bytes > (srv_state->cp_bytes +retransmit) ){
						retransmit_unacked_packets(qp, &retransmit,srv_state);
					}else{
						fprintf(stderr, "stop retransmitting, bytes_remaining %"PRIu32"\n", bytes_remaining);
						retransmit = 0;
						return -1;	

					}

				}
			}else{
				fprintf(stderr,"received acked_count %d < expected_ack %d, retransmit %d\n", 
						acked_count, srv_state->expected_ack, recvheader->retransmit);
				//	die("");
			}
			if(timeout)
				goto send_again;	


		}
	}// end of multiple ack count
	if(retransmit)
		goto send_again;
	//printf("end of sending a packet\n");
	return 0;
}

/****************************************************************************/
/*                       Client related routines                            */
/****************************************************************************/
void *watchdog_server(void *ctx){

	int sockfd, newsockfd; // portno;
	socklen_t clilen;
	rudp_cli_state_t *cli_state = (rudp_cli_state_t *)ctx;
	int portno = cli_state->inter_port_no;
	//char buffer[S_QPA];
	struct sockaddr_in serv_addr, cli_addr;
	int n;
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0) 
		die("ERROR opening socket");
	bzero((char *) &serv_addr, sizeof(serv_addr));
	//portno = atoi(argv[1]);
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = INADDR_ANY;
	serv_addr.sin_port = htons(portno);
	if (bind(sockfd, (struct sockaddr *) &serv_addr,
				sizeof(serv_addr)) < 0) 
		die("ERROR on binding @ watchdog_server ");
	printf("watch dog server, portno %d\n", portno);
	listen(sockfd,5);

	clilen = sizeof(cli_addr);
	newsockfd = accept(sockfd, 
			(struct sockaddr *) &cli_addr, 
			&clilen);
	if (newsockfd < 0) 
		die("ERROR on accept");

	printf("receive connection \n");
	/*memset(buffer, 0, sizeof(buffer));
	  n = recv(newsockfd,buffer,sizeof(buffer),0);
	  if (n < 0) die("ERROR reading from socket");
	  uint64_t remote_clock = strtoll(buffer, NULL, 10);	
	  printf("received value %"PRIu64"\n", remote_clock);

	// send own clock
	memset(buffer, 0, sizeof(buffer));
	sprintf(buffer,"%llu" , basic_clock);
	n = send(newsockfd,buffer,sizeof(buffer),0);
	if (n < 0) 
	die("ERROR writing to socket");*/
	//	printf("final clock %"PRIu64"\n", cli_state->basic_clock);

	char buffer[128];
	//basic_clock  = GET_MIN(basic_clock,v); 
	while(1){
		memset(buffer, 0, sizeof(buffer));
		n = recv(newsockfd,buffer,sizeof(buffer),0);
		if (n < 0) die("ERROR reading from socket");
		else if(n==0) return NULL;
		printf("received value %s\n", buffer);
		if(!strcmp(buffer,"RECON")){
			printf("RECONNECT\n");
			uint32_t rq_psn = get_current_my_psn(cli_state->qp, 0);
			//memset(buffer, 0, sizeof(buffer));
			//sprintf(buffer,"%s","OK");
			uint32_t psn = htonl(rq_psn);
			n = send(newsockfd, &psn,sizeof(psn),0);
			if (n < 0) die("ERROR sending to socket");


		}	
	}
}


/*
 * rudp_cli_init
 *
 * Initializes the RUDP layer for the client
 */
int post_ack(uint64_t rcv_t, uint32_t imm_data, rudp_cli_state_t *cli_state){
	struct ibv_sge sg;
	struct ibv_send_wr wr;
	struct ibv_send_wr *bad_wr;

	memset(&sg, 0, sizeof(sg));
	sg.addr	  = (uintptr_t)cli_state->send_region;
	sg.length = 0;
	sg.lkey	  = cli_state->send_mr->lkey;

	memset(&wr, 0, sizeof(wr));
	wr.wr_id      = 0;
	wr.sg_list    = &sg;
	wr.num_sge    = 1;
	wr.opcode     = IBV_WR_RDMA_WRITE_WITH_IMM;

	wr.send_flags = IBV_SEND_SIGNALED;
	uint64_t hw_time = query_hardware_time(cli_state->ib_ctx)-cli_state->basic_clock;	
	uint32_t elapsed = (hw_time-rcv_t)/NIC_CLOCK_MHZ;
	//printf("software overhead %"PRIu32"\n", hw_time-rcv_t);
	//	uint32_t d = (elapsed+imm_data)%MAX_32BIT_NUM;
	uint32_t d = elapsed;
	d = d<<4;
	d = d + imm_data;
	wr.imm_data   = htonl(d);
	wr.wr.rdma.remote_addr = (uintptr_t )cli_state->peer_mr->addr;		
	wr.wr.rdma.rkey        = cli_state->peer_mr->rkey;

	if (ibv_post_send(cli_state->qp, &wr, &bad_wr)) {
		fprintf(stderr, "Error, ibv_post_send() failed\n");
		return -1;
	}	
}
void poll_ack_cq(struct ibv_cq *cq, rudp_cli_state_t *cli_state){
	struct ibv_exp_wc wc;
	int k = ibv_exp_poll_cq(cq, 1, &wc, sizeof(wc));
	//if(k){
	//TO DO: to dequeue the work requests
	//}
}
void * poll_imm_cq(struct ibv_comp_channel *comp_channel, rudp_cli_state_t *cli_state)
{
	struct ibv_cq *cq;
	struct ibv_exp_wc wc;
	void *ev_ctx;

	int k=0;
	uint64_t uc_nic_ts;
	while(1){	
		TEST_NZ(ibv_get_cq_event(comp_channel, &cq, &ev_ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));
		k = ibv_exp_poll_cq(cq, 1, &wc, sizeof(wc));
		while(k){

			uint32_t imm_data = ntohl(wc.imm_data);
			if(cli_state->send_transport){//UC AND UD
				if(wc.exp_wc_flags & IBV_EXP_WC_WITH_TIMESTAMP){
					uc_nic_ts = wc.timestamp - cli_state->basic_clock;
				}
				//if(imm_data ==0xffff)
				//	printf ("%.1024s", cli_state->recv_region);
				//printf("clock [ %"PRIu32", %"PRIu32" ]\n", uc_nic_ts, imm_data);
				poll_ack_cq(cli_state->send_cq, cli_state);	
				post_ack(uc_nic_ts,imm_data, cli_state);
			}else{
				//FIXME if and only if this is a READ but get immediate data
				run_dcbnetlink(cli_state->dcbstate, imm_data);
				//printf("rate %"PRIu32"\n",imm_data );
			}	
			post_receive(cli_state->qp, cli_state->recv_region, cli_state->recv_mr);	
			k = ibv_exp_poll_cq(cq, 1, &wc, sizeof(wc));

		}
	}
}

	int
rudp_cli_init (rudp_cli_state_t *state, rudp_cli_state_t *cli_state)
{
	int i;

	cli_state->local_qp_attribute = state->local_qp_attribute;
	cli_state->remote_qp_attribute = state->remote_qp_attribute;
	cli_state->send_transport = state->send_transport;
	cli_state->opcode = state->opcode;
	cli_state->ib_ctx = state->ib_ctx;
	cli_state->qp = state->qp;
	cli_state->recv_channel = state->recv_channel;
	cli_state->send_channel = state->send_channel;
	cli_state->send_cq = state->send_cq;
	cli_state->inter_port_no = state->inter_port_no;

	cli_state->peer_mr = state->peer_mr;
	cli_state->recv_region = state->recv_region;
	cli_state->recv_mr = state->recv_mr;
	cli_state->send_region = state->send_region;
	cli_state->send_mr = state->send_mr;
	for(i=0;i<MAX_SEND_WR/2;i++)
		post_receive(cli_state->qp,cli_state->recv_region, cli_state->recv_mr);	

	pthread_t watchdog;
	int portno = cli_state->inter_port_no;
	cli_state->basic_clock = query_hardware_time(cli_state->ib_ctx);
	//printf("server clock %"PRIu64"\n",cli_state->basic_clock );	

	if(cli_state->opcode == READ_OPERATOR){
		if (init_dcbnetlink(&cli_state->dcbstate,IFACE) < 0) {
			fprintf(stderr, "Error initializing DCB NETLINK state\n");
			return -1; 
		}
		run_dcbnetlink(cli_state->dcbstate, 0);
	}
	if(cli_state->send_transport == RC_TRANSPORT)	
		TEST_NZ(pthread_create(&watchdog, NULL, watchdog_server, cli_state));
	poll_imm_cq(cli_state->recv_channel,cli_state);
	/*	cli_state->advw_size = state->advw_size;
		cli_state->random_seed  = state->random_seed;
		cli_state->data_loss_prob = state->data_loss_prob;
		cli_state->recv_rate = state->recv_rate;
	 */
	/* Initialize the receive window parameters */
	/*	cli_state->advw_start = 0;
		cli_state->advw_end = 0;
		cli_state->expected_seq = 0;
		cli_state->advw_free = cli_state->advw_size;
		pthread_mutex_init(&cli_state->advw_lock, NULL);
	 */
	/* Allocate and initialize the receive window */
	/*	cli_state->advw = 
		(rudp_payload_t *)malloc(cli_state->advw_size * sizeof(rudp_payload_t));
		if (!cli_state->advw) {
		return -1;
		}
		bzero(cli_state->advw, cli_state->advw_size * sizeof(rudp_payload_t));

		for (i = 0; i < cli_state->advw_size; i++) {
		cli_state->advw[i].data = (char *)malloc(RUDP_PAYLOAD_SIZE);
		if (!cli_state->advw[i].data) {
		return -1;
		}
		bzero(cli_state->advw[i].data, RUDP_PAYLOAD_SIZE);
		}
	 */
	/* Initialize the random seed */
	//	srand(cli_state->random_seed);

	/* Register for SIGALRM */
	//	signal(SIGALRM, sigalarm_handler);

	return 0;
}

/*
 * rudp_cli_destroy
 *
 * Cleans up the resources allocated to the RUDP layer on the client
 */
	int
rudp_cli_destroy (rudp_cli_state_t *cli_state)
{
	int i;

	/* Free the allocated resources */
	if (cli_state) {
		/*	if (cli_state->advw) {
			for (i = 0; i < cli_state->advw_size; i++) {
			if (cli_state->advw[i].data) {
			free(cli_state->advw[i].data);
			}
			}
			}
			free(cli_state->advw);*/
		free(cli_state);
	}
	return 0;
}



/* End of File */
