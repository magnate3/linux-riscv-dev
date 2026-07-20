#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>
//#include <infiniband/verbs_exp.h>



#include "common.h"

void get_qp_info(struct context *s_ctx, struct connection *conn){
	union ibv_gid my_gid= get_gid(s_ctx->ctx);

	conn->local_qp_attr = (struct qp_attr *)malloc(sizeof(struct qp_attr));
	conn->local_qp_attr->gid_global_interface_id = my_gid.global.interface_id;
	conn->local_qp_attr->gid_global_subnet_prefix = my_gid.global.subnet_prefix;
	conn->local_qp_attr->lid                     = get_local_lid(s_ctx->ctx);
	conn->local_qp_attr->qpn                     = conn->qp->qp_num;
	conn->local_qp_attr->psn                     = lrand48() & 0xffffff;

}

void check_timestamp_enable(struct ibv_context *ctx){
	struct ibv_exp_device_attr attr;
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

	printf("after core_clock\n");
	struct ibv_exp_values queried_values;
	memset(&queried_values, 0, sizeof(queried_values));
	queried_values.comp_mask = IBV_EXP_VALUES_HW_CLOCK;
	int ret = ibv_exp_query_values(ctx, IBV_EXP_VALUES_HW_CLOCK, &queried_values);
	if (!ret && (queried_values.comp_mask & IBV_EXP_VALUES_HW_CLOCK))
		printf("hw_clock = %llu\n", (long long unsigned)queried_values.hwclock);


	}
struct context * init_ctx(struct ibv_device *ib_dev, struct context *s_ctx ){
	struct ibv_context *ctx = ibv_open_device(ib_dev);
	struct ibv_device_attr device_attr;
	int rc;

/*	rc = ibv_query_device(ctx, &device_attr);
	if (rc) {
		fprintf(stderr, "Error, failed to query the device attributes\n");
		return NULL; 
	}
	printf("max_qp %d, max_cqe %d, max_qp_wr %d\n",device_attr.max_qp, device_attr.max_cqe, device_attr.max_qp_wr);*/
	s_ctx = build_context(ctx,s_ctx);
	s_ctx->nr_conns = 0;
	//check_timestamp_enable(s_ctx->ctx);
	return s_ctx;
}


// Transition connected QP indexed qp_i through RTR and RTS stages
int connect_ctx(struct connection *conn, int my_psn, struct qp_attr *dest, int priority,int isRone)
{
	//printf("sl %d\n", priority);
	struct ibv_qp_attr conn_attr = {
		.qp_state= IBV_QPS_RTR,
		.path_mtu= IBV_MTU_SIZE,
		.dest_qp_num= dest->qpn,
		.rq_psn= dest->psn,
		.ah_attr= {
			.is_global= 1,
			.dlid= 0,
			.sl= priority,
			.src_path_bits= 0,
			.port_num= IB_PHYS_PORT
		}
	};

	conn_attr.ah_attr.grh.dgid.global.interface_id = 
		dest->gid_global_interface_id;
	conn_attr.ah_attr.grh.dgid.global.subnet_prefix = 
		dest->gid_global_subnet_prefix;

	conn_attr.ah_attr.grh.sgid_index = GID_INDEX;
	conn_attr.ah_attr.grh.hop_limit = 1;

	int rtr_flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
		| IBV_QP_RQ_PSN;
	if(!conn->send_transport) {
		conn_attr.max_dest_rd_atomic = 16;
		conn_attr.min_rnr_timer = 12;
		rtr_flags |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
	}
	if (ibv_modify_qp(conn->qp, &conn_attr, rtr_flags)) {
		fprintf(stderr, "Failed to modify QP to RTR\n");
		return 1;
	}

	memset(&conn_attr, 0, sizeof(conn_attr));
	conn_attr.qp_state    = IBV_QPS_RTS;
	conn_attr.sq_psn    = my_psn;
	int rts_flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
	if(!conn->send_transport) {
		if(isRone == 1){
			conn_attr.timeout = 10;
			conn_attr.retry_cnt = 1;
			conn_attr.rnr_retry = 1;
		}else{
			conn_attr.timeout = 14;
			conn_attr.retry_cnt = 7;
			conn_attr.rnr_retry = 7;

		}
		conn_attr.max_rd_atomic = 16;
		rts_flags |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
			IBV_QP_MAX_QP_RD_ATOMIC;
	}
	if (ibv_modify_qp(conn->qp, &conn_attr, rts_flags)) {
		fprintf(stderr, "Failed to modify QP to RTS\n");
		return 1;
	}

	return 0;
}

void create_ah(struct connection *ctx,struct ibv_pd *pd ){
	struct ibv_ah_attr ah_attr = {
		.is_global= 1,
		.dlid= 0,
		.sl= SL,
		.src_path_bits= 0,
		.port_num= IB_PHYS_PORT
	};


	ah_attr.grh.dgid.global.interface_id = 
		ctx->remote_qp_attr->gid_global_interface_id;
	ah_attr.grh.dgid.global.subnet_prefix = 
		ctx->remote_qp_attr->gid_global_subnet_prefix;

	ah_attr.grh.sgid_index = GID_INDEX;
	ah_attr.grh.hop_limit = 1;

	ctx->ah = ibv_create_ah(pd, &ah_attr);
	if(!ctx->ah){
		printf("Failed to create ah\n");
		exit(-1);
	}

}
int modify_dgram_qp_to_rts(struct connection *ctx, int localpsn)
{

	struct ibv_qp_attr dgram_attr = {
		.qp_state= IBV_QPS_RTR,
	};

	if (ibv_modify_qp(ctx->qp, &dgram_attr, IBV_QP_STATE)) {
		fprintf(stderr, "Failed to modify dgram QP to RTR\n");
		return 1;
	}

	dgram_attr.qp_state= IBV_QPS_RTS;
	dgram_attr.sq_psn=localpsn;

	if(ibv_modify_qp(ctx->qp, 
				&dgram_attr, IBV_QP_STATE|IBV_QP_SQ_PSN)) {
		fprintf(stderr, "Failed to modify dgram QP to RTS\n");
		return 1;
	}

	return 0;
}


union ibv_gid get_gid(struct ibv_context *context)
{
	union ibv_gid ret_gid;
	ibv_query_gid(context, IB_PHYS_PORT, GID_INDEX, &ret_gid);

	/*	fprintf(stderr, "GID: Interface id = %lld subnet prefix = %lld\n", 
		(long long) ret_gid.global.interface_id, 
		(long long) ret_gid.global.subnet_prefix);
	 */
	return ret_gid;
}

uint16_t get_local_lid(struct ibv_context *context)
{
	struct ibv_port_attr attr;

	if (ibv_query_port(context, IB_PHYS_PORT, &attr))
		return 0;

	return attr.lid;
}

void print_qp_attr(struct qp_attr *dest)
{
	fflush(stdout);
	fprintf(stderr, "\t%d %d %d\n", dest->lid, dest->qpn, dest->psn);
}




void modify_qp_to_init(struct connection *ctx)
{

	struct ibv_qp_attr conn_attr = {
		.qp_state= IBV_QPS_INIT,
		.pkey_index= 0,
		.port_num= IB_PHYS_PORT,
		.qp_access_flags= IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ
	};
	if (ibv_modify_qp(ctx->qp, &conn_attr,
				IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
		fprintf(stderr, "Failed to modify conn. QP to INIT\n");
		return;
	}

}


void ud_modify_qp_to_init(struct connection *ctx){
	struct ibv_qp_attr dgram_attr = {
		.qp_state= IBV_QPS_INIT,
		.pkey_index= 0,
		.port_num= IB_PHYS_PORT,
		.qkey = 0x11111111
	};


	if (ibv_modify_qp(ctx->qp, &dgram_attr,
				IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
		fprintf(stderr, "Failed to modify dgram. QP to INIT\n");
		return;
	}

}
struct context * build_context(struct ibv_context *verbs,struct context *s_ctx)
{
	if (s_ctx) {
		if (s_ctx->ctx != verbs)
			die("cannot handle events in more than one context.");
		return s_ctx;
	}

	s_ctx = (struct context *)malloc(sizeof(struct context));
	s_ctx->ctx = verbs;

	TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));

	struct ibv_exp_cq_init_attr *cq_attr;
	cq_attr = (struct ibv_exp_cq_init_attr *)malloc(sizeof(struct ibv_exp_cq_init_attr));
	cq_attr->flags = IBV_EXP_CQ_TIMESTAMP;
	cq_attr->comp_mask = IBV_EXP_CQ_INIT_ATTR_FLAGS;

	TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));
	TEST_Z(s_ctx->cq = ibv_exp_create_cq(s_ctx->ctx, MAX_SEND_WR+1, NULL, s_ctx->comp_channel, 0, cq_attr)); /* cqe=10 is arbitrary */
	TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));

	s_ctx->usr_recv_cq = ConstructQueue(MAX_SEND_WR+1); 	
	
	TEST_Z(s_ctx->send_comp_channel = ibv_create_comp_channel(s_ctx->ctx));
	TEST_Z(s_ctx->send_cq = ibv_exp_create_cq(s_ctx->ctx, MAX_SEND_WR+1, NULL, s_ctx->send_comp_channel, 0, cq_attr)); /* cqe=10 is arbitrary */
	TEST_NZ(ibv_req_notify_cq(s_ctx->send_cq, 0));

	s_ctx->usr_send_cq = ConstructQueue(MAX_SEND_WR+1); 	
	return s_ctx;
}

void build_qp_attr(struct ibv_qp_init_attr *qp_attr, struct connection *conn, struct context *s_ctx)
{
	memset(qp_attr, 0, sizeof(*qp_attr));

	qp_attr->send_cq = s_ctx->send_cq;
	qp_attr->recv_cq = s_ctx->cq;
	switch(conn->send_transport){
		case 0: qp_attr->qp_type = IBV_QPT_RC; break;
		case 1: qp_attr->qp_type = IBV_QPT_UC; break;
		case 2: qp_attr->qp_type = IBV_QPT_UD; break;
		default: printf("WRONG transport type!!\n"); break;
	}

	qp_attr->sq_sig_all = conn->signal_all;

	qp_attr->cap.max_inline_data = MAX_INLINE_SIZE;
	qp_attr->cap.max_send_wr = MAX_SEND_WR;
	qp_attr->cap.max_recv_wr = MAX_SEND_WR;
	qp_attr->cap.max_send_sge = 1;
	qp_attr->cap.max_recv_sge = 1;
}


void post_receives_comm(struct connection *conn)
{
	struct ibv_recv_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;

	wr.wr_id = (uintptr_t)conn;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;

	sge.addr = (uintptr_t)conn->recv_region;
	sge.length = RECV_BUFFER_SIZE;
	sge.lkey = conn->recv_mr->lkey;

	TEST_NZ(ibv_post_recv(conn->qp, &wr, &bad_wr));
}

char* alloc_raw_pages(int cnt, int size)
{
  /*
   *  Don't touch the page since then allocator would not allocate the page right now.
   */
  int flag = MAP_SHARED | MAP_ANONYMOUS;
  if (size == 2*1024*1024)
    flag |= MAP_HUGETLB;
  char* ptr = mmap(NULL, (int64_t) cnt * size, PROT_READ | PROT_WRITE, flag, -1, 0);
  if (ptr == (char*) -1) {
    perror("alloc_raw_pages");
    return NULL;
  }
  return ptr;
}



void set_memory(struct connection *conn, struct context *s_ctx, int USE_WRITE, int key){
/*	int sid = shmget(key, RECV_BUFFER_SIZE, IPC_CREAT | SHM_R | SHM_W | SHM_HUGETLB);
	if (sid < 0) {
		perror("shmget");
	}
	CPE(sid < 0, "Master server request area shmget() failed\n", sid);
	
	conn->recv_region = (char *)shmat(sid, 0, 0);*/
	
	//conn->recv_region = alloc_raw_pages(2, RECV_BUFFER_SIZE/2);
	//server_req_area_mr = ibv_reg_mr(cb->pd, (char *)server_req_area, M_2, FLAGS);
	//CPE(!server_req_area_mr, "Failed to register server's request area", errno);
	conn->recv_region = (char *)malloc(RECV_BUFFER_SIZE);

	if(!conn->recv_region){
		printf("malloc recv memory fails\n");
		exit(-1);
	}
		
	memset((char *)conn->recv_region, 0, RECV_BUFFER_SIZE);
	TEST_Z(conn->recv_mr = ibv_reg_mr(
				s_ctx->pd, 
				conn->recv_region, 
				RECV_BUFFER_SIZE, 
				IBV_ACCESS_LOCAL_WRITE | ((USE_WRITE == READ_OPERATOR) ? IBV_ACCESS_REMOTE_READ : IBV_ACCESS_REMOTE_WRITE)));


/*	int sid = shmget(key+1, RECV_BUFFER_SIZE, IPC_CREAT | 0666 | SHM_HUGETLB);
	CPE(sid < 0, "Master server request area shmget() failed\n", sid);

	conn->send_region = (char *)shmat(sid, 0, 0);*/
	//conn->send_region = alloc_raw_pages(2, RECV_BUFFER_SIZE/2);
	conn->send_region = (char *)malloc(RECV_BUFFER_SIZE);
	if(!conn->send_region){
		printf("malloc send memory fails\n");
		exit(-1);
	}

	TEST_Z(conn->send_mr = ibv_reg_mr(
				s_ctx->pd, 
				conn->send_region, 
				RECV_BUFFER_SIZE, 
				IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE));
}





void * throughput_timer(void *context){
	FILE *tput;
	struct timespec timer;
	struct connection *conn = (struct connection*)context;
	int pre=0;
	tput = fopen("tput","w+");
	while(1){
		sleep(1);

		if(pre == conn->num_completions){
			fclose(tput);
			return NULL;
		}
		clock_gettime(CLOCK_MONOTONIC, &timer);
		fprintf(tput, "%d %d %llu \n",
				conn->send_message_size,  conn->num_completions, (long long unsigned)timer.tv_sec*BILLION + timer.tv_nsec);
		//fflush(stdout);
		pre = conn->num_completions;
	}
}

void per_request_latency(struct connection *conn, long long unsigned *starttimestamp, long long unsigned *endtimestamp){
	FILE *request_latency;
	request_latency = fopen("request_latency", "w+");
	if(conn->isClient){
		int i=0;
		for(i=0;i<MAX_TIMESTAMP_SIZE;i++){
			if(starttimestamp[i]==0 || endtimestamp[i]==0)
				break;
			fprintf( request_latency, "%d %llu\n", conn->send_message_size, (endtimestamp[i]-starttimestamp[i])/1000);
		}
		fclose(request_latency);
	}else{
		//server side
		printf("num_completions %d\n", conn->num_completions );
	}
}
void avg_latency(struct connection *conn){
	if(conn->isClient){
		static struct timespec start, end;
		long long unsigned diff;
		if(conn->num_sendcount == 1)
			clock_gettime(CLOCK_MONOTONIC,&start);

		if(conn->num_completions == (conn->num_requests+conn->window_size)){
			clock_gettime(CLOCK_MONOTONIC, &end);
			diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
			//diff = (end.tv_sec - start.tv_sec);
			printf("%d %d %llu\n",  conn->send_message_size, conn->num_sendcount,  diff/1000);
			fflush(stdout);

			exit(0);
		}
	}else{
		//server side
		if(conn->num_completions == (conn->num_requests+conn->window_size)){
			printf("num_completions %d\n", conn->num_completions );
			//exit(0);
		}
	}
}
void record_timestamp(int count,  long long unsigned * start_timestamp){
	if( count >= 0 && count<MAX_TIMESTAMP_SIZE){
		struct timespec timer;
		clock_gettime(CLOCK_MONOTONIC, &timer);
		long long unsigned timestamp=(long long unsigned)timer.tv_sec*BILLION + timer.tv_nsec;
		printf("count: %d, %llu\n", count, timestamp);	
		start_timestamp[count]=timestamp;
	}
}


uint64_t on_write_read(struct context *s_ctx, struct connection *conn, int sig, int USE_WRITE, int isRone){
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)conn;
	if (USE_WRITE == READ_OPERATOR){
		wr.opcode = IBV_WR_RDMA_READ;
	}
	else if(USE_WRITE == WRITE_OPERATOR){
		wr.opcode = IBV_WR_RDMA_WRITE; //IBV_WR_RDMA_WRITE 
		//wr.imm_data = htonl(0x1234);
	}else if(USE_WRITE == WRITE_IMM_OPERATOR){
		wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; //IBV_WR_RDMA_WRITE 
	}
	wr.sg_list = &sge;
	wr.num_sge = 1;
	//if(s_mode == M_READ || s_inline == 0){
	//  wr.send_flags = IBV_SEND_SIGNALED;
	// }else if (s_mode== M_WRITE && s_inline == 1)
/*	if(USE_WRITE != READ_OPERATOR && conn->send_message_size <= MAX_INLINE_SIZE)
		wr.send_flags = IBV_SEND_INLINE;*/
	if (sig ==1){
		wr.send_flags = wr.send_flags | IBV_SEND_SIGNALED;
		conn->isSignal = 1;
	}

	//int blocks = RECV_BUFFER_SIZE/conn->send_message_size;
	//reading the memory one by one per request
	//int offset = (conn->num_sendcount%blocks)*conn->send_message_size;
	//reading the same memory region between 0~blocks-1, blocks~2*blocks-1, 2*blocks~ 2*blocks.
	//int remote_offset = ((conn->num_sendcount/blocks)%26)*conn->send_message_size;
	long long unsigned offset = (long long unsigned) (conn->num_sendcount*conn->send_message_size);
	uint32_t remote_offset = offset %RECV_BUFFER_SIZE;
	if(RECV_BUFFER_SIZE -remote_offset < conn->send_message_size)
		remote_offset = 0;
	wr.wr.rdma.remote_addr = (uintptr_t)conn->peer_mr->addr+remote_offset;
	wr.wr.rdma.rkey = conn->peer_mr->rkey;

	
	if(USE_WRITE == WRITE_OPERATOR || USE_WRITE == WRITE_IMM_OPERATOR){
		//snprintf(conn->send_region, conn->send_message_size, "client side  %d", conn->num_sendcount);
		sge.addr = (uintptr_t)conn->send_region;
		sge.length = conn->send_message_size;
		sge.lkey = conn->send_mr->lkey;
	}else{
		int local_offset = offset%RECV_BUFFER_SIZE;
		if(RECV_BUFFER_SIZE - local_offset < conn->send_message_size)
			local_offset = 0;
		sge.addr = (uintptr_t)conn->recv_region+local_offset;
		sge.length = conn->send_message_size;
		sge.lkey = conn->recv_mr->lkey;
	}
	
	//return 0;
	uint64_t start_ts;
	if(isRone==1){
		rudp_send_read_write(&wr, s_ctx->send_comp_channel, conn->qp, conn->num_sendcount, s_ctx->send_cq, s_ctx->srv_state);
		return 0;
	}else{
		start_ts = query_hardware_time(s_ctx->ctx);
		TEST_NZ(ibv_post_send(conn->qp, &wr, &bad_wr));
		return start_ts;
	}
}

int on_send(void *context, int sig)
{
	struct connection *conn = (struct connection *)context;
	struct ibv_send_wr wr, *bad_wr = NULL;
	struct ibv_sge sge;
	//snprintf(conn->send_region, conn->send_message_size, "message from passive/server side with pid %d", getpid());

	//if(conn->isClient)
		//snprintf(conn->send_region, conn->send_message_size, "from client with num_completions %d", conn->num_completions);
	//	fillzero(conn->send_region, 'c', conn->send_message_size);
	//else
		//snprintf(conn->send_region, conn->send_message_size, "from server with num_completions %d", conn->num_completions);
	//	fillzero(conn->send_region, 's', conn->send_message_size);

	//printf("connected. posting send...\n");
	//printf("conn->send_message_size = %d\n", conn->send_message_size);
	//printf("conn->send_region %s\n", conn->send_region);
	memset(&wr, 0, sizeof(wr));
	wr.wr_id = (uintptr_t)conn;
	wr.opcode = IBV_WR_SEND;
	wr.sg_list = &sge;
	wr.num_sge = 1;
//	wr.send_flags = IBV_SEND_INLINE; //IBV_SEND_SIGNALED;

	if (sig == 1){
		wr.send_flags = wr.send_flags | IBV_SEND_SIGNALED;
		conn->isSignal = 1;
	}

	if(conn->send_transport == UD_TRANSPORT){
		wr.wr.ud.ah = conn->ah;
		wr.wr.ud.remote_qpn = conn->remote_qp_attr->qpn;
		wr.wr.ud.remote_qkey = 0x11111111;
	}

	sge.addr = (uintptr_t)conn->send_region;
	sge.length = conn->send_message_size;
	sge.lkey = conn->send_mr->lkey;

	ibv_post_send(conn->qp, &wr, &bad_wr);
	//printf("error code: %s\n", strerror(err));
	return 0;
}

int on_disconnect(struct context *ctx)
{
	int i;
	for(i=0;i<ctx->nr_conns;i++){
		struct connection *conn= ctx->conns[i];

		if (ibv_destroy_qp(conn->qp)) {
			fprintf(stderr, "Couldn't destroy connected QP\n");
			return 1;
		}

		//	printf("peer disconnected.2\n");
		ibv_dereg_mr(conn->send_mr);
		ibv_dereg_mr(conn->recv_mr);

		free(conn);

	}
//	printf("peer disconnected.0\n");
	if (ibv_destroy_cq(ctx->cq)) {
		fprintf(stderr, "Couldn't destroy connected RECV CQ\n");
		return 1;
	}	
//	printf("peer disconnected.1\n");
	if (ibv_destroy_cq(ctx->send_cq)) {
		fprintf(stderr, "Couldn't destroy connected RECV CQ\n");
		return 1;
	}	
	
	if (ibv_dealloc_pd(ctx->pd)) {
		fprintf(stderr, "Couldn't deallocate PD\n");
		return 1;
	}

	if(ibv_destroy_comp_channel(ctx->comp_channel)){
		fprintf(stderr, "failed to close receive event channel\n");
	}
	if(ibv_destroy_comp_channel(ctx->send_comp_channel)){
		fprintf(stderr, "failed to close send_event channel\n");
	}
	if (ibv_close_device(ctx->ctx)) {
		fprintf(stderr, "Couldn't release context\n");
		return 1;
	}
	//FIXME shall we free the hugepage region
	//free(conn->send_region);
	//free(conn->recv_region);

	free(ctx);
	//  rdma_destroy_id(id);
	return 0;
}


void fillzero(char *region, char c, int send_message_size){
	int i;
	for (i=0;i< send_message_size;i++){                                          
		region[i]= c;
	}                                                                                                                              
}
