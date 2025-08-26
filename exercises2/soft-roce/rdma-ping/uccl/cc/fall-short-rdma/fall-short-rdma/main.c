#define __USE_GNU
#define _GNU_SOURCE

#include <sched.h>


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <rdma/rdma_cma.h>


#include "common.h"
//#include "rudp.h"
#include "server.h"
#include "client.h"
#include "rdtsc.h"

#define OPERATOR READ_OPERATOR 
#define DUPLICATE_COPIES 1

#define CPUGHZ  2.0f


void reply_operation(struct connection *conn, int USE_WRITE);
void *write_cq(void *context);
void run_client(int port, char *host, struct connection *conn);
void run_server(int portno,struct connection *conn, struct ibv_pd *pd);
void on_completion(struct ibv_wc *wc, int k);
void * poll_cq(void *ctx);
void * poll_send_cq(void *ctx);
void recv_message(struct connection *conn);
int init_rone_server(struct context *s_ctx,int index);
int init_rone_client(struct context *s_ctx);


long long unsigned starttimestamp[MAX_TIMESTAMP_SIZE];
long long unsigned endtimestamp[MAX_TIMESTAMP_SIZE];
/* Globals */
server_params_t *serv_params;
/* Globals */
client_params_t *cli_params;


int send_message_size = 2048;
int num_send_request =0;
int window_size= 1;
int signaled=1;
//int cpu_id=0;
int portno=55281;
int internal_port_no = 66889;
int is_client=0;
int event_mode = 0;
int duration=0;
int num_threads=1;

#define MAX_NUM_IB_THREADS	16
struct context *multi_ctx[MAX_NUM_IB_THREADS];
pthread_t ib_threads[MAX_NUM_IB_THREADS];
char *hosts[MAX_NUM_IB_THREADS]={NULL};

static sigjmp_buf jmpbuf;
static void 
sigalarm_handler (int signo)
{
	siglongjmp(jmpbuf, 1);
}
void parseOpt(int argc, char **argv){
	int c; 
	while ((c = getopt(argc, argv, "s:m:n:c:b:h:p:e:D:M:")) != -1) {
		switch (c)
		{
			case 'm':
				send_message_size = atoi(optarg);
				if(send_message_size > RECV_BUFFER_SIZE){
					printf("send_message size is larger than receive buffer size %d\n", RECV_BUFFER_SIZE);
					exit(-1);
				}
				printf("message_size: %d\n", send_message_size);
				break;
			case 'n':
				num_send_request = atoi(optarg);
				break;
/*			case 'i':
				cpu_id = atoi(optarg);
				printf("cpu id @ parseOpt: %d\n", cpu_id);
				break;*/
			case 'p':
				portno = atoi(optarg);
				break;
			case 'h':
				{
					char *hostnames = strtok(optarg,",");
					int i=0;
					while(hostnames){
						if(i>=MAX_NUM_IB_THREADS)
							die("the number of clients is larger than the max");
						hosts[i++]= hostnames;
						hostnames=strtok(NULL, ",");
					}
				}
				break;
			case 'c':
				is_client = atoi(optarg);
				break;
			case 'b':
				internal_port_no = atoi(optarg);
				break;
			case 's':
				signaled = atoi(optarg);
				break;
			case 'e':
				event_mode = atoi(optarg);
				printf("enable event mode\n");
				break;
			case 'D':
				duration = atoi(optarg);
				break;
			case 'M':
				num_threads = atoi(optarg);
				break;
			default:
				fprintf(stderr, "usage: %s -m<message_size> -n<num_send_request> -s<signaled> -i<cpu_id> -e<event_mode> -h<host_ip> -c<is_client> -p<port_no> -D<duration> -M<num_threads>\n", argv[0]);
				exit(-1);
		}
	}
	if(event_mode && duration){
		printf("event mode can be only with number of requests\n");
		exit(-1);
	}

	if(num_send_request && duration){
		printf("please only specify one parameter, either num_requests or duration\n");
		exit(-1);
	}
			
	

}

static void * incast(void * arg){
	int wait_for_acks_ret;
	int index = (int) arg;

	bindingCPU(index);
	struct context *s_ctx=multi_ctx[index];
	struct connection *conn = s_ctx->conn;
	struct timespec start, end;
	int dur= 0;
	clock_gettime(CLOCK_MONOTONIC, &start);

	if(duration > 0){	
	//	clock_gettime(CLOCK_MONOTONIC, &end);
	//	dur =end.tv_sec - start.tv_sec;
		
		signal(SIGALRM, sigalarm_handler);
		if(sigsetjmp(jmpbuf,1)!=0){
			goto end_sending;
		}
		alarm(duration);
		
	}
	//for (i=0;i<conn->num_requests;i++){
	while(conn->num_sendcount < conn->num_requests ||dur < duration ){
		wait_for_acks_ret = on_write_read(s_ctx,conn,1, OPERATOR);
		if(wait_for_acks_ret){
			printf("reconnection happens, sendcount %d\n", conn->num_sendcount);
			//reconnect_qp(conn->qp,wait_for_acks_ret , conn->remote_qp_attr, conn->send_transport);
			//init_rone_server(conn);
		}
		conn->num_sendcount++;
		if(duration > 0){
			clock_gettime(CLOCK_MONOTONIC, &end);
			dur =end.tv_sec - start.tv_sec;
		}
	}
	wait_for_acks(conn->qp, s_ctx->send_comp_channel,0, s_ctx->send_cq, s_ctx->srv_state);
	
end_sending:
	clock_gettime(CLOCK_MONOTONIC, &end);

	long long unsigned diff = (long long unsigned)(BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec);
	long long unsigned data = (long long unsigned)conn->num_sendcount*conn->send_message_size*8;

	printf("data %llu, %llu\n", conn->num_sendcount, conn->send_message_size);
	double tput = data*1.0/diff;
	printf("throughput @ thread %d time %llu ns, size %llu bits, tput %f Gb/s\n", index, diff, data, tput);


}
int main(int argc, char **argv)
{
	parseOpt(argc,argv); 
	if(is_client==1 && hosts == NULL){
		printf("the number of parameters doest not match with server or client \n \
				client:<port><server><isClient> \n \
				server:<port>\n");
		exit(1);
	}

	struct ibv_device **dev_list = ibv_get_device_list(NULL) ;
	struct ibv_device *ib_dev = dev_list[0];

	int index;
	struct context *s_ctx;
	for(index=0;index<num_threads;index++){
		s_ctx = NULL;
		s_ctx = init_ctx(ib_dev,s_ctx);
		multi_ctx[index] = s_ctx;		
		//initilize connection
		struct connection *conn= s_ctx->conn;
		conn->send_message_size = send_message_size;
		conn->num_requests = num_send_request;
		conn->send_transport = RC_TRANSPORT;
		conn->signal_all = signaled;
		conn->window_size = window_size;
		conn->isClient= is_client;
		//if (conn->isClient == 0)	
		//	TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq,NULL));
		//TEST_NZ(pthread_create(&s_ctx->send_cq_poller_thread, NULL, poll_send_cq, NULL));

		conn->actual_completions =0;
		//queue pair initilization
		struct ibv_qp_init_attr qp_attr;
		build_qp_attr(&qp_attr,s_ctx);  
		TEST_Z(conn->qp = ibv_create_qp(s_ctx->pd, &qp_attr));
		get_qp_info(s_ctx);

		modify_qp_to_init(conn);
		//setting the memory area
		set_memory(conn, s_ctx, OPERATOR);
		/*	int blocks = RECV_BUFFER_SIZE/send_message_size;
			int i=0;
			for(i=0;i<blocks;i++){
			char c=i%26+'a';
			if (conn->isClient == 1){
		//set server memory
		fillzero(s_ctx->conn->recv_region+i*send_message_size, c, send_message_size);
		}else{
		//clear client memory
		fillzero(s_ctx->conn->recv_region+i*send_message_size, '0', send_message_size);
		}
		}*/

		if(conn->isClient){   
			printf("server %s\n", hosts[index]); 
			run_client(portno,hosts[index],conn);
			if(connect_ctx(conn, conn->local_qp_attr->psn, conn->remote_qp_attr)){
				printf("connect_ctx error at client side\n");
				return 1;
			}
			//if(OPERATOR == READ_OPERATOR  && conn->signal_all == 0)
			//	TEST_NZ(pthread_create(&s_ctx->write_poller_thread, NULL, write_cq, s_ctx->conn));
			//TEST_NZ(pthread_create(&s_ctx->throughput_timer_thread, NULL, throughput_timer, s_ctx->conn));
			sleep(1);	
			init_rone_server(s_ctx,index);
			//init_rtt();
			//exit(0);	
			//reply_operation(s_ctx->conn, OPERATOR);   
		
		}else{
			run_server(portno, conn, s_ctx->pd);
			init_rone_client(s_ctx);
		}

	}

	int i;
	for(i = 1; i < num_threads; i++){
		if(pthread_create(&ib_threads[i], NULL, incast, (void *) i) != 0)
			die("main(): Failed to create worker thread .");	
	}

	(void) incast((void*)(intptr_t)0);
	  /* Waiting for completion */
  	for( i= 1; i < num_threads; i++)
     		if(pthread_join(ib_threads[i], NULL) !=0 )
             		die("main(): Join failed for worker thread i");

		//if (conn->isClient == 0)	
		//	pthread_join(s_ctx->cq_poller_thread,NULL);
		//pthread_join(s_ctx->send_cq_poller_thread,NULL);

		return 0;
	}

	static int
read_server_params (rudp_srv_state_t *rudp_srv_state)
{
	int fd, val, lineno = 1;
	char buf[MAXLINE];

	/* Open the input file */
	fd = open(SERVER_INPUT, O_RDONLY);
	if (!fd) {
		return -1;
	}

	/* Read the parameters one line at a time */
	bzero(serv_params, sizeof(server_params_t));
	bzero(rudp_srv_state, sizeof(rudp_srv_state_t));
	bzero(buf, MAXLINE);
	while (readline(fd, buf, MAXLINE)) {
		val = atoi(buf);
		if (val == 0) {
			return -1;
		}

		if (lineno == 1) {
			serv_params->port = val;
		} else if (lineno == 2) {
			rudp_srv_state->max_cwnd_size = val;
		}
		lineno++;
		bzero(buf, MAXLINE);
	}

	printf("\nSERVER PARAMS:\n");
	printf("port: %d\n", serv_params->port);
	printf("sending window size: %d\n", rudp_srv_state->max_cwnd_size);

	return 0;
}


int init_rone_server(struct context *s_ctx, int index ){

	int ret;
	rudp_srv_state_t rudp_srv_state;
	struct connection  *conn= s_ctx->conn;
	/* Initialize the structure for holding server parameters */
	serv_params = (server_params_t *)malloc(sizeof(server_params_t));
	if (!serv_params) {
		printf("main: failed to initialize server parameters\n");
		return -1;
	}

	/* Read the server parameters from server.in */
	ret = read_server_params(&rudp_srv_state);
	if (ret != 0) {
		printf("main: failed to read server parameters from server.in\n");
		return -1;
	}

	rudp_srv_state.local_qp_attribute = conn->local_qp_attr;
	rudp_srv_state.remote_qp_attribute = conn->remote_qp_attr;
	rudp_srv_state.send_transport = conn->send_transport;
	rudp_srv_state.ib_ctx = s_ctx->ctx;
	rudp_srv_state.send_channel = s_ctx->send_comp_channel;
	rudp_srv_state.recv_cq = s_ctx->cq;
	rudp_srv_state.send_cq = s_ctx->send_cq;
	rudp_srv_state.qp = conn->qp;
	rudp_srv_state.usr_recv_cq = s_ctx->usr_recv_cq;
	rudp_srv_state.usr_send_cq = s_ctx->usr_send_cq;
	rudp_srv_state.hostname  = hosts[index];
	rudp_srv_state.inter_port_no  = internal_port_no;
	rudp_srv_state.event = event_mode;
	//rudp_srv_state.duration = duration;
	/* Initialize the parameters based on what is sent by the server */
	s_ctx->srv_state = (rudp_srv_state_t *)malloc(sizeof(rudp_srv_state_t));
	if (!s_ctx->srv_state) {
		return -1;
	}

	bzero(s_ctx->srv_state, sizeof(rudp_srv_state_t));

//check_timestamp_enable(rudp_srv_state.ib_ctx);	
	/* Initialize the RUDP library */
	ret = rudp_srv_init(&rudp_srv_state, s_ctx->srv_state);
	if (ret != 0) {
		printf("main: failed to initialize the RUDP library\n");
		return -1;
	}

}

/*static int
  read_client_params (rudp_cli_state_t *rudp_cli_state)
  {
  int fd, lineno = 1;
  char buf[MAXLINE];
  struct hostent *hostent;

  fd = open(CLIENT_INPUT, O_RDONLY);
  if (!fd) {
  return -1;
  }

  bzero(cli_params, sizeof(client_params_t));
  bzero(rudp_cli_state, sizeof(rudp_cli_state_t));
  bzero(buf, MAXLINE);
  while (readline(fd, buf, MAXLINE)) {
  switch (lineno) {
  case 1:
  strncpy(cli_params->server_ip, buf, IP_LEN);
  cli_params->server_ip[strlen(buf) - 1] = 0;
  break;
  case 2:
  cli_params->server_port = atoi(buf);
  if (cli_params->server_port == 0) {
  return -1;
  }
  break;
  case 3:
  strncpy(cli_params->file, buf, MAX_FILENAME_LEN);
  cli_params->file[strlen(buf) - 1] = 0;
  break;
  case 4:
  rudp_cli_state->advw_size = atoi(buf);
  if (rudp_cli_state->advw_size == 0) {
  return -1;
  }
  break;
  case 5:
  rudp_cli_state->random_seed = atoi(buf);
  if (rudp_cli_state->random_seed == 0) {
  return -1;
  }
  break;
  case 6:
  rudp_cli_state->data_loss_prob = atof(buf);
  break;
  case 7:
  rudp_cli_state->recv_rate = atoi(buf);
  if (rudp_cli_state->recv_rate == 0) {
  return -1;
  }
  break;
  }
  lineno++;
  bzero(buf, MAXLINE);
  }

  printf("CLIENT PARAMS:\n");
  printf("server ip: %s\n", cli_params->server_ip);
  printf("server port: %d\n", cli_params->server_port);
  printf("filename: %s\n", cli_params->file);
  printf("advertised window: %d\n", rudp_cli_state->advw_size);
  printf("seed value: %d\n", rudp_cli_state->random_seed);
  printf("data loss probability: %f\n", rudp_cli_state->data_loss_prob);
  printf("receive rate (ms): %d\n", rudp_cli_state->recv_rate);

  hostent = gethostbyname(cli_params->server_ip);
  if (!hostent) {
  printf("\nread_client_params: invalid server address\n");
  return -1;
  }

return 0;
}
*/


int init_rone_client(struct context *s_ctx){
	int ret, is_local, sock_len;
	rudp_cli_state_t rudp_cli_state;
	char buf[MAXLINE];
	struct sockaddr_in server_addr, sock_addr;

	/* Initialize the structure for holding client parameters */
	cli_params = (client_params_t *)malloc(sizeof(client_params_t));
	if (!cli_params) {
		printf("main: failed to initialize client parameters\n");
		return -1;
	}

	/* Read the client parameters from client.in */
	/*    ret = read_client_params(&rudp_cli_state);
	      if (ret != 0) {
	      printf("main: failed to read client parameters from client.in\n");
	      return -1;
	      }
	 */
	struct connection *conn = s_ctx->conn;
	rudp_cli_state.local_qp_attribute = conn->local_qp_attr;
	rudp_cli_state.remote_qp_attribute = conn->remote_qp_attr;
	rudp_cli_state.send_transport = conn->send_transport;
	rudp_cli_state.ib_ctx = s_ctx->ctx;
	rudp_cli_state.qp = conn->qp;
	rudp_cli_state.recv_channel = s_ctx->comp_channel;
	rudp_cli_state.send_channel = s_ctx->send_comp_channel;
	rudp_cli_state.send_cq = s_ctx->send_cq;
	rudp_cli_state.peer_mr = conn->peer_mr;
	rudp_cli_state.send_region = conn->send_region;
	rudp_cli_state.send_mr = conn->send_mr;
	rudp_cli_state.inter_port_no = internal_port_no;
	/* Initialize the RUDP library */
	/* Initialize the parameters based on what is sent by the client */
	s_ctx->cli_state = (rudp_cli_state_t *)malloc(sizeof(rudp_cli_state_t));
	if (!s_ctx->cli_state) {
		return -1;
	}

	/* Store the parameters specified in client.in file */
	bzero(s_ctx->cli_state, sizeof(rudp_cli_state_t));

	ret = rudp_cli_init(&rudp_cli_state, s_ctx->cli_state);
	if (ret != 0) {
		printf("main: failed to initialize the RUDP library\n");
		return -1;
	}
	return 0;

}

/*void reply_operation(struct connection *conn, int USE_WRITE){
	if(conn->num_sendcount == 0 && conn->isClient == 1 ){
		int i;
		for(i=0;i<conn->window_size/DUPLICATE_COPIES;i++){
			conn->num_sendcount ++;
			if(USE_WRITE){
				int j=0;
				for (j=0; j< DUPLICATE_COPIES;j++)
					on_write_read(s_ctx, conn,0, USE_WRITE);
				//rudp_send_read_write(s_ctx,conn, 0, USE_WRITE);
			}else
				on_send(conn,0);
			printf("num_sendcount %d\n", conn->num_sendcount);
			record_timestamp(conn->num_sendcount-conn->num_requests/2, starttimestamp);
		}
		return;
	}

	conn->num_sendcount++;
	if(conn->num_sendcount*DUPLICATE_COPIES%MAX_SEND_WR == 0){
		if(USE_WRITE){
			int	j=0;
			for (j=0;j<DUPLICATE_COPIES;j++)
				on_write_read(s_ctx, conn,1, USE_WRITE);
			//rudp_send_read_write(s_ctx,conn, 0, USE_WRITE);

		}else
			on_send(conn,1);
	}else{
		//while(1){ if(conn->isSignal == 0) break;}
		if(USE_WRITE){
			int j=0;
			for(j=0;j<DUPLICATE_COPIES;j++)
				on_write_read(s_ctx, conn,0, USE_WRITE);
			//rudp_send_read_write(s_ctx,conn, 0, USE_WRITE);
		}else
			on_send(conn,0);
	}
	record_timestamp(conn->num_sendcount-conn->num_requests/2, starttimestamp);
}*/

/*void recv_message(struct connection *conn){
	conn->actual_completions ++;
	char * region = conn->recv_region;
	int blocks = RECV_BUFFER_SIZE/conn->send_message_size;
	char c = (char)region[(conn->num_completions)%blocks*conn->send_message_size]; 
	char exp =( (conn->num_completions)/blocks)%26+'a';
	if( c == exp ){
		record_timestamp(conn->num_completions - conn->num_requests/2, endtimestamp);
		conn->num_completions ++;
	}

	if (conn->num_completions < conn->num_requests && conn->num_sendcount*DUPLICATE_COPIES-conn->actual_completions < conn->window_size){
		// keep the requests on the network less than MAX_SEND_WR
		reply_operation(conn, OPERATOR);
	}else if (conn->num_completions >= conn->num_requests){
		printf("conn->num_completions: %d\n", conn->num_completions);
		per_request_latency(conn, starttimestamp, endtimestamp);
		exit(0); 		
	}
}*/

void on_completion(struct ibv_wc *wc, int k)
{
	struct connection *conn = (struct connection *)(uintptr_t)wc->wr_id;
	//printf("opcode = %s\n", ibv_wc_opcode_str(wc->opcode));
	if (wc->status != IBV_WC_SUCCESS){
		printf("wc->status: %s\n", ibv_wc_status_str(wc->status));
		die("");
	}

/*	if(wc->opcode == IBV_WC_RDMA_READ){    
		conn->isSignal = 0;
		if(conn->signal_all){
			recv_message(conn);
			//printf("conn->num_completions: 1\n");
			 // conn->num_completions++;

			  //record_timestamp(conn->num_completions, endtimestamp);
		}//sending 
	}*/

}

void * write_cq(void *context)
{
	struct connection *conn = (struct connection *)context;
	char *region = conn->recv_region;
	int blocks = RECV_BUFFER_SIZE/conn->send_message_size;
	while (1) {
		char c = (char)region[conn->num_completions%blocks*conn->send_message_size]; 
		if(c > ((conn->num_completions/blocks)%blocks+96)){
			printf("messages %.*s, completions: %d\n", 4, region+conn->num_completions%blocks*conn->send_message_size, conn->num_completions );
			//		fillzero(region+conn->num_sendcount%blocks*conn->send_message_size,'0', conn->send_message_size);
			//recv_message(conn);
		}
	}

	return NULL;
}

//void * poll_cq(void *ctx)
//{
	/*	struct ibv_cq *cq;
		struct ibv_wc wc;

		while (1) {
		TEST_NZ(ibv_get_cq_event(s_ctx->comp_channel, &cq, &ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));
		int k =ibv_poll_cq(cq, 1, &wc);
		while (k){
	//printf("poll cq k = %d\n", k);
	on_completion(&wc,k);
	k = ibv_poll_cq(cq, 1, &wc);
	}
	}
	 */
/*	int prev_psn=-1;	
	while(1){
		sleep(1);
		int rq_psn= get_current_my_psn(s_ctx->conn->qp,0);		*/
		/*if (rq_psn == prev_psn)
		  reconnect_qp(s_ctx->conn->qp, s_ctx->conn->local_qp_attr->psn, s_ctx->conn->remote_qp_attr, s_ctx->conn->send_transport);	
		  else
		  prev_psn = rq_psn;	
		 */
/*	}

	return NULL;
}*/

/*void * poll_send_cq(void *ctx)
{
	struct ibv_cq *cq;
	struct ibv_wc wc;

	while (1){
		TEST_NZ(ibv_get_cq_event(s_ctx->send_comp_channel, &cq, &ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));
		int k = ibv_poll_cq(cq, 1, &wc);
		while (k){
			printf("k = %d\n", k);
			//		on_completion(&wc,k);
			k = ibv_poll_cq(cq, 1, &wc);
		}
	}
	return NULL;
}*/
void run_server(int portno,struct connection *conn, struct ibv_pd *pd){
	int sockfd, newsockfd; // portno;
	socklen_t clilen;

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
		die("ERROR on binding");
	listen(sockfd,5);

	clilen = sizeof(cli_addr);
	newsockfd = accept(sockfd, 
			(struct sockaddr *) &cli_addr, 
			&clilen);
	if (newsockfd < 0) 
		die("ERROR on accept");

	//memset(s_ctx->remote_qp_attr, 0, S_QPA);
	conn->remote_qp_attr = (struct qp_attr *)malloc(sizeof(struct qp_attr));
	n = recv(newsockfd,conn->remote_qp_attr,S_QPA,0);
	if (n < 0) die("ERROR reading from socket");

	//print_qp_attr(conn->remote_qp_attr);

	n = send(newsockfd,conn->local_qp_attr,S_QPA,0);
	if (n < 0) die("ERROR writing to socket");
	//print_qp_attr(conn->local_qp_attr);

	//if(USE_WRITE){  
	conn->peer_mr = (struct ibv_mr*)malloc(sizeof(struct ibv_mr));
	n=recv(newsockfd, conn->peer_mr,sizeof(struct ibv_mr),0);
	if (n < 0) die("ERROR reading from peer_mr");
	n=send(newsockfd,conn->recv_mr, sizeof(struct ibv_mr),0);
	if (n<0) die("ERROR writing send_r");   
	// }

	if(conn->send_transport == 2){
		create_ah(conn, pd);
		modify_dgram_qp_to_rts(conn,  conn->local_qp_attr->psn);
	}else{
		if(connect_ctx(conn, conn->local_qp_attr->psn, 
					conn->remote_qp_attr)) {
			fprintf(stderr, "Couldn't connect to remote QP\n");
			exit(0);
		}
	}


	close(newsockfd);
	close(sockfd);
	return ; 
}

void run_client(int portno, char *hostname, struct connection *conn){
	int sockfd,  n;
	struct sockaddr_in serv_addr;
	struct hostent *server;

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
		die("ERROR connecting");

	n = send(sockfd,conn->local_qp_attr, S_QPA,0);
	if (n < 0) 
		die("ERROR writing to socket");
//		print_qp_attr(conn->local_qp_attr);

	conn->remote_qp_attr = (struct qp_attr *)malloc(sizeof(struct qp_attr));
	memset(conn->remote_qp_attr, 0, S_QPA);
	n = recv(sockfd,conn->remote_qp_attr,S_QPA,0);
	if (n < 0) 
		die("ERROR reading from socket");
//	print_qp_attr(conn->remote_qp_attr);

	// if(USE_WRITE){
	n=send(sockfd,conn->recv_mr, sizeof(struct ibv_mr),0);
	if (n<0)die("ERROR writing send_r");

	conn->peer_mr = (struct ibv_mr*)malloc(sizeof(struct ibv_mr));
	n=recv(sockfd, conn->peer_mr,sizeof(struct ibv_mr),0);
	if (n < 0) die("ERROR reading from peer_mr");
	//}

	close(sockfd);
	return ;
}


