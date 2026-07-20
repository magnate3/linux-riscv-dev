#define __USE_GNU
#define _GNU_SOURCE

#include <sched.h>


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <rdma/rdma_cma.h>


#include "common.h"

#define OPERATOR READ_OPERATOR 

void reply_operation(struct connection *conn, int USE_WRITE);
void *write_cq(void *context);
void run_client(int port, char *host, struct connection *conn);
void run_server(int portno,struct connection *conn, struct ibv_pd *pd);
void on_completion(struct ibv_wc *wc, int k);
void * poll_cq(void *ctx);
void * poll_send_cq(void *ctx);
void recv_message(struct connection *conn);

long long unsigned starttimestamp[MAX_TIMESTAMP_SIZE];
long long unsigned endtimestamp[MAX_TIMESTAMP_SIZE];
int send_message_size = 256;
int num_send_request =0;
int window_size= 64;
int signaled=1;
int cpu_id=0;
char *host=NULL;
int portno=55281;
int is_client=0;

void parseOpt(int argc, char **argv){
	int c; 
	while ((c = getopt(argc, argv, "s:m:n:c:w:h:p:i")) != -1) {
		switch (c)
		{
			case 'm':
				send_message_size = atoi(optarg);
				printf("message_size: %d\n", send_message_size);
				break;
			case 'n':
				num_send_request = atoi(optarg);
				break;
			case 'i':
				cpu_id = atoi(optarg);
				break;
			case 'p':
				portno = atoi(optarg);
				break;
			case 'h':
				host = optarg;
				break;
			case 'c':
				is_client = atoi(optarg);
				break;
			case 'w':
				window_size = atoi(optarg);
				break;
			case 's':
				signaled = atoi(optarg);
				break;
			default:
				fprintf(stderr, "usage: %s -m<message_size> -n<num_send_request> -s<signaled> -i<cpu_id> -w<window_size> -h<host_ip> -c<is_client> -p<port_no>\n", argv[0]);
				exit(-1);
		}
	}

}


struct context *s_ctx = NULL;

int main(int argc, char **argv)
{
	parseOpt(argc,argv); 
	if(is_client==1 && host == NULL){
		printf("the number of parameters doest not match with server or client \n \
				client:<port><server><isClient> \n \
				server:<port>\n");
		exit(1);
	}

	bindingCPU(cpu_id);

	struct ibv_device **dev_list = ibv_get_device_list(NULL) ;
	struct ibv_device *ib_dev = dev_list[0];
	s_ctx = init_ctx(ib_dev,s_ctx);
	TEST_NZ(pthread_create(&s_ctx->cq_poller_thread, NULL, poll_cq,NULL));
	TEST_NZ(pthread_create(&s_ctx->send_cq_poller_thread, NULL, poll_send_cq, NULL));

	//initilize connection
	struct connection *conn= s_ctx->conn;
	conn->send_message_size = send_message_size;
	conn->num_requests = num_send_request;
	conn->send_transport = RC_TRANSPORT;
	conn->signal_all = signaled;
	conn->window_size = window_size;
	conn->isClient= is_client;

	//queue pair initilization
	struct ibv_qp_init_attr qp_attr;
	build_qp_attr(&qp_attr,s_ctx);  
	TEST_Z(conn->qp = ibv_create_qp(s_ctx->pd, &qp_attr));
	modify_qp_to_init(conn);
	get_qp_info(s_ctx);
	
	//setting the memory area
	set_memory(conn, s_ctx, OPERATOR);
	int blocks = RECV_BUFFER_SIZE/send_message_size;
	int i=0;
	for(i=0;i<blocks;i++){
		char c=i%26+'a';
		if (conn->isClient == 0){
			//set server memory
			fillzero(s_ctx->conn->recv_region+i*send_message_size, c, send_message_size);
		}else{
			//clear client memory
			fillzero(s_ctx->conn->recv_region+i*send_message_size, '0', send_message_size);
		}
	}

	if(conn->isClient){    
		run_client(portno,host,conn);
		if(connect_ctx(conn, conn->local_qp_attr->psn, conn->remote_qp_attr)){
			printf("connect_ctx error at client side\n");
			return 1;
		}
		if(OPERATOR == READ_OPERATOR  && conn->signal_all == 0)
			TEST_NZ(pthread_create(&s_ctx->write_poller_thread, NULL, write_cq, s_ctx->conn));
		reply_operation(s_ctx->conn, OPERATOR);   
	}else{
		run_server(portno, conn, s_ctx->pd);
	}

	pthread_join(s_ctx->cq_poller_thread,NULL);
	pthread_join(s_ctx->send_cq_poller_thread,NULL);

	return 0;
}


void reply_operation(struct connection *conn, int USE_WRITE){
	if(conn->num_sendcount == 0 && conn->isClient == 1 ){
		int i;
		for(i=0;i<conn->window_size;i++){
			if(USE_WRITE){
				on_write_read(conn,0, USE_WRITE);
			}else
				on_send(conn,0);
			record_timestamp(conn->num_sendcount-conn->num_requests/2, starttimestamp);
			conn->num_sendcount ++;
		}
	}

	if(conn->num_sendcount%MAX_SEND_WR == 0){
		if(USE_WRITE)
			on_write_read(conn,1, USE_WRITE);
		else
			on_send(conn,1);
	}else{
		while(1){ if(conn->isSignal == 0) break;}
		if(USE_WRITE)
			on_write_read(conn,0, USE_WRITE);
		else
			on_send(conn,0);
	}
	record_timestamp(conn->num_sendcount-conn->num_requests/2, starttimestamp);
	conn->num_sendcount++;
}

void recv_message(struct connection *conn){
	record_timestamp(conn->num_completions-conn->num_requests/2, endtimestamp);
	conn->num_completions ++;
	if (conn->num_completions < conn->num_requests){
		reply_operation(conn, OPERATOR);
	}else if (conn->num_completions > (conn->num_requests+conn->window_size)){
		per_request_latency(conn, starttimestamp, endtimestamp);
		exit(0); 		
	}
}

void on_completion(struct ibv_wc *wc, int k)
{
	struct connection *conn = (struct connection *)(uintptr_t)wc->wr_id;
	//printf("opcode = %s\n", ibv_wc_opcode_str(wc->opcode));
	if (wc->status != IBV_WC_SUCCESS){
		printf("wc->status: %s\n", ibv_wc_status_str(wc->status));
		die("");
	}
	if(wc->opcode == IBV_WC_RDMA_READ){    
		conn->isSignal = 0;
		if(conn->signal_all){
			recv_message(conn);
		}//sending 
	}

}

void * write_cq(void *context)
{
	struct connection *conn = (struct connection *)context;
	char *region = conn->recv_region;
	int blocks = RECV_BUFFER_SIZE/conn->send_message_size;
	while (1) {
		if((char)region[conn->num_completions%blocks*conn->send_message_size] !='0'){
			fillzero(region+conn->num_sendcount%blocks*conn->send_message_size,'0', conn->send_message_size);
			recv_message(conn);
		}
	}

	return NULL;
}

void * poll_cq(void *ctx)
{
	struct ibv_cq *cq;
	struct ibv_wc wc;

	while (1) {
		TEST_NZ(ibv_get_cq_event(s_ctx->comp_channel, &cq, &ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));
		int k =ibv_poll_cq(cq, 1, &wc);
		while (k){
			//printf("poll cq k = %d\n", k);
			on_completion(&wc,k);
		}
		k = ibv_poll_cq(cq, 1, &wc);
	}

	return NULL;
}

void * poll_send_cq(void *ctx)
{
	struct ibv_cq *cq;
	struct ibv_wc wc;

	while (1){
		TEST_NZ(ibv_get_cq_event(s_ctx->send_comp_channel, &cq, &ctx));
		ibv_ack_cq_events(cq, 1);
		TEST_NZ(ibv_req_notify_cq(cq, 0));
		int k = ibv_poll_cq(cq, 1, &wc);
		while (k){
			//printf("k = %d\n", k);
			on_completion(&wc,k);
		}
		k = ibv_poll_cq(cq, 1, &wc);
	}
	return NULL;
}
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

	print_qp_attr(conn->remote_qp_attr);

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
//	print_qp_attr(conn->local_qp_attr);

	conn->remote_qp_attr = (struct qp_attr *)malloc(sizeof(struct qp_attr));
	memset(conn->remote_qp_attr, 0, S_QPA);
	n = recv(sockfd,conn->remote_qp_attr,S_QPA,0);
	if (n < 0) 
		die("ERROR reading from socket");
	print_qp_attr(conn->remote_qp_attr);

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


