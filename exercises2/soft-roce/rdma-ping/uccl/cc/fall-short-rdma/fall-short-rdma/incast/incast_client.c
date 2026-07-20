#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h> //for exp()
#include <sys/time.h> //for gettimeofday()
#include <sys/epoll.h>
#include "csapp.h"
#include "../util.h"
#define XCLOCK_GETTIME(tp)   ({  \
		if (clock_gettime(CLOCK_REALTIME, (tp)) != 0) {                   \
		printf("XCLOCK_GETTIME failed!\n");              \
		exit(-1);                                            \
		}})

#define XEPOLL_CTL(efd, flags, fd, event)   ({  \
		if (epoll_ctl(efd, flags, fd, event) < 0) { \
		perror ("epoll_ctl");                   \
		exit (-1);                              \
		}})


#define WRITE_BUF_SIZE 128	/* Per-connection internal buffer size for writes. */
#define READ_BUF_SIZE (1<<20) /* Read buffer size. 32MB. */
#define MAXEVENTS 1024  /* Maximum number of epoll events per call */
#define INIT_CONNECT_MEASURE



int startSecs = 0;
pthread_t latency_threads[MAX_CPUS];
char *hosts[MAX_CPUS]={NULL};
int total_flows;
static int flows[MAX_CPUS];
static int done[MAX_CPUS];
static int connection_create[MAX_CPUS][1000];


int send_message_size = 1024;
int num_send_request =1;
int window_size= 1;
int signaled=0;
//int cpu_id=0;
int portno=55281;
int internal_port_no = 66889;
int is_client=0;
int event_mode = 0;
int duration=0;
int num_threads=1;
int queue_depth=16;
int is_sync=0;
int rate_limit_mode=1;
long long unsigned syntime;


/*
 * Data structure to keep track of server connection state.
 *
 * The connection objects are also maintained in a global doubly-linked list.
 * There is a dummy connection head at the beginning of the list.
 */
struct conn {
	/* Points to the previous connection object in the doubly-linked list. */
	struct conn *prev;	
	/* Points to the next connection object in the doubly-linked list. */
	struct conn *next;	
	/* File descriptor associated with this connection. */
	int fd;			
	/* Internal buffer to temporarily store the contents of a read. */
	char buffer[WRITE_BUF_SIZE];	
	/* Size of the data stored in the buffer. */
	//size_t size;			
	/* The incast that this connection is a part of. */
	struct incast *incast;

	/* Number of bytes requested for the current message. */
	uint64_t request_bytes;
	/* Number of bytes of the current message read. */
	uint64_t read_bytes;		

	/* The startting time of the connection. */
	struct timespec start;
	/* The finishing time of the connection. */
	struct timespec finish;
};
/* 
 * Data structure to keep track of active server connections.
 */
struct incast_pool { 
	/* The epoll file descriptor. */
	int efd;
	/* The epoll events. */
	struct epoll_event events[MAXEVENTS];
	/* Number of ready events returned by epoll. */
	int nevents;  	  		
	/* Doubly-linked list of active server connection objects. */
	//    struct incast *incast_head;
	/* Number of active incasts. */
	//unsigned int nr_incasts;
}; 

/*
 * Data structure to keep track of each independent incast event.
 * Each incast holds the completion count and timers.
 */
struct incast {
	/* Points to the previous incast object in the doubly-linked list. */
	struct incast *prev;
	/* Points to the next incast object in the doubly-linked list. */
	struct incast *next;
	/* The server request unit. */
	uint64_t sru;
	/* The total number of requests sent out. */
	uint32_t numreqs;
	uint32_t nflows;
	/* The number of completed requests so far. */
	uint32_t completed;
	
	uint8_t read_buf[READ_BUF_SIZE];
	int core;
	/* Doubly-linked list of active server connection objects. */
	struct conn *conn_head;

	/* The startting time of the incast. */
	struct timespec start;
	/* The finishing time of the incast. */
	struct timespec finish;
	/* Number of active connections. */
	unsigned int nr_conns;

	struct incast_pool *epool;

};


/* Set verbosity to 1 for debugging. */
static int verbose = 0;

struct incast *multi_incasts[MAX_CPUS];
struct timespec main_start, main_end;

struct hostent *hp;
/* Local function definitions. */
static void increment_and_check_incast(struct incast *inc);

double get_secs(struct timespec time)
{
	return (time.tv_sec) + (time.tv_nsec * 1e-9);
}
double get_secs_since_start(struct timespec time, long startsecs)
{
	return get_secs(time) - startsecs;
}

/*
 * open_server - open connection to server at <hostname, port> 
 *   and return a socket descriptor ready for reading and writing.
 *   Returns -1 and sets errno on Unix error. 
 *   Returns -2 and sets h_errno on DNS (gethostbyname) error.
 */
int open_server(char *hostname, int port) 
{
	int serverfd;
	struct sockaddr_in serveraddr;
	int opts = 0;
	int optval = 1;

	if ((serverfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
		return -1; /* check errno for cause of error */

	/* Set the connection descriptor to be non-blocking. */
	opts = fcntl(serverfd, F_GETFL);
	if (opts < 0) {
		printf("fcntl error.\n");
		exit(-1);
	}
	opts = (opts | O_NONBLOCK);
	if (fcntl(serverfd, F_SETFL, opts) < 0) {
		printf("fcntl set error.\n");
		exit(-1);
	}

	/* Set the connection to allow for reuse */
	if (setsockopt(serverfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
		printf("setsockopt error.\n");
		exit(-1);
	}
	/*else{
	
		printf("%s\n", (char *)hp->h_name);
	}*/
	bzero((char *) &serveraddr, sizeof(serveraddr));
	serveraddr.sin_family = AF_INET;
	bcopy((char *)hp->h_addr, 
			(char *)&serveraddr.sin_addr.s_addr, hp->h_length);
	serveraddr.sin_port = htons(port);

	/* Establish a connection with the server */
	if (connect(serverfd, (SA *) &serveraddr, sizeof(serveraddr)) < 0) {
		if (errno !=EINPROGRESS) {
			printf("server connect error");
			perror("");
			return -1;
		}     
	}
	return serverfd;
}

/*******************************************************************************
 * Maintaining Client Connections.
 ******************************************************************************/

/* 
 * Requires:
 * c should be a connection object and not be NULL.
 * p should be a connection pool and not be NULL.
 *
 * Effects:
 * Adds the connection object to the tail of the doubly-linked list.
 */
	static void
add_conn_list(struct conn *c, struct incast *i)
{
	if(c == NULL || i == NULL )
		printf("NULL in \n");
	if( i->conn_head == NULL )
		printf("NULL conn_head\n");
	if( i->conn_head->next == NULL)
		printf("i->conn_head->next\n");
	c->next = i->conn_head->next;
	c->prev = i->conn_head;
	i->conn_head->next->prev = c;
	i->conn_head->next = c;
}

/* 
 * Requires:
 * c should be a connection object and not be NULL.
 *
 * Effects:
 * Removes the connection object from the doubly-linked list.
 */
	static void
remove_conn_list(struct conn *c)
{
	c->next->prev = c->prev;
	c->prev->next = c->next;
}

void remove_connection(struct conn *c){
	/* Close the file descriptor. */
	Close(c->fd); 

	/* Decrement the number of connections. */
	c->incast->nr_conns--;

	/* Remove the connection from the list. */
	remove_conn_list(c);

	/* Stop Timing and Log. */
	XCLOCK_GETTIME(&c->finish);
	double tot_time = get_nsecs(c->finish) - get_nsecs(c->start);
	double bandwidth = (c->read_bytes * 8.0 )/(tot_time);
	//printf("- [%.9g, %lu, %.9g, 1]\n", tot_time, c->read_bytes, bandwidth);
//	printf("- [%.9g, %lu, %.9g, %d]\n", tot_time / 1000, c->read_bytes, bandwidth, c->fd);

	/* Error Checking. */
	if (c->read_bytes != c->request_bytes) {
		printf("Read bytes (%lu) != Request bytes (%lu) for fd %d\n",
				c->read_bytes, c->request_bytes, c->fd);
		fprintf(stderr, "Read bytes (%lu) != Request bytes (%lu) for fd %d\n",
				c->read_bytes, c->request_bytes, c->fd);
	}

	/* Increment and check the number of finished connections. */
	increment_and_check_incast(c->incast);

	/* Free the connection object. */
	Free(c);

}
/* 
 * Requires:
 * c should be a connection object and not be NULL.
 * inc should be an incast and not be NULL.
 * p should be an incast pool and not be NULL.
 *
 * Effects:
 * Closes a server connection and cleans up the associated state. Removes it
 * from the doubly-linked list and frees the connection object.
 */
	void
remove_server(struct conn *c, struct incast_pool *p)
{
	if (verbose)
		printf("Closing connection fd %d...\n", c->fd);

	/* Supposedly closing the file descriptor cleans up epoll,
	 * but do it first anyways to be nice... */
	XEPOLL_CTL(p->efd, EPOLL_CTL_DEL, c->fd, NULL);

}

/* 
 * Requires:
 * connfd should be a valid connection descriptor.
 * inc should be an incast and not be NULL.
 *
 * Effects:
 * Allocates a new connection object and initializes the associated state. Adds
 * it to the doubly-linked list.
 */
	static void 
add_server(int connfd, struct timespec start, struct incast *inc) 
{
	struct conn *new_conn;
	struct epoll_event event;

	/* Allocate a new connection object. */
	new_conn =(struct conn*) Malloc(sizeof(struct conn));
		
	//printf("add_server done %d, %d\n", connfd, inc->core);
	new_conn->fd = connfd;
	//new_conn->size = 0;
	new_conn->incast = inc;

	/* No bytes have been requested or written yet. */
	new_conn->request_bytes = 0;
	new_conn->read_bytes = 0;

	//printf("1add_server done %d, %d\n", connfd, inc->core);
	/* Add this descriptor to the write descriptor set. */
	event.data.fd = connfd;
	event.data.ptr = new_conn;
	event.events = EPOLLOUT;
	XEPOLL_CTL(inc->epool->efd, EPOLL_CTL_ADD, connfd, &event);

	/* Update the number of server connections. */
	inc->nr_conns++;

	new_conn->start = start;

	//printf("2add_server done %d, %d\n", connfd, inc->core);
	add_conn_list(new_conn, inc);
	//printf("0add_server done %d, %d\n", connfd, inc->core);
}

/* 
 * Requires:
 * hostname should be a valid hostname
 * port should be a valid port
 * p should be a connection pool and not be NULL.
 *
 * Effects:
 * Accepts a new server connection. Sets the resulting connection file
 * descriptor to be non-blocking. Adds the server to the connection pool.
 */
	static void
start_new_connection(char *hostname, int port, struct incast *inc, int j)
{
	int connfd;
	struct timespec start;

	//XXX: Wrong spot for this
	/* Start Timing. */
	XCLOCK_GETTIME(&start);

	/* Accept the new connection. */
	connfd = open_server(hostname, port);
	
	if (connfd < 0) {
		printf("# Unable to open connection to (%s, %d)\n", hostname, port);
		return;
		//exit(-1);
	}

	if (verbose) {
		printf("Started new connection with (%s %d) new fd %d...\n",
				hostname, port, connfd);
	}

	/* Create new connection object and add it to the connection pool. */
	add_server(connfd, start, inc);
}

/*******************************************************************************
 * Manage Incast Events.
 ******************************************************************************/

/* 
 * Requires:
 * sru should be the server request unit in bytes.
 * numreqs should be the number of requests
 *
 * Effects:
 * Allocates an incast object and initializes it. 
 */
	static struct incast *
alloc_incast(uint64_t sru, uint32_t nflows, uint32_t numreqs)
{
	struct incast *inc;

	if (verbose)
		printf("Allocating incast\n");

	inc = Malloc(sizeof(struct incast));
	inc->sru = sru;
	inc->numreqs = numreqs;
	inc->completed = 0;
	inc->nflows = nflows; 
	inc->nr_conns = 0;
	memset(inc->read_buf, 0, sizeof(inc->read_buf));
	/* Allocate and initialize the dummy connection head. */
	inc->conn_head = Malloc(sizeof(struct conn));
	inc->conn_head->next = inc->conn_head;
	inc->conn_head->prev = inc->conn_head;

	/* Start Timing. */
	XCLOCK_GETTIME(&inc->start);

	return (inc);
}

/* 
 * Requires:
 * inc should be an incast object and not be NULL.
 *
 * Effects:
 * Frees the incast object and the memory holding the contents of the message.
 */
	static void
free_incast(struct incast *inc)
{
	if (verbose)
		printf("Freeing incast\n");

	Free(inc->conn_head);
	//Free(inc->read_buf);
	Free(inc);
//	exit(0);
	return ;
}

/* 
 * Requires:
 * inc should be an incast object and not be NULL.
 *
 * Effects:
 * Removes the incast object from the doubly-linked list.
 */
	static void
remove_incast_list(struct incast *inc)
{
	inc->next->prev = inc->prev;
	inc->prev->next = inc->next;
}



/* 
 * Requires:
 * sru is the server request unit
 * numreqs is the number of servers to the request from
 * hosts is an array of the form [hostname, port]*
 * hostsc is the length of the hosts array
 * p is a valid incast pool and is not NULL
 *
 * Effects:
 * Starts the incast event.
 */
	static void
start_incast(uint64_t sru, uint32_t nflows, char *hostname, int port, struct incast_pool *p, int index)
{
	struct incast *inc;

	int j;
	/* Alloc the incast and add it to the incast_pool */
	inc = alloc_incast((uint64_t)sru*num_send_request, nflows, 1);
//	printf("index %d, nflows %d\n", index,nflows);
	multi_incasts[index] = inc;
	inc->core = index;
	inc->epool = p;
	//p->nr_incasts++;

	/* Connect to the servers */
	if (verbose)
		printf("Starting new connection to (%s %d)...\n", hostname, port);
	for(j=0;j<nflows;j++){
#ifdef INIT_CONNECT_MEASURE
	        struct timespec start, end;
        	long long unsigned diff;
		clock_gettime(CLOCK_MONOTONIC, &start);
#endif
		//printf("add flow %d, at core %d\n",  j, index);
		start_new_connection(hostname, port, inc, j);
#ifdef INIT_CONNECT_MEASURE
		clock_gettime(CLOCK_MONOTONIC, &end);
		diff = (long long unsigned)(BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec);
		//printf("index %d, %dth flow, %llu us\n",index, j,diff / 1000);
		connection_create[index][j] = diff /1000;
#endif

	}
}

/* 
 * Requires:
 * inc is a valid incast object ans is not NULL
 * p is a valid incast pool and is not NULL
 *
 * Effects:
 * Finishes the incast event.
 */
static void
//finish_incast(struct incast *inc, struct incast_pool *p)
finish_incast(struct incast *inc)
{
	/* Assert that there are no connections left for this incast. */
	assert(inc->conn_head->next == inc->conn_head);

	/* Assert that the incast finished. */
	assert(inc->nflows == inc->completed);

	/* Remove this incast from the pool. */
	//remove_incast_list(inc);
	//p->nr_incasts--;

	if (verbose)
		printf("Finished receiving %lu bytes from each of the %d flows\n",
				inc->sru, inc->nflows);

	/* Stop Timing and Log. */
	XCLOCK_GETTIME(&inc->finish);
	double tot_time = get_secs(inc->finish) - get_secs(inc->start);
	double bandwidth = (inc->sru * inc->numreqs * inc->nflows *8.0)/(1024.0 * 1024.0 * 1024.0 * tot_time);
	printf("- [%.9g, %lu, %.9g, %d]\n", tot_time, inc->sru * inc->numreqs *inc->nflows, bandwidth, inc->nflows);
	done[inc->core] = TRUE;
	/* Free the incast. */
	//free_incast(inc);
}

/* 
 * Requires:
 * inc is a valid incast object ans is not NULL
 *
 * Effects:
 * Increments the completed count for the incast.
 * Finishes the incast event if it is finished.
 */
	static void
increment_and_check_incast(struct incast *inc)
{
	inc->completed++;
	if (inc->completed == inc->nflows)
		finish_incast(inc);
}

/* 
 * Requires:
 * p should be an incast pool and not be NULL.
 *
 * Effects:
 * Initializes an empty incast pool. Allocates and initializes dummy list
 * heads.
 */
	static void 
init_pool(struct incast_pool *p) 
{
	/* Initially, there are no connected descriptors. */
	//p->nr_incasts = 0;                   

	/* Allocate and initialize the dummy connection head. */
	/*    p->incast_head = Malloc(sizeof(struct incast));
	      p->incast_head->next = p->incast_head;
	      p->incast_head->prev = p->incast_head;
	 */
	/* Init epoll. */
	p->efd = epoll_create1(0);
	if (p->efd < 0) {
		printf ("epoll_create error!\n");
		exit(1);
	}
}

/*******************************************************************************
 * Read and Write Messages.
 ******************************************************************************/

/* 
 * Requires:
 * p should be a connection pool and not be NULL.
 *
 * Effects:
 * Reads from each ready file descriptor in the read set and handles the
 * incoming messages appropriately.
 */
	static void
read_message(struct conn *c, struct incast_pool *p)
{
	int n;

	/* Read from that socket. */
	n = recv(c->fd, c->incast->read_buf, READ_BUF_SIZE, 0);

	/* Data read. */
	if (n > 0) {

		c->read_bytes += n;
		if (verbose) {
			printf("Read %d bytes from fd %d:\n", n, c->fd);
		}

		if (c->read_bytes >= c->request_bytes) {
			/* We have finished the request. */
			if (verbose) {
				printf("Finished reading %lu bytes from fd %d:\n", c->read_bytes, c->fd);
			}
			//remove_server(c, p);
			remove_connection(c);
		}
	}
	/* Error (possibly). */
	else if (n < 0) {
		/* If errno is EAGAIN, it just means we need to read again. */
		if (errno != EAGAIN) {
			fprintf(stderr, "Unable to read response from fd %d!\n",
					c->fd);
			//remove_server(c, p);
			remove_connection(c);
		}
	}
	/* Connection closed by server. */
	else {
		//remove_server(c, p);
		remove_connection(c);
	}
}

/* 
 * Requires:
 * p should be a connection pool and not be NULL.
 *
 * Effects:
 * Writes the appropriate messages to each ready file descriptor in the write set.
 */
	static void
write_message(struct conn *c, struct incast_pool *p)
{
	int n;
	uint64_t sru, net_sru;
	struct epoll_event event;

//	printf("write_message %"PRIu64"\n", c->incast->sru);
	/* Perform the write system call. */
	sru = c->incast->sru;
	net_sru = htobe64(sru);
	n = write(c->fd, &net_sru, sizeof(net_sru));

	/* Data written. */
	if (n == sizeof(sru)) {
		if (verbose)
			printf("Finished requesting %lu bytes from fd %d.\n",
					sru, c->fd);
		c->request_bytes = sru;
		/* The request size has been received, update epoll.
		 * Remove it from the write set and add it to the 
		 * read set. */
		event.data.fd = c->fd;
		event.data.ptr = c;
		event.events = EPOLLIN;
		XEPOLL_CTL(p->efd, EPOLL_CTL_MOD, c->fd, &event);
	}
	/* Error (possibly). */
	else if (n < 0) {
		/* If errno is EAGAIN, it just means we have to write again. */
		if (errno != EAGAIN) {
			fprintf(stderr, "Unable to write request to fd %d!\n",
					c->fd);
			remove_server(c, p);
		}
	}
	/* Connection closed by server. */
	else {
		fprintf(stderr, "Unable to write request to fd %d!\n",
				c->fd);
		remove_server(c, p);	
	}
}

void parseOpt(int argc, char **argv){
        int c;
        int i=0;
        while ((c = getopt(argc, argv, "s:m:n:c:b:h:p:e:d:M:S:R:f:")) != -1) {
                switch (c)
                {
                        case 'm':
                                send_message_size = atoi(optarg);
                                if(send_message_size > READ_BUF_SIZE ){
                                        printf("send_message size is larger than receive buffer size %d\n", READ_BUF_SIZE);
                                        exit(-1);
                                }
                                //printf("message_size: %d\n", send_message_size);
                                break;
                        case 'n':
                                num_send_request = atoi(optarg);
                                break;
                                /*                      case 'i':
                                                        cpu_id = atoi(optarg);
                                                        printf("cpu id @ parseOpt: %d\n", cpu_id);
                                                        break;*/
                        case 'R':
                                rate_limit_mode = atoi(optarg);
                                break;
                        case 'p':
                                portno = atoi(optarg);
                                break;
                        case 'h':
                                {
                                        char *hostnames = strtok(optarg,",");
                                        while(hostnames){
                                                if(i>=MAX_CPUS)
                                                        die("the number of clients is larger than the max");
						hosts[i++]= hostnames;
                                                hostnames=strtok(NULL, ",");
                                        }
                                }
                                break;
                        case 'c':
                                is_client = atoi(optarg);
                                break;
                        case 'd':
                                queue_depth = atoi(optarg);
                                break;
                        case 'S':
                                is_sync = atoi(optarg);
                                break;
                        case 'b':
                                internal_port_no = atoi(optarg);
                                break;
                        case 's':
                                signaled = atoi(optarg);
                                break;
                        case 'e':
                                event_mode = atoi(optarg);
                                //printf("enable event mode\n");
                                break;
                        case 'D':
                                duration = atoi(optarg);
                                break;
                        case 'M':
                                num_threads = atoi(optarg);
				if(num_threads > MAX_CPUS){
					num_threads = MAX_CPUS;
					fprintf(stderr,"num_threads is set to %d\n", num_threads);
				}
					
                                break;
                        case 'f':
                                total_flows = atoi(optarg);
                                break;
                        default:
                                fprintf(stderr, "usage: %s -m<message_size> -n<num_send_request> -s<signaled>  -e<event_mode> -h<host_ip> -c<is_client> -p<port_no> -M<num_threads> -b<internal_port_no> -R<rate_limit_mode>\n", argv[0]);
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
       /* if(is_client == 1 && num_threads > i){
                printf("thread number is larger than server number!!\n");
                exit(-1);
        }*/
}

void *RunMain(void *arg){
	struct incast_pool pool;
	struct conn *connp;
	int i;
        int index = (int) arg;

	bindingCPU(index);
	/* Initialize the connection pool. */
	init_pool(&pool);

	//printf("flows %d on core %d\n",flows[index], index);
	/* Start an incast. */
	start_incast(send_message_size, flows[index], hosts[0], portno+index, &pool, index);
	if(get_nsecs(main_start) == 0)
		XCLOCK_GETTIME(&main_start);



	//printf("- [totTime, totBytes, bandwidthGbps, numSockets]\n");
	while(!done[index])
	{ 
		/* 
		 * Wait until:
		 * 1. Socket is ready for request to be written.
		 * 2. Data is available to be read from a socket. 
		 */
		pool.nevents = epoll_wait (pool.efd, pool.events, MAXEVENTS, 0);
		for (i = 0; i < pool.nevents; i++) {
			if ((pool.events[i].events & EPOLLERR) ||
					(pool.events[i].events & EPOLLHUP)) {
				/* An error has occured on this fd */
				fprintf (stderr, "epoll error\n");
				perror("");
				close (pool.events[i].data.fd);
				exit(-1);
			}

			/* Handle Reads. */
			if (pool.events[i].events & EPOLLIN) {
				connp = (struct conn *) pool.events[i].data.ptr;
				read_message(connp, &pool);
			}

			/* Handle Writes. */
			if (pool.events[i].events & EPOLLOUT) {
				connp = (struct conn *) pool.events[i].data.ptr;
				write_message(connp, &pool);
			}
		}
	}

	return; 
}

void PrintConnection(){
        int core=0, i=0;
        printf("connection:\n");
        for(core=0; core< num_threads; core++){
                for (i=0;i<flows[core];i++){
                        //if(connection_create[core][i] == 0)
                        //      break;
                        printf("- [%d, %llu]\n", core, connection_create[core][i]);
                }
        }


}



int main(int argc, char **argv) 
{
	uint64_t sru;
		int i;

	/* initialize random() */
	srandom(time(0));
        parseOpt(argc,argv);
	hp = gethostbyname(hosts[0]);
	/* Fill in the server's IP address and port */
	if (hp == NULL || hp->h_addr_list[0] == NULL){
		
		printf("hp is null\n");
		//return -2; /* check h_errno for cause of error */
		exit(-1);
	}

	//printf("latency:\n");
        int flow_per_thread = total_flows / num_threads;
        int flow_remainder_cnt = total_flows % num_threads;
        for (i = 0; i < num_threads; i++) {
                done[i] = FALSE;
                flows[i] = flow_per_thread;

                if (flow_remainder_cnt-- > 0)
                        flows[i]++;

                if (flows[i] == 0)
                        continue;

                if (pthread_create(&latency_threads[i],
                                        NULL, RunMain, (void *)i)) {
                        perror("pthread_create");
                        exit(-1);
                }
	}


        
        /* Waiting for completion */
        for( i= 0; i < num_threads; i++)
                if(pthread_join(latency_threads[i], NULL) !=0 )
                        die("main(): Join failed for worker thread i");

	/* Stop Timing and Log. */
	XCLOCK_GETTIME(&main_end);
	double tot_time = get_secs(main_end) - get_secs(main_start);
	long long unsigned bytes=(long long unsigned) send_message_size * num_send_request * total_flows; 
	double bandwidth =(double) (bytes*8.0)/(1024.0 * 1024.0 * 1024.0 * tot_time);
	printf("- [%.9g, %lu, %.9g, %d]\n", tot_time, bytes, bandwidth, total_flows);

	for(i=0;i<num_threads;i++)
		free(multi_incasts[i]);
	
//	PrintConnection();	

}
