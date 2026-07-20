#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <string.h>
#include <assert.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "common.h"

///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#define TCP_PORT_OFFSET 23456
#define TCP_PORT_RANGE 1000

#define SINGLE_QUEUE_SIZE 10

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

__device__ int arr_min(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int rhs, lhs;

    for (int stride = 1; stride < arr_size; stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            rhs = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            lhs = arr[tid];
            if (rhs != 0) {
                if (lhs == 0)
                    arr[tid] = rhs;
                else
                    arr[tid] = min(arr[tid], rhs);
            }
        }
        __syncthreads();
    }

    int ret = arr[arr_size - 1];
    return ret;
}

__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;

    for (int stride = 1; stride < min(blockDim.x, arr_size); stride *= 2) {
        if (tid >= stride && tid < arr_size) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < arr_size) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__device__ void gpu_process_image_device(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}


__global__ void gpu_process_image(uchar *in, uchar *out) {
    __shared__ int histogram[256];
    __shared__ int hist_min[256];

    int tid = threadIdx.x;

    if (tid < 256) {
        histogram[tid] = 0;
    }
    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x)
        atomicAdd(&histogram[in[i]], 1);

    __syncthreads();

    prefix_sum(histogram, 256);

    if (tid < 256) {
        hist_min[tid] = histogram[tid];
    }
    __syncthreads();

    int cdf_min = arr_min(hist_min, 256);

    __shared__ uchar map[256];
    if (tid < 256) {
        int map_value = (float)(histogram[tid] - cdf_min) / (SQR(IMG_DIMENSION) - cdf_min) * 255;
        map[tid] = (uchar)map_value;
    }

    __syncthreads();

    for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x) {
        out[i] = map[in[i]];
    }
    return;
}

/* TODO: copy queue-based GPU kernel from hw2 */

/////////////     GPU     /////////////////
__device__ void dequeue_request(volatile uchar* queue, volatile int* flags, 
    uchar* image_out, int* image_id){
    int tid = threadIdx.x;
    __shared__ int index;
    if(tid==0){
    index = -1;
        for(int i=0; i<SINGLE_QUEUE_SIZE; i++){
            if(flags[i] != -1){
                index = i;
                *image_id = flags[i];
                break;
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if(index != -1){
      for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x){
              image_out[i] = queue[index*SQR(IMG_DIMENSION)+i];
      }
    }
    __syncthreads();
    __threadfence_system();
    if(tid==0 && index != -1){
        flags[index] = -1;   
    }
    __threadfence_system();
}

__device__ void enqueue_response(volatile uchar* queue, volatile int* flags, 
    uchar* image_out, int image_id){
    int tid = threadIdx.x;
    __shared__ int index;
    if(tid==0){
    index = -1;
        for(int i=0; index==-1; i=(i+1)%SINGLE_QUEUE_SIZE){
            if(flags[i] == -1){
                index = i;
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if(index != -1){
      for (int i = tid; i < SQR(IMG_DIMENSION); i += blockDim.x){
               queue[index*SQR(IMG_DIMENSION)+i] = image_out[i] ;
      }
    }
    __syncthreads();
    __threadfence_system();
    if(tid==0 && index != -1){
        flags[index] = image_id;   
    }
    __syncthreads();
    __threadfence_system();
}


__global__ void test_kernel(volatile uchar* cpu_gpu_queue, volatile int* cpu_gpu_flags, 
    volatile uchar* gpu_cpu_queue, volatile int* gpu_cpu_flags, volatile int* running){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int num_blocks = gridDim.x;
    __shared__ uchar image_in[SQR(IMG_DIMENSION)];
    __shared__ uchar image_out[SQR(IMG_DIMENSION)];
    __shared__ int image_id;
    __shared__ int queue_index;
    __shared__ int flags_index;
    __shared__ bool started;
    if(tid==0){
        started = false;
        queue_index = bid * SINGLE_QUEUE_SIZE * SQR(IMG_DIMENSION);
        flags_index = bid * SINGLE_QUEUE_SIZE;
    }
    __syncthreads();
    __threadfence_system();
    while(*running < num_blocks+1){
        if(tid==0){
            if(!started){
                started = true;
                atomicAdd((int*)running, 1);
            }
            image_id = -1;
        }
        __syncthreads();
        __threadfence_system();
        dequeue_request(cpu_gpu_queue + queue_index, 
            cpu_gpu_flags + flags_index, (uchar*)image_in, &image_id);
        __syncthreads();
        __threadfence_system();
        if(image_id != -1){
            gpu_process_image_device((uchar*)image_in, (uchar*)image_out);
        }
        __syncthreads();
        __threadfence_system();
        if(image_id != -1){
            enqueue_response(gpu_cpu_queue + queue_index, gpu_cpu_flags + flags_index, 
                (uchar*)image_out, image_id);
        }
        __syncthreads();
        __threadfence_system();
    }
}

/* TODO: end */

void process_image_on_gpu(uchar *img_in, uchar *img_out)
{
    uchar *gpu_image_in, *gpu_image_out;
    CUDA_CHECK(cudaMalloc(&gpu_image_in, SQR(IMG_DIMENSION)));
    CUDA_CHECK(cudaMalloc(&gpu_image_out, SQR(IMG_DIMENSION)));

    CUDA_CHECK(cudaMemcpy(gpu_image_in, img_in, SQR(IMG_DIMENSION), cudaMemcpyHostToDevice));
    gpu_process_image<<<1, 1024>>>(gpu_image_in, gpu_image_out);
    CUDA_CHECK(cudaMemcpy(img_out, gpu_image_out, SQR(IMG_DIMENSION), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(gpu_image_in));
    CUDA_CHECK(cudaFree(gpu_image_out));
}

void print_usage_and_die(char *progname) {
    printf("usage: [port]\n");
    exit(1);
}

struct server_context {
    mode_enum mode;

    int tcp_port;
    int listen_fd; /* Listening socket for TCP connection */
    int socket_fd; /* Connected socket for TCP connection */

    rpc_request *requests; /* Array of outstanding requests received from the network */
    uchar *images_in; /* Input images for all outstanding requests */
    uchar *images_out; /* Output images for all outstanding requests */

    /* InfiniBand/verbs resources */
    struct ibv_context *context;
    struct ibv_cq *cq;
    struct ibv_pd *pd;
    struct ibv_qp *qp;
    struct ibv_mr *mr_requests; /* Memory region for RPC requests */
    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
	
    /* TODO: add pointers and memory region(s) for CPU-GPU queues */
	  volatile uchar *cpu_gpu_queue;
    volatile uchar *gpu_cpu_queue; 
	  volatile int *cpu_gpu_flags; 
    volatile int *gpu_cpu_flags; 
	  volatile int *running;
	
    struct ibv_mr *mr_cpu_gpu_queue; 
    struct ibv_mr *mr_gpu_cpu_queue; 
	  struct ibv_mr *mr_cpu_gpu_flags; 
    struct ibv_mr *mr_gpu_cpu_flags; 
	  struct ibv_mr *mr_running; 
    
};

int max_thread_blocks(int threads_num){
	struct cudaDeviceProp devProp;
	CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0));
	int regs_per_thread = 32;
	int threads_per_threadblock = threads_num;
	int shared_mem_per_threadblock = sizeof(uchar)*SINGLE_QUEUE_SIZE*SQR(IMG_DIMENSION);
  int bound1 = devProp.sharedMemPerMultiprocessor/shared_mem_per_threadblock;
  int bound2 =  devProp.sharedMemPerMultiprocessor/shared_mem_per_threadblock;
  int bound3 = devProp.regsPerMultiprocessor/regs_per_thread/threads_per_threadblock;
  int tmp = bound1 < bound2 ? bound1 : bound2;
  int min = tmp < bound3 ? tmp : bound3;
	int max_threadblocks = devProp.multiProcessorCount * min;
  return max_threadblocks;
}

void allocate_memory(server_context *ctx)
{
    CUDA_CHECK(cudaHostAlloc(&ctx->images_in, OUTSTANDING_REQUESTS * SQR(IMG_DIMENSION), 0));
    CUDA_CHECK(cudaHostAlloc(&ctx->images_out, OUTSTANDING_REQUESTS * SQR(IMG_DIMENSION), 0));
    ctx->requests = (rpc_request *)calloc(OUTSTANDING_REQUESTS, sizeof(rpc_request));

    /* TODO take CPU-GPU stream allocation code from hw2 */
	
	int num_blocks = max_thread_blocks(1024);
        
        size_t queue_size_bytes = num_blocks * SINGLE_QUEUE_SIZE * SQR(IMG_DIMENSION) * sizeof(uchar);
        size_t flags_size_bytes = num_blocks * SINGLE_QUEUE_SIZE * sizeof(int);
        CUDA_CHECK(cudaHostAlloc(&ctx->running, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&ctx->cpu_gpu_queue, queue_size_bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&ctx->gpu_cpu_queue, queue_size_bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&ctx->cpu_gpu_flags, flags_size_bytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&ctx->gpu_cpu_flags, flags_size_bytes, cudaHostAllocMapped));

	  *(ctx->running) = 0;
    for(int i=0; i<num_blocks * SINGLE_QUEUE_SIZE; i++){
            ctx->cpu_gpu_flags[i] = -1;
            ctx->gpu_cpu_flags[i] = -1;
        }
    __sync_synchronize();
}

void tcp_connection(server_context *ctx)
{
    /* setup a TCP connection for initial negotiation with client */
    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        perror("socket");
        exit(1);
    }
    ctx->listen_fd = lfd;

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(ctx->tcp_port);

    if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        perror("bind");
        exit(1);
    }

    if (listen(lfd, 1)) {
        perror("listen");
        exit(1);
    }

    printf("Server waiting on port %d. Client can connect\n", ctx->tcp_port);

    int sfd = accept(lfd, NULL, NULL);
    if (sfd < 0) {
        perror("accept");
        exit(1);
    }
    printf("client connected\n");
    ctx->socket_fd = sfd;
}

void initialize_verbs(server_context *ctx)
{
    /* get device list */
    struct ibv_device **device_list = ibv_get_device_list(NULL);
    if (!device_list) {
        printf("ERROR: ibv_get_device_list failed\n");
        exit(1);
    }

    /* select first (and only) device to work with */
    ctx->context = ibv_open_device(device_list[0]);

    /* create protection domain (PD) */
    ctx->pd = ibv_alloc_pd(ctx->context);
    if (!ctx->pd) {
        printf("ERROR: ibv_alloc_pd() failed\n");
        exit(1);
    }

    
    ctx->mr_requests = ibv_reg_mr(ctx->pd, ctx->requests, sizeof(rpc_request) * OUTSTANDING_REQUESTS, IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_requests) {
        printf("ibv_reg_mr() failed for requests\n");
        exit(1);
    }

    
    ctx->mr_images_in = ibv_reg_mr(ctx->pd, ctx->images_in, OUTSTANDING_REQUESTS * SQR(IMG_DIMENSION), IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_images_in) {
        printf("ibv_reg_mr() failed for input images\n");
        exit(1);
    }

    
    ctx->mr_images_out = ibv_reg_mr(ctx->pd, ctx->images_out, OUTSTANDING_REQUESTS * SQR(IMG_DIMENSION), IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_images_out) {
        printf("ibv_reg_mr() failed for output images\n");
        exit(1);
    }

    /* TODO register additional memory regions for CPU-GPU queues */
	
	int thread_blocks_num = max_thread_blocks(1024);
 

    ctx->mr_running = ibv_reg_mr(ctx->pd, (void*)ctx->running, 
		sizeof(int),
		IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_running) {
        printf("ibv_reg_mr() failed for running\n");
        exit(1);
    }
	
	ctx->mr_cpu_gpu_queue = ibv_reg_mr(ctx->pd, (void*)ctx->cpu_gpu_queue, 
	sizeof(uchar) * SQR(IMG_DIMENSION) * SINGLE_QUEUE_SIZE * thread_blocks_num,
	IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_cpu_gpu_queue) {
        printf("ibv_reg_mr() failed for cpu_gpu_queue\n");
        exit(1);
    }
    
    
    ctx->mr_cpu_gpu_flags = ibv_reg_mr(ctx->pd, (void*)ctx->cpu_gpu_flags, 
	sizeof(int) * SINGLE_QUEUE_SIZE * thread_blocks_num, 
	IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_cpu_gpu_flags) {
        printf("ibv_reg_mr() failed for cpu_gpu_flags\n");
        exit(1);
    }
    
    
    ctx->mr_gpu_cpu_queue = ibv_reg_mr(ctx->pd, (void*)ctx->gpu_cpu_queue, 
	 sizeof(uchar) * SQR(IMG_DIMENSION) * SINGLE_QUEUE_SIZE * thread_blocks_num,
	 IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_gpu_cpu_queue) {
        printf("ibv_reg_mr() failed for gpu_cpu_queue\n");
        exit(1);
    }
	
    
    ctx->mr_gpu_cpu_flags = ibv_reg_mr(ctx->pd, (void*)ctx->gpu_cpu_flags, 
		sizeof(int) * SINGLE_QUEUE_SIZE * thread_blocks_num,
		IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->mr_gpu_cpu_flags) {
        printf("ibv_reg_mr() failed for gpu_cpu_flags\n");
        exit(1);
    }
    
    /* create completion queue (CQ). We'll use same CQ for both send and receive parts of the QP */
    ctx->cq = ibv_create_cq(ctx->context, 2 * OUTSTANDING_REQUESTS, NULL, NULL, 0); /* create a CQ with place for two completions per request */
    if (!ctx->cq) {
        printf("ERROR: ibv_create_cq() failed\n");
        exit(1);
    }

    /* create QP */
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.send_cq = ctx->cq;
    qp_init_attr.recv_cq = ctx->cq;
    qp_init_attr.qp_type = IBV_QPT_RC; /* we'll use RC transport service, which supports RDMA */
    qp_init_attr.cap.max_send_wr = OUTSTANDING_REQUESTS; /* max of 1 WQE in-flight in SQ per request. that's enough for us */
    qp_init_attr.cap.max_recv_wr = OUTSTANDING_REQUESTS; /* max of 1 WQE in-flight in RQ per request. that's enough for us */
    qp_init_attr.cap.max_send_sge = 1; /* 1 SGE in each send WQE */
    qp_init_attr.cap.max_recv_sge = 1; /* 1 SGE in each recv WQE */
    ctx->qp = ibv_create_qp(ctx->pd, &qp_init_attr);
    if (!ctx->qp) {
        printf("ERROR: ibv_create_qp() failed\n");
        exit(1);
    }
}

void exchange_parameters(server_context *ctx, ib_info_t *client_info)
{
    /* ok, before we continue we need to get info about the client' QP, and send it info about ours.
     * namely: QP number, and LID.
     * we'll use the TCP socket for that */

    /* first query port for its LID (L2 address) */
    int ret;
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(ctx->context, IB_PORT_SERVER, &port_attr);
    if (ret) {
        printf("ERROR: ibv_query_port() failed\n");
        exit(1);
    }

    /* now send our info to client */
    struct ib_info_t my_info;
    my_info.lid = port_attr.lid;
    my_info.qpn = ctx->qp->qp_num;
    /* TODO add additional server rkeys / addresses here if needed */
	 
    my_info.blocks_num = max_thread_blocks(1024);
  	my_info.rkey_running = (int)ctx->mr_running->rkey;
	  my_info.addr_running = (uint64_t)ctx->mr_running->addr;
    
	  my_info.rkey_gpu_cpu_queue = (int)ctx->mr_gpu_cpu_queue->rkey;
    my_info.addr_gpu_cpu_queue = (uint64_t)ctx->mr_gpu_cpu_queue->addr;
    my_info.rkey_cpu_gpu_queue = (int)ctx->mr_cpu_gpu_queue->rkey;
    my_info.addr_cpu_gpu_queue = (uint64_t)ctx->mr_cpu_gpu_queue->addr;
	
	  my_info.rkey_gpu_cpu_flags = (int)ctx->mr_gpu_cpu_flags->rkey;
    my_info.addr_gpu_cpu_flags = (uint64_t)ctx->mr_gpu_cpu_flags->addr;
    my_info.rkey_cpu_gpu_flags = (int)ctx->mr_cpu_gpu_flags->rkey;
    my_info.addr_cpu_gpu_flags = (uint64_t)ctx->mr_cpu_gpu_flags->addr;

    ret = send(ctx->socket_fd, &my_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("send");
        exit(1);
    }

    /* get client's info */
    recv(ctx->socket_fd, client_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("recv");
        exit(1);
    }

    /* we don't need TCP anymore. kill the socket */
    close(ctx->socket_fd);
    close(ctx->listen_fd);
    ctx->socket_fd = ctx->listen_fd = 0;
}

/* Post a receive buffer of the given index (from the requests array) to the receive queue */
void post_recv(server_context *ctx, int index)
{
    struct ibv_recv_wr recv_wr = {}; /* this is the receive work request (the verb's representation for receive WQE) */
    ibv_sge sgl;

    recv_wr.wr_id = index;
    sgl.addr = (uintptr_t)&ctx->requests[index];
    sgl.length = sizeof(ctx->requests[0]);
    sgl.lkey = ctx->mr_requests->lkey;
    recv_wr.sg_list = &sgl;
    recv_wr.num_sge = 1;
    if (ibv_post_recv(ctx->qp, &recv_wr, NULL)) {
        printf("ERROR: ibv_post_recv() failed\n");
        exit(1);
    }
}

void connect_qp(server_context *ctx, ib_info_t *client_info)
{
    /* this is a multi-phase process, moving the state machine of the QP step by step
     * until we are ready */
    struct ibv_qp_attr qp_attr;

    /*QP state: RESET -> INIT */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = IB_PORT_SERVER;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ; /* we'll allow client to RDMA write and read on this QP */
    int ret = ibv_modify_qp(ctx->qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to INIT failed\n");
        exit(1);
    }

    /*QP: state: INIT -> RTR (Ready to Receive) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = client_info->qpn; /* qp number of client */
    qp_attr.rq_psn      = 0 ;
    qp_attr.max_dest_rd_atomic = 1; /* max in-flight RDMA reads */
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 0; /* No Network Layer (L3) */
    qp_attr.ah_attr.dlid = client_info->lid; /* LID (L2 Address) of client */
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = IB_PORT_SERVER;
    ret = ibv_modify_qp(ctx->qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to RTR failed\n");
        exit(1);
    }

    /*QP: state: RTR -> RTS (Ready to Send) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.max_rd_atomic = 1;
    ret = ibv_modify_qp(ctx->qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to RTS failed\n");
        exit(1);
    }

    /* now let's populate the receive QP with recv WQEs */
    for (int i = 0; i < OUTSTANDING_REQUESTS; i++) {
        post_recv(ctx, i);
    }
}

void event_loop(server_context *ctx)
{
    /* so the protocol goes like this:
     * 1. we'll wait for a CQE indicating that we got an Send request from the client.
     *    this tells us we have new work to do. The wr_id we used in post_recv tells us
     *    where the request is.
     * 2. now we send an RDMA Read to the client to retrieve the request.
     *    we will get a completion indicating the read has completed.
     * 3. we process the request on the GPU.
     * 4. upon completion, we send an RDMA Write with immediate to the client with
     *    the results.
     */

    struct ibv_send_wr send_wr;
    struct ibv_send_wr *bad_send_wr;
    rpc_request* req;
    uchar *img_in;
    uchar *img_out;
    ibv_sge sgl;

    bool terminate = false;

    while (!terminate) {
        /*step 1: poll for CQE */
        struct ibv_wc wc;
        int ncqes;
        do {
            ncqes = ibv_poll_cq(ctx->cq, 1, &wc);
        } while (ncqes == 0);
        if (ncqes < 0) {
            printf("ERROR: ibv_poll_cq() failed\n");
            exit(1);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            printf("ERROR: got CQE with error '%s' (%d) (line %d)\n", ibv_wc_status_str(wc.status), wc.status, __LINE__);
            exit(1);
        }

        switch (wc.opcode) {
        case IBV_WC_RECV:
            /* Received a new request from the client */
            req = &ctx->requests[wc.wr_id];
            img_in = &ctx->images_in[wc.wr_id * SQR(IMG_DIMENSION)];

            /* Terminate signal */
            if (req->request_id == -1) {
                printf("Terminating...\n");
                terminate = true;
                break;
            }

            if (ctx->mode != MODE_RPC_SERVER) {
                printf("Got client RPC request when running in queue mode.\n");
                exit(1);
            }
            
            /* send RDMA Read to client to read the input */
            memset(&send_wr, 0, sizeof(struct ibv_send_wr));
            send_wr.wr_id = wc.wr_id;
            sgl.addr = (uintptr_t)img_in;
            sgl.length = req->input_length;
            sgl.lkey = ctx->mr_images_in->lkey;
            send_wr.sg_list = &sgl;
            send_wr.num_sge = 1;
            send_wr.opcode = IBV_WR_RDMA_READ;
            send_wr.send_flags = IBV_SEND_SIGNALED;
            send_wr.wr.rdma.remote_addr = req->input_addr;
            send_wr.wr.rdma.rkey = req->input_rkey;

            if (ibv_post_send(ctx->qp, &send_wr, &bad_send_wr)) {
                printf("ERROR: ibv_post_send() failed\n");
                exit(1);
            }
            break;

        case IBV_WC_RDMA_READ:
            /* Completed RDMA read for a request */
            req = &ctx->requests[wc.wr_id];
            img_in = &ctx->images_in[wc.wr_id * SQR(IMG_DIMENSION)];
            img_out = &ctx->images_out[wc.wr_id * SQR(IMG_DIMENSION)];

            process_image_on_gpu(img_in, img_out);
            
            /* send RDMA Write with immediate to client with the response */
            memset(&send_wr, 0, sizeof(struct ibv_send_wr));
            send_wr.wr_id = wc.wr_id;
            ibv_sge sgl;
            sgl.addr = (uintptr_t)img_out;
            sgl.length = req->output_length;
            sgl.lkey = ctx->mr_images_out->lkey;
            send_wr.sg_list = &sgl;
            send_wr.num_sge = 1;
            send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            send_wr.send_flags = IBV_SEND_SIGNALED;
            send_wr.wr.rdma.remote_addr = req->output_addr;
            send_wr.wr.rdma.rkey = req->output_rkey;
            send_wr.imm_data = req->request_id;

            if (ibv_post_send(ctx->qp, &send_wr, &bad_send_wr)) {
                printf("ERROR: ibv_post_send() failed\n");
                exit(1);
            }
            break;

        case IBV_WC_RDMA_WRITE:
            /* Completed RDMA Write - reuse buffers for receiving the next requests */
            post_recv(ctx, wc.wr_id);

            break;
        default:
            printf("Unexpected completion\n");
            assert(false);
        }
    }
}

void teardown_context(server_context *ctx)
{
    /* cleanup */
    ibv_destroy_qp(ctx->qp);
    ibv_destroy_cq(ctx->cq);
    ibv_dereg_mr(ctx->mr_requests);
    ibv_dereg_mr(ctx->mr_images_in);
    ibv_dereg_mr(ctx->mr_images_out);
    
	/* TODO destroy the additional server MRs here if needed */
    CUDA_CHECK(cudaFreeHost((void *)ctx->cpu_gpu_queue));
    CUDA_CHECK(cudaFreeHost((void *)ctx->gpu_cpu_queue));
	  CUDA_CHECK(cudaFreeHost((void *)ctx->cpu_gpu_flags));
    CUDA_CHECK(cudaFreeHost((void *)ctx->gpu_cpu_flags));
    CUDA_CHECK(cudaFreeHost((void *)ctx->running));
	
	  ibv_dereg_mr(ctx->mr_cpu_gpu_queue);
    ibv_dereg_mr(ctx->mr_gpu_cpu_queue);
	  ibv_dereg_mr(ctx->mr_cpu_gpu_flags);
    ibv_dereg_mr(ctx->mr_gpu_cpu_flags);
	  ibv_dereg_mr(ctx->mr_running);
	
    ibv_dealloc_pd(ctx->pd);
    ibv_close_device(ctx->context);
}

int main(int argc, char *argv[]) {
    server_context ctx;

    parse_arguments(argc, argv, &ctx.mode, &ctx.tcp_port);
    if (!ctx.tcp_port) {
        srand(time(NULL));
        ctx.tcp_port = TCP_PORT_OFFSET + (rand() % TCP_PORT_RANGE); /* to avoid conflicts with other users of the machine */
    }

    /* Initialize memory and CUDA resources */
    allocate_memory(&ctx);

    /* Create a TCP connection with the client to exchange InfiniBand parameters */
    tcp_connection(&ctx);

    /* now that client has connected to us via TCP we'll open up some Infiniband resources and send it the parameters */
    initialize_verbs(&ctx);

    /* exchange InfiniBand parameters with the client */
    ib_info_t client_info;
    exchange_parameters(&ctx, &client_info);

    /* now need to connect the QP to the client's QP. */
    connect_qp(&ctx, &client_info);

    if (ctx.mode == MODE_QUEUE) {
        /* TODO run the GPU persistent kernel from hw2, for 1024 threads per block */
		int thread_blocks_num = max_thread_blocks(1024);
        test_kernel<<<thread_blocks_num, 1024>>>(ctx.cpu_gpu_queue,
         ctx.cpu_gpu_flags, ctx.gpu_cpu_queue, ctx.gpu_cpu_flags, ctx.running);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* now finally we get to the actual work, in the event loop */
    /* The event loop can be used for queue mode for the termination message */
    event_loop(&ctx);

    printf("Done\n");

    teardown_context(&ctx);

    return 0;
}
