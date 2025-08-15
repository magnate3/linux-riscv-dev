#include "rdma_library.h"
#include "log.h"

#include <linux/kernel.h>
#include <rdma/ib_verbs.h>
#include <rdma/rdma_cm.h>
#include <linux/smp.h>
#include <linux/version.h>
#include <linux/cpufreq.h>

#define CHECK_MSG_RET(arg, msg, ret) {\
   if ((arg) == 0){\
      printk(KERN_INFO msg);\
      return (ret);\
   }\
}

#define CHECK_MSG_NO_RET(arg, msg) {\
   if ((arg) == 0){\
      printk(KERN_INFO msg);\
      return;\
   }\
}

extern int initd;


static struct ib_device_singleton {
    struct ib_device_attr mlnx_device_attr;
    struct ib_client ibclient;
    bool ready_to_use;
    struct ib_device* dev;
    struct ib_event_handler ieh;
    struct ib_port_attr attr;
    struct ib_gid_attr gid_attr;
    union ib_gid gid;

    int lid;
} rdma_ib_device;

batch_request_pool* get_batch_request_pool(int size)
{
    int i;
    struct batch_request_pool* pool;

    pool = vmalloc(sizeof(*pool));
    pool->size = size;
    pool->head = 0;
    pool->tail = size - 1;
    spin_lock_init(&pool->lock);
    pool->data = vmalloc(sizeof(struct batch_request*) * size);
    pool->all = vmalloc(sizeof(struct batch_request*) * size);

    pool->io_req = vmalloc(sizeof(struct request*) * 1024);
    memset(pool->io_req, 0, sizeof(struct request*) * 1024);

    for(i = 0; i < size; i++){
        pool->data[i] = vmalloc(sizeof(struct batch_request));
        pool->data[i]->id = i;
        pool->all[i] = pool->data[i];
    }

#if MEASURE_LATENCY
    memset(pool->latency_dist, 0, sizeof(pool->latency_dist));
#endif
    return pool;
}

void destroy_batch_request_pool(batch_request_pool* pool)
{
    int i;
    for (i = 0; i < pool->size; i++)
        if (pool->data[i]) vfree(pool->data[i]);
    vfree(pool->data);
    vfree(pool->io_req);
    vfree(pool);
}

batch_request* get_batch_request(batch_request_pool* pool)
{
    struct batch_request* ret = NULL;
    spin_lock_irq(&pool->lock);
    if(pool->head == pool->tail)
    {
        pr_err("Pool is almost empty");
        ret = NULL;
    }
    else
    {
        ret = pool->data[pool->head];
        BUG_ON(ret == NULL);
        pool->data[pool->head] = NULL;
        pool->head = (pool->head + 1) % pool->size;
#if MEASURE_LATENCY
        ret->start_time = get_cycle();
#endif
    //    LOG_KERN(LOG_INFO, "head %d tail %d", pool->head, pool->tail);
    }
    spin_unlock_irq(&pool->lock);
    return ret;
}

void return_batch_request(batch_request_pool* pool, batch_request* req)
{
    int new_tail,bucket;
    BUG_ON(req == NULL);
    spin_lock_irq(&pool->lock);
#if MEASURE_LATENCY
    if(req->first)
    {
      bucket = (get_cycle() - req->start_time)*1000/cpu_khz;
      LOG_KERN(LOG_INFO, "cycle %d, bucket %d", get_cycle() - req->start_time, bucket);
      pool->latency_dist[bucket>=LATENCY_BUCKET?LATENCY_BUCKET-1:bucket]++;
    }
#endif
    new_tail = (pool->tail + 1) % pool->size;
    if(new_tail == pool->head)
    {
        pr_err("Err: new_tail == head == %d", new_tail);
        return;
    }
    if(pool->data[new_tail])
    {
        pr_err("Err: New tail value %p", pool->data[new_tail]);
        return;
    }
    pool->data[new_tail] = req;
    pool->tail = new_tail;
    //LOG_KERN(LOG_INFO, "head %d tail %d", pool->head, pool->tail);
    spin_unlock_irq(&pool->lock);
}

void debug_pool_insert(struct batch_request_pool* pool, struct request* req)
{
    int i, free = -1;

    LOG_KERN(LOG_INFO, "insert req %p", req);
    spin_lock_irq(&pool->lock);
    for (i = 0; i < 1024; i++)
    {
        BUG_ON(pool->io_req[i] == req);
        if(pool->io_req[i] == 0)
        {
            free = i;
            break;
        }
    }
    BUG_ON(free == -1);
    LOG_KERN(LOG_INFO, "insert req %p at loc %d", req, free);
    pool->io_req[free] = req;
    spin_unlock_irq(&pool->lock);
}

void debug_pool_remove(struct batch_request_pool* pool, struct request* req)
{
    int i;

    spin_lock_irq(&pool->lock);
    for (i = 0; i < 1024; i++)
    {
        if(pool->io_req[i] == req)
        {
            LOG_KERN(LOG_INFO, "removing req %p at loc %d", req, i);
            pool->io_req[i] = NULL;
            spin_unlock_irq(&pool->lock);
            return;
        }
    }
    BUG();
}


static void async_event_handler(struct ib_event_handler* ieh, struct ib_event *ie)
{
    LOG_KERN(LOG_INFO, "async_event_handler", 0);
}

static int populate_port_data(rdma_ctx_t ctx)
{
    int retval;
    
    //LOG_KERN(LOG_INFO, ("Get port data\n"));
    retval = ib_query_port(rdma_ib_device.dev, 1, &rdma_ib_device.attr);
    CHECK_MSG_RET(retval == 0, "Error querying port", -1);
    CHECK_MSG_RET(rdma_ib_device.attr.active_mtu == 5, "Error: Wrong device", -1);

    rdma_ib_device.lid = rdma_ib_device.attr.lid;
    ib_query_gid(rdma_ib_device.dev, 1, 0, &rdma_ib_device.gid, &rdma_ib_device.gid_attr);

    return 0;
} 

static void add_device(struct ib_device* dev)
{
    // first check this is the mellanox device
    int retval = ib_query_port(dev, 1, &rdma_ib_device.attr);
    LOG_KERN(LOG_INFO, "add_device");
    CHECK_MSG_NO_RET(retval == 0, "Error querying port");

    if(rdma_ib_device.attr.active_mtu == 5) {
        rdma_ib_device.dev = dev;
    } else {
        return;
    }

    // get device attributes
    ib_query_device(dev, &rdma_ib_device.mlnx_device_attr); 

    // register handler
    INIT_IB_EVENT_HANDLER(&rdma_ib_device.ieh, dev, async_event_handler);
    ib_register_event_handler(&rdma_ib_device.ieh);

    // We got the right device
    // library can be used now
    LOG_KERN(LOG_INFO, "ready_to_use = ture");
    rdma_ib_device.ready_to_use = true;
}

static void remove_device(struct ib_device* dev)
{
    LOG_KERN(LOG_INFO, "remove_device");
}

static void init_ib(void)
{
    rdma_ib_device.ibclient.name = "FBOX_DISAG_MEM";
    rdma_ib_device.ibclient.add = add_device;
    rdma_ib_device.ibclient.remove = remove_device;
    ib_register_client(&rdma_ib_device.ibclient);
}

int rdma_library_ready(void)
{
    return rdma_ib_device.ready_to_use;
}

int rdma_library_init(void)
{
    memset(&rdma_ib_device, 0, sizeof(rdma_ib_device));
    init_ib();
    return 0;
}

int rdma_library_exit(void)
{
    ib_unregister_event_handler(&rdma_ib_device.ieh);
    ib_unregister_client(&rdma_ib_device.ibclient);
    return 0;
}

static int translate_ip(char* ip_addr, u8 IP[])
{
    int ret;
    //char sub_addr[4][4];
    int vals[4];

    LOG_KERN(LOG_INFO, "Translating ip: %s", ip_addr);

    ret = sscanf(ip_addr, "%d.%d.%d.%d", 
            &vals[0], &vals[1], &vals[2], &vals[3]);
    LOG_KERN(LOG_INFO, "ret: %d", ret);
    CHECK_MSG_RET(ret > 0,"Error translating ip_addr", -1);

    //for (i = 0; i < 4; ++i) {
    //    kstrtou8(sub_addr[i], 10, &IP[i]);
    //}

    IP[0] = vals[0];
    IP[1] = vals[1];
    IP[2] = vals[2];
    IP[3] = vals[3];

    return 0;
}

static u32 create_address(u8 *ip)
{
   u32 addr = 0;
   int i;
   for (i = 0; i < 4; ++i) {
      addr += ip[i];
      if (i==3) {
         break;
      }
      addr <<= 8;
   }
   return addr;
}



static int connect(rdma_ctx_t ctx, char* ip_addr, int port)
{
    int retval;
    struct sockaddr_in servaddr;
    u8 IP[4] = {};

    strcpy(ctx->server_addr, ip_addr);
    ctx->server_port = port;

    retval = translate_ip(ip_addr, IP);
    if (retval != 0)
        return -1;

    // create
    retval = sock_create(AF_INET, SOCK_STREAM, IPPROTO_TCP, &ctx->sock); 
    CHECK_MSG_RET(retval == 0, "Error creating socket", -1);

    // connect
    memset(&servaddr, 0, sizeof(servaddr));  
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = htonl(create_address(IP));

    CHECK_MSG_RET(ctx->sock != 0, "Error creating socket", -1);
    CHECK_MSG_RET(ctx->sock->ops->connect != NULL, "Connect not found", -1);

    LOG_KERN(LOG_INFO, "connecting to %s", ip_addr);
    retval = ctx->sock->ops->connect(ctx->sock, 
                           (struct sockaddr *)&servaddr, sizeof(servaddr), 0);
    LOG_KERN(LOG_INFO, "connected retval: %d", retval);
    CHECK_MSG_RET(retval == 0, "Error connecting", -1);

    return 0;
}

static int reconnect(rdma_ctx_t ctx)
{
    return connect(ctx, ctx->server_addr, ctx->server_port);
}

static int receive_data(rdma_ctx_t ctx, char* data, int size) {
    struct msghdr msg;
    struct iovec iov;
    int retval;
    mm_segment_t oldfs;
    
    iov.iov_base = data;
    iov.iov_len  = size;
    
    msg.msg_name = 0;
    msg.msg_namelen = 0;
#if LINUX_VERSION_CODE < KERNEL_VERSION(3,19,0)
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = NULL;
#else
    iov_iter_init(&msg.msg_iter, READ, &iov, 1, iov.iov_len); 
#endif
    msg.msg_controllen = 0;
    msg.msg_flags = 0;
    //msg.msg_iov->iov_base= data;
    //msg.msg_iov->iov_len = size;
    

    oldfs = get_fs();
    set_fs(KERNEL_DS);

    retval = sock_recvmsg(ctx->sock, &msg, size, 0);

    set_fs(oldfs);

    return 0;
}

static int send_data(rdma_ctx_t ctx, char* data, int size) {
    struct msghdr msg;
    struct iovec iov;
    int retval;
    mm_segment_t oldfs;
    
    printk(KERN_INFO "Exchanging data");

    iov.iov_base = data;
    iov.iov_len  = size;
    
    msg.msg_name     = 0;
    msg.msg_namelen  = 0;
#if LINUX_VERSION_CODE < KERNEL_VERSION(3,19,0)
    msg.msg_iov      = &iov;
    msg.msg_iovlen   = 1;
    msg.msg_control  = NULL;
#else
    iov_iter_init(&msg.msg_iter, WRITE, &iov, 1, iov.iov_len);
#endif
    msg.msg_controllen = 0;
    msg.msg_flags    = 0;
    //msg.msg_iov->iov_len = size;
    //msg.msg_iov->iov_base = data;

    printk(KERN_INFO "Sending data..");
    oldfs = get_fs();
    set_fs(KERNEL_DS);

#if LINUX_VERSION_CODE < KERNEL_VERSION(4,1,0) 
    retval = sock_sendmsg(ctx->sock, &msg, size);
#else
    retval = sock_sendmsg(ctx->sock, &msg);
#endif
    set_fs(oldfs);

    return 0;
}

static int handshake(rdma_ctx_t ctx)
{
    int retval;
    char data[500];
    unsigned long long int vaddr = 0;
    

    // first send mem size
    snprintf(data, 500, "%llu", ctx->rem_mem_size);
    printk(KERN_WARNING "Sending: %s", data);
    send_data(ctx, data, strlen(data));

    // receive handshake data from server
    retval = receive_data(ctx, data, 500);
    printk(KERN_WARNING "data received: %s", data);
    
    sscanf(data, "%016Lx:%u:%x:%x:%x", &ctx->rem_vaddr, &ctx->rem_rkey, 
           &ctx->rem_qpn, &ctx->rem_psn, &ctx->rem_lid);
    LOG_KERN(LOG_INFO, "rem_vaddr: %llu rem_rkey:%u rem_qpn:%d rem_psn:%d rem_lid:%d", 
           ctx->rem_vaddr, ctx->rem_rkey, ctx->rem_qpn, ctx->rem_psn, ctx->rem_lid);

    sprintf(data, "%016Lx:%u:%x:%x:%x", 
            vaddr, ctx->rkey, ctx->qpn, ctx->psn, ctx->lid);

    // send handshake data to server
    send_data(ctx, data, strlen(data));

    return 0;
}

static int teardown(rdma_ctx_t ctx)
{
    int retval;
    char data[500];

    snprintf(data, 500, "teardown");
    do
    {
        send_data(ctx, data, strlen(data));
        retval = receive_data(ctx, data, 500);
    }
    while(strncmp(data, "done", 4) != 0);
    LOG_KERN(LOG_INFO, "rdma connection terminated");
}

static int modify_qp(rdma_ctx_t ctx)
{
    int retval;
    struct ib_qp_attr attr;
    
    memset(&attr, 0, sizeof(attr));

    attr.qp_state = IB_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IB_ACCESS_REMOTE_WRITE  | 
                                IB_ACCESS_REMOTE_READ |
                                IB_ACCESS_REMOTE_ATOMIC;

    retval = ib_modify_qp(ctx->qp, &attr, IB_QP_STATE | 
                                               IB_QP_PKEY_INDEX | 
                                               IB_QP_PORT | 
                                               IB_QP_ACCESS_FLAGS);
    CHECK_MSG_RET(retval == 0, "Error moving to INIT", -1);

    LOG_KERN(LOG_INFO, "Preparing for RTR. mtu: %d rem_qpn: %d rem_psn: %d rem_lid: %d",
           rdma_ib_device.attr.active_mtu, ctx->rem_qpn, ctx->rem_psn, ctx->rem_lid);

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IB_QPS_RTR;
    attr.path_mtu = rdma_ib_device.attr.active_mtu;
    attr.dest_qp_num = ctx->rem_qpn;
    attr.rq_psn = ctx->rem_psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = ctx->rem_lid;
    attr.ah_attr.sl = 0; // service level
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
   
    retval = ib_modify_qp(ctx->qp, &attr,
		    IB_QP_STATE | 
                    IB_QP_AV | 
                    IB_QP_PATH_MTU | 
                    IB_QP_DEST_QPN | 
                    IB_QP_RQ_PSN | 
                    IB_QP_MAX_DEST_RD_ATOMIC |
                    IB_QP_MIN_RNR_TIMER);

    CHECK_MSG_RET(retval == 0, "Error moving to RTR", -1);

    attr.qp_state = IB_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 6;
    attr.sq_psn = ctx->psn;
    attr.max_rd_atomic = 1;
    
    retval = ib_modify_qp(ctx->qp, &attr, IB_QP_STATE | 
                                               IB_QP_TIMEOUT | 
                                               IB_QP_RETRY_CNT | 
                                               IB_QP_RNR_RETRY | 
                                               IB_QP_SQ_PSN | 
                                               IB_QP_MAX_QP_RD_ATOMIC);

    CHECK_MSG_RET(retval == 0, "Error moving to RTS", -1);
    return 0;
}

u64 rdma_map_address(void* addr, int length)
{
    u64 dma_addr;


    dma_addr = ib_dma_map_single(rdma_ib_device.dev, addr, length, DMA_BIDIRECTIONAL);
    if (ib_dma_mapping_error(rdma_ib_device.dev, dma_addr) != 0) {
        LOG_KERN(LOG_INFO, "Error mapping myaddr", 0);
        return 0; //error
    }

    return dma_addr;
}

void rdma_unmap_address(u64 addr, int length)
{
    ib_dma_unmap_single(rdma_ib_device.dev, addr, length, DMA_BIDIRECTIONAL);
    return;
}

static int rdma_setup(rdma_ctx_t ctx)
{
    // create receive buffer
    ctx->rdma_recv_buffer = kmalloc(RDMA_BUFFER_SIZE, GFP_KERNEL);
    CHECK_MSG_RET(ctx->rdma_recv_buffer != 0, "Error kmalloc", -1);

    // create memory region
    ctx->mr = ib_get_dma_mr(ctx->pd, IB_ACCESS_REMOTE_READ | 
                                     IB_ACCESS_REMOTE_WRITE | 
                                     IB_ACCESS_LOCAL_WRITE);
    CHECK_MSG_RET(ctx->mr != 0, "Error creating MR", -1);

    ctx->rkey = ctx->mr->rkey;

    // get dma_addr
    ctx->dma_addr = ib_dma_map_single(rdma_ib_device.dev, ctx->rdma_recv_buffer, 
            RDMA_BUFFER_SIZE, DMA_BIDIRECTIONAL);
    CHECK_MSG_RET(ib_dma_mapping_error(rdma_ib_device.dev, ctx->dma_addr) == 0,
            "Error ib_dma_map_single", -1);

    // modify QP until RTS
    modify_qp(ctx);
    return 0;
}


#if MODE == MODE_ASYNC || MODE == MODE_ONE
static void comp_handler_send(struct ib_cq* cq, void* cq_context)
{
    struct ib_wc wc;
    int ret;
    struct batch_request* batch_req, *curr;
    struct request* req;
    spinlock_t* queue_lock;
    rdma_ctx_t ctx = (rdma_ctx_t)cq_context;
 
    LOG_KERN(LOG_INFO, "COMP HANDLER pid %d, cpu %d", current->pid, smp_processor_id());

    do {
        while (ib_poll_cq(cq, 1, &wc)> 0) {
            if (wc.status == IB_WC_SUCCESS) {
                //LOG_KERN(LOG_INFO, "IB_WC_SUCCESS %p op: %s byte_len: %u",
                //            (struct batch_request*)wc.wr_id,
                //            wc.opcode == IB_WC_RDMA_READ ? "IB_WC_RDMA_READ" : 
                //            wc.opcode == IB_WC_RDMA_WRITE ? "IB_WC_RDMA_WRITE" :
                //            "other", (unsigned) wc.byte_len);
                batch_req = (struct batch_request*)wc.wr_id;
                //LOG_KERN(LOG_INFO, "id = %d outstanding_reqs = %d", batch_req->id, batch_req->outstanding_reqs);
#if MODE == MODE_ONE
                batch_req->comp_reqs++;
                if(batch_req->outstanding_reqs == batch_req->comp_reqs && batch_req->all_request_sent)
#else
                batch_req->outstanding_reqs--;
                if(batch_req->outstanding_reqs == 0)
#endif
                {
                    #if CUSTOM_MAKE_REQ_FN
                        LOG_KERN(LOG_INFO, "returning batch request %d", batch_req->id);
                        bio_endio(batch_req->bio, 0);
                        return_batch_request(ctx->pool, batch_req);
                    #else
                        for(curr = batch_req; curr != NULL; curr = (struct batch_request*)curr->next)
                        {
                            req = (struct request*)curr->req;
                            queue_lock = req->q->queue_lock;
                            LOG_KERN(LOG_INFO, "returning batch request %d", curr->id);
                            #if DEBUG_OUT_REQ
                            debug_pool_remove(ctx->pool, req);
                            #endif
                            spin_lock_irq(queue_lock);
                            __blk_end_request_all(req, 0);
                            spin_unlock_irq(queue_lock);
                            //__blk_end_request(curr->req, 0, curr->nsec);
                            return_batch_request(ctx->pool, curr);
                        }
                    #endif
                }
            } else {
                pr_err("FAILURE %d", wc.status);
                BUG();
            }
        }
        ret = ib_req_notify_cq(cq, IB_CQ_NEXT_COMP | IB_CQ_REPORT_MISSED_EVENTS);
        if (ret < 0) {
            pr_err("ib_req_notify_cq < 0, ret = %d", ret);
        }
    } while (ret > 0);

}


#elif (MODE == MODE_SYNC)
static void comp_handler_send(struct ib_cq* cq, void* cq_context)
{
    LOG_KERN(LOG_INFO, "OMG. This function is called........\n");
}


void poll_cq(rdma_ctx_t ctx)
{
    struct ib_wc wc[10];
    struct ib_cq* cq = ctx->send_cq;
    int count, i, bucket;

    LOG_KERN(LOG_INFO, "COMP HANDLER pid %d, cpu %d", current->pid, smp_processor_id());

    do {
        while ((count = ib_poll_cq(cq, 10, wc)) > 0) {
            #if SIMPLE_POLL
            # if MEASURE_LATENCY
            if(wc[0].wr_id)
            {
                bucket = (get_cycle() - (unsigned long long)wc[i].wr_id)/3000;
                ctx->pool->latency_dist[bucket>=LATENCY_BUCKET?LATENCY_BUCKET-1:bucket]++;  
            }                        
            # endif
            LOG_KERN(LOG_INFO, "poll %d reqs, out req %d", count, ctx->outstanding_requests);
            ctx->outstanding_requests-=count;
            #else
            for(i = 0; i < count; i++){
                if (wc[i].status == IB_WC_SUCCESS) {
                    LOG_KERN(LOG_INFO, "IB_WC_SUCCESS %llu op: %s byte_len: %u",
                            (unsigned long long)wc[i].wr_id,
                            wc[i].opcode == IB_WC_RDMA_READ ? "IB_WC_RDMA_READ" : 
                            wc[i].opcode == IB_WC_RDMA_WRITE ? "IB_WC_RDMA_WRITE" 
                               :"other", (unsigned) wc[i].byte_len);
                    #if MEASURE_LATENCY
                    if(wc[i].wr_id)
                    {
                        bucket = (get_cycle() - (unsigned long long)wc[i].wr_id)/3000;
                        ctx->pool->latency_dist[bucket>=LATENCY_BUCKET?LATENCY_BUCKET-1:bucket]++;  
                    }                        
                    #endif
                } else {
                    LOG_KERN(LOG_INFO, "FAILURE %d", wc[i].status);
                }
                ctx->outstanding_requests--;
            }
            #endif
        }
        /*ret = ib_req_notify_cq(cq, IB_CQ_NEXT_COMP | IB_CQ_REPORT_MISSED_EVENTS);
        if (ret < 0) {
            LOG_KERN(LOG_INFO, "ib_req_notify_cq < 0, ret = %d", ret);
            ctx->outstanding_requests = 0;
        }*/
    } while ( ctx->outstanding_requests > 0);

    LOG_KERN(LOG_INFO, "COMP HANDLER done");
}

#endif


void comp_handler_recv(struct ib_cq* cq, void* cq_context)
{
//    rdma_ctx_t ctx = (rdma_ctx_t)cq_context;
    LOG_KERN(LOG_INFO, "COMP HANDLER", 0);
}

void cq_event_handler_send(struct ib_event* ib_e, void* v)
{
    LOG_KERN(LOG_INFO, "CQ HANDLER", 0);
}

void cq_event_handler_recv(struct ib_event* ib_e, void* v)
{
    LOG_KERN(LOG_INFO, "CQ HANDLER", 0);
}

int rdma_exit(rdma_ctx_t ctx)
{
    int retval = reconnect(ctx);
    teardown(ctx);

    CHECK_MSG_RET(ctx->sock != 0, "Error releasing socket", -1);
    sock_release(ctx->sock);

    destroy_batch_request_pool(ctx->pool);

    memset(ctx, 0, sizeof(struct rdma_ctx));
    kfree(ctx);

    return 0;
}

rdma_ctx_t rdma_init(int npages, char* ip_addr, int port, int mem_pool_size)
{
    int retval;
    rdma_ctx_t ctx;

    LOG_KERN(LOG_INFO, "RDMA_INIT. ip_addr: %s port: %d npages: %d", ip_addr, port, npages);
    
    ctx = kmalloc(sizeof(struct rdma_ctx), GFP_KERNEL);
    if (!ctx)
        return NULL;

    memset(ctx, 0, sizeof(struct rdma_ctx));
    ctx->rem_mem_size = (unsigned long long)npages * (1024 * 4);

    if (!rdma_ib_device.ready_to_use) {
        LOG_KERN(LOG_INFO, "ERROR", 0);
    }

    ctx->pd = ib_alloc_pd(rdma_ib_device.dev);
    CHECK_MSG_RET(ctx->pd != 0, "Error creating pd", 0);
   
    // Note that we set the CQ context to our ctx structure 
    ctx->send_cq = ib_create_cq(rdma_ib_device.dev, 
            comp_handler_send, cq_event_handler_send, ctx, CQE_SIZE, 0);
    ctx->recv_cq = ib_create_cq(rdma_ib_device.dev, 
            comp_handler_recv, cq_event_handler_recv, ctx, CQE_SIZE, 0);
    CHECK_MSG_RET(ctx->send_cq != 0, "Error creating CQ", 0);
    CHECK_MSG_RET(ctx->recv_cq != 0, "Error creating CQ", 0);
    
    CHECK_MSG_RET(ib_req_notify_cq(ctx->recv_cq, IB_CQ_NEXT_COMP) == 0,
            "Error ib_req_notify_cq", 0);
    CHECK_MSG_RET(ib_req_notify_cq(ctx->send_cq, IB_CQ_NEXT_COMP) == 0,
            "Error ib_req_notify_cq", 0);

    // initialize qp_attr
    memset(&ctx->qp_attr, 0, sizeof(struct ib_qp_init_attr));
    ctx->qp_attr.send_cq = ctx->send_cq;
    ctx->qp_attr.recv_cq = ctx->recv_cq;
    ctx->qp_attr.cap.max_send_wr  = CQE_SIZE;
    ctx->qp_attr.cap.max_recv_wr  = CQE_SIZE;
    ctx->qp_attr.cap.max_send_sge = 1;
    ctx->qp_attr.cap.max_recv_sge = 1;
    ctx->qp_attr.cap.max_inline_data = 0;
    ctx->qp_attr.qp_type = IB_QPT_RC;
    ctx->qp_attr.sq_sig_type = IB_SIGNAL_ALL_WR;
    
    ctx->qp = ib_create_qp(ctx->pd, &ctx->qp_attr);

    // connect with server with TCP
    retval = connect(ctx, ip_addr, port);
    if (retval != 0)
        return 0;
  
    retval = populate_port_data(ctx);
    if (retval != 0)
        return 0;

    // some necessary stuff 
    ctx->lid = rdma_ib_device.attr.lid;
    ctx->qpn = ctx->qp->qp_num;
    get_random_bytes(&ctx->psn, sizeof(ctx->psn));
    ctx->psn &= 0xffffff;
    
    // exchange data to bootstrap RDMA
    retval = handshake(ctx); 
    if (retval != 0)
        return 0;

    // create memory region
    // modify QP to RTS
    retval = rdma_setup(ctx);
    if (retval != 0)
        return 0;

    sock_release(ctx->sock);

    init_waitqueue_head(&ctx->queue);
    atomic64_set(&ctx->wr_count, 0);

    atomic_set(&ctx->operation_count, 0);
    atomic_set(&ctx->comp_handler_count, 0);


    ctx->pool = get_batch_request_pool(mem_pool_size);
    return ctx;
}

bool merge_wr(struct ib_send_wr* old_wr, struct ib_sge *old_sg, struct ib_send_wr* new_wr, struct ib_sge *new_sg) 
{
    if(old_wr->opcode == new_wr->opcode && old_sg->addr + old_sg->length == new_sg->addr && old_wr->wr.rdma.remote_addr + old_sg->length == new_wr->wr.rdma.remote_addr)
    {
        old_sg->length += new_sg->length;
        return true;
    }
    else
    {
        return false;
    }
}

void make_wr(rdma_ctx_t ctx, struct ib_send_wr* wr, struct ib_sge *sg, RDMA_OP op,
        u64 dma_addr, uint64_t remote_offset, uint length, struct batch_request* batch_req)
{
    memset(sg, 0, sizeof(struct ib_sge));
    sg->addr     = (uintptr_t)dma_addr;
    sg->length   = length;
    sg->lkey     = ctx->mr->lkey;

    memset(wr, 0, sizeof(*wr));
#if MODE == MODE_ASYNC || MODE == MODE_ONE
    wr->wr_id      = (u64)batch_req;
#elif MODE == MODE_SYNC
    wr->wr_id      = 0;
#else
    #error "Wrong Mode"
#endif
    wr->sg_list    = sg;
    wr->num_sge    = 1;
    wr->opcode     = (op==RDMA_READ?IB_WR_RDMA_READ : IB_WR_RDMA_WRITE);
    wr->send_flags = IB_SEND_SIGNALED;
    wr->wr.rdma.remote_addr = ctx->rem_vaddr + remote_offset;
    wr->wr.rdma.rkey        = ctx->rem_rkey;
}

void simple_make_wr(rdma_ctx_t ctx, struct ib_send_wr* wr, struct ib_sge *sg, RDMA_OP op,
        u64 dma_addr, uint64_t remote_offset, uint length, struct batch_request* batch_req)
{
    sg->addr     = (uintptr_t)dma_addr;
    sg->length   = length;

#if MODE == MODE_ASYNC || MODE == MODE_ONE
    wr->wr_id      = (u64)batch_req;
#elif MODE == MODE_SYNC
    wr->wr_id      = 0;
#else
    #error "Wrong Mode"
#endif
    wr->opcode     = (op==RDMA_READ?IB_WR_RDMA_READ : IB_WR_RDMA_WRITE);
    wr->wr.rdma.remote_addr = ctx->rem_vaddr + remote_offset;
}


int send_wr(rdma_ctx_t ctx, RDMA_OP op, u64 dma_addr, uint64_t remote_offset,
        uint length, struct batch_request* batch_req)
{
    struct ib_send_wr* bad_wr;
    struct ib_sge sg;
    struct ib_send_wr wr;
    int retval;

    make_wr(ctx, &wr, &sg, op, dma_addr, remote_offset, length, batch_req);

    retval = ib_post_send(ctx->qp, &wr, &bad_wr);
    if (retval)
    {
        pr_err("Error posting write request, msg %d", retval);
        BUG();
    }
    return 0;
}

int rdma_op(rdma_ctx_t ct, rdma_req_t req, int n_requests)
{
    int i;
    struct rdma_ctx* ctx = ct;


    
    LOG_KERN(LOG_INFO, "rdma op with %d reqs, pid %d, cpu %d", n_requests, current->pid, smp_processor_id());
    ctx->outstanding_requests = n_requests;
    //atomic_set(&ctx->outstanding_requests, n_requests);
    for (i = 0; i < n_requests; ++i) {
        if (req[i].rw == RDMA_READ || req[i].rw == RDMA_WRITE) {
            send_wr((rdma_ctx_t)ctx, req[i].rw, req[i].dma_addr, req[i].remote_offset, req[i].length, req[i].batch_req);
        } else {
            LOG_KERN(LOG_INFO, "Wrong op", 0);
            ctx->outstanding_requests = 0;
            //atomic_set(&ctx->outstanding_requests, 0);
            return -1;
        }
    }

    //LOG_KERN(LOG_INFO, "Waiting for requests completion n_req = %lu", ctx->outstanding_requests);
    // wait until all requests are done

#if MODE == MODE_SYNC
    poll_cq(ctx);
#endif
    return 0;
}

