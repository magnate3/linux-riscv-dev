/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#define _GNU_SOURCE
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <rdma/rdma_cma.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>
#include <string.h>
#include <malloc.h>
#include <netdb.h>

#include "list.h"


enum log_level {
    LOG_LEVEL_ERROR,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_TRACE
};

enum packet_type {
    PACKET_FIN = 2
};

#define LOG(_level, _fmt, ...) \
    { \
        if (LOG_LEVEL_##_level <= g_options.log_level) { \
            printf("%12s:%-4d %-5s " _fmt "\n", basename(__FILE__), \
                   __LINE__, #_level, ## __VA_ARGS__); \
        } \
    }
#define LOG_ERROR(_fmt, ...)   LOG(ERROR, _fmt, ## __VA_ARGS__)
#define LOG_INFO(_fmt, ...)    LOG(INFO,  _fmt, ## __VA_ARGS__)
#define LOG_DEBUG(_fmt, ...)   LOG(DEBUG, _fmt, ## __VA_ARGS__)
#define LOG_TRACE(_fmt, ...)   LOG(TRACE, _fmt, ## __VA_ARGS__)
#define BIT(_index)            (1ul << (_index))
#define DC_KEY                 0x1
#define MEM_ACCESS_FLAGS       (IBV_ACCESS_LOCAL_WRITE  | \
                                IBV_ACCESS_REMOTE_WRITE | \
                                IBV_ACCESS_REMOTE_READ)
#define IB_GRH_SIZE            40
#define NUM_DCI                8
#define MIN(a, b)              ({ \
                                  typeof(a) _a = (a);  \
                                  typeof(b) _b = (b);  \
                                  _a < _b ? _a : _b;   \
                               })
#define MAX(a, b)              ({ \
                                  typeof(a) _a = (a);  \
                                  typeof(b) _b = (b);  \
                                  _a > _b ? _a : _b;   \
                               })

/* Test options */
typedef struct {
    int                       log_level;
    const char                *dest_address;
    unsigned                  port_num;
    unsigned                  conn_backlog;
    unsigned                  num_connections;
    int                       conn_timeout_ms;
    size_t                    rdma_read_size;
    unsigned                  num_iterations;
    size_t                    max_outstanding_reads;
    size_t                    max_read_size;
    size_t                    tx_queue_len;
    size_t                    rx_queue_len;
    size_t                    max_send_sge;
    size_t                    max_recv_sge;
    unsigned                  max_rd_atomic;
    unsigned                  min_rnr_timer;
    unsigned                  xport_timeout;
    unsigned                  rnr_retry;
    unsigned                  xport_retry_cnt;
    uint8_t                   gid_index;
    uint8_t                   hop_limit;
    uint8_t                   traffic_class;
    uint8_t                   sl;
} options_t;

/* Connection to remote peer (on either client or server) */
typedef struct connection connection_t;
struct connection {
    struct rdma_cm_id         *rdma_id;
    struct ibv_ah             *dc_ah;
    uint32_t                  remote_dctn;
    uint32_t                  rkey;
    uint64_t                  remote_addr;
    size_t                    read_offset;
    list_link_t               list;
};

/* Registered memory */
typedef struct {
    void                      *ptr;
    struct ibv_mr             *mr;
} buffer_t;

/* Global test context */
typedef struct {
    struct rdma_event_channel *event_ch;
    struct rdma_cm_id         *listen_cm_id;
    struct ibv_cq             *cq;
    struct ibv_srq            *srq;
    buffer_t                  recv_buf;
    buffer_t                  rdma_buf;
    //struct ibv_exp_dct        *dct;
    struct ibv_qp             *dct;
    struct ibv_qp             *dcis[NUM_DCI];
    struct ibv_qp_ex          *dcis_ex[NUM_DCI];
    struct mlx5dv_qp_ex       *m_dcis_ex[NUM_DCI];
    unsigned                  dci_outstanding[NUM_DCI];
    uint64_t                  free_dci_bitmap;
    connection_t              *conns;
    unsigned                  num_conns;
    unsigned                  num_established;
    unsigned                  num_disconnect;
    unsigned                  num_fin_recvd;
    unsigned                  recv_available;
    unsigned long             num_outstanding_reads;
} test_context_t;

/* Control packet */
typedef struct {
    uint8_t                   type;        /* packet_type */
    uint32_t                  conn_index;  /* index of connection to FIN */
} packet_t;

/* Private data passed over rdma_cm protocol */
typedef struct {
    uint32_t                  conn_index;  /* Index of connection in the array */
    uint32_t                  dct_num;     /* DCT number (for DC only) */
    uint32_t                  rkey;        /* Remote key for RDMA */
    uint64_t                  virt_addr;   /* Remote buffer address for RDMA */
    union ibv_gid             gid;         /* GID to use for sending packets */
} conn_priv_t;

options_t g_options = {
    .log_level             = LOG_LEVEL_INFO,
    .dest_address          = "",
    .port_num              = 20000,
    .conn_backlog          = 1000,
    .num_connections       = 1,
    .conn_timeout_ms       = 2000,
    .rdma_read_size        = 1024 * 1024,
    .num_iterations        = 1000,
    .max_outstanding_reads = 8,
    .max_read_size         = 32768,
    .tx_queue_len          = 128,
    .rx_queue_len          = 128,
    .max_send_sge          = 1,
    .max_recv_sge          = 1,
    .max_rd_atomic         = 8,
    .min_rnr_timer         = 17,
    .xport_timeout         = 17,
    .rnr_retry             = 7,
    .xport_retry_cnt       = 7,
    .gid_index             = 3,
    .hop_limit             = 64,
    .traffic_class         = 0,
    .sl                    = 0
};

test_context_t g_test = {
    .event_ch        = NULL,
    .listen_cm_id    = NULL,
    .cq              = NULL,
    .srq             = NULL,
    .recv_buf = {
         .ptr        = NULL,
         .mr         = NULL
     },
    .rdma_buf = {
        .ptr         = NULL,
        .mr          = NULL
    },
    .dct             = NULL,
    .conns           = NULL,
    .num_conns       = 0,
    .num_established = 0,
    .num_disconnect  = 0,
    .num_fin_recvd   = 0,
    .recv_available  = 0
};

static void usage(const options_t *defaults) {
    printf("Usage: many2one [ options ] [ server-address]\n");
    printf("Common options:\n");
    printf("   -p <num>       Server port number to use (%d)\n", defaults->port_num);
    printf("   -v             Increase logging level\n");
    printf("   -G <index>     (DC only) GID index to use (%d)\n", defaults->gid_index);
    printf("   -T <class>     (DC only) Traffic class / DSCP to use (%d)\n", defaults->traffic_class);
    printf("   -H <limit>     (DC only) Hop limit / TTL (%d)\n", defaults->hop_limit);
    printf("   -S <sl>        (DC only) SL / Ethernet priority (%d)\n", defaults->sl);
    printf("\n");
    printf("Client options:\n");
    printf("   -t <ms>        Connection timeout, milliseconds (%d)\n", defaults->conn_timeout_ms);
    printf("\n");
    printf("Server options:\n");
    printf("   -n <num>       Number of connections to expect (%d)\n", defaults->num_connections);
    printf("   -i <num>       Number of iterations to run (%d)\n", defaults->num_iterations);
    printf("   -r <size>      Maximal size of a single rdma read (%zu)\n", defaults->max_read_size);
    printf("   -o <num>       Maximal number of outstanding reads (%zu)\n", defaults->max_outstanding_reads);
}

static int parse_opts(int argc, char **argv) {
    options_t defaults = g_options;
    int c;

    while ( (c = getopt(argc, argv, "hp:n:vt:i:r:o:G:T:H:S:")) != -1 ) {
        switch (c) {
        case 'p':
            g_options.port_num = atoi(optarg);
            break;
        case 'n':
            g_options.num_connections = atoi(optarg);
            break;
        case 'v':
            ++g_options.log_level;
            break;
        case 't':
            g_options.conn_timeout_ms = atoi(optarg);
            break;
        case 'i':
            g_options.num_iterations = atoi(optarg);
            break;
        case 'r':
            g_options.max_read_size = atol(optarg);
            break;
        case 'o':
            g_options.max_outstanding_reads = atol(optarg);
            break;
        case 'G':
            g_options.gid_index = atoi(optarg);
            break;
        case 'T':
            g_options.traffic_class = atoi(optarg);
            break;
        case 'H':
            g_options.hop_limit = atoi(optarg);
            break;
        case 'S':
            g_options.sl = atoi(optarg);
            break;
        case 'h':
            usage(&defaults);
            exit(0);
        default:
            LOG_ERROR("Invalid option '%c'", c);
            usage(&defaults);
            return -1;
        }
    }

    if (optind < argc) {
        g_options.dest_address    = argv[optind++];
    }

    return 0;
}

static int init_buffer(struct ibv_pd *pd, size_t size, buffer_t *buf)
{
    if (buf->ptr) {
        return 0; /* already initialized */
    }

    buf->ptr = memalign(4096, size);
    if (!buf->ptr) {
        LOG_ERROR("Failed to allocate buffer");
        return -1;
    }

    buf->mr = ibv_reg_mr(pd, buf->ptr, size, MEM_ACCESS_FLAGS);
    if (!buf->mr) {
        LOG_ERROR("ibv_reg_mr() failed: %m");
        return -1;
    }

    LOG_DEBUG("Registered buffer %p length %zu lkey 0x%x rkey 0x%x",
              buf->ptr, size, buf->mr->lkey, buf->mr->rkey);
    return 0;
}

static void cleanup_buffer(buffer_t *buf)
{
    if (buf->ptr) {
        free(buf->ptr);
    }
    if (buf->mr) {
        ibv_dereg_mr(buf->mr);
    }
}

static int init_test()
{
    /* Create rdma_cm event channel */
    g_test.event_ch = rdma_create_event_channel();
    if (!g_test.event_ch) {
        LOG_ERROR("rdma_create_event_channel() failed: %m");
        return -1;
    }

    /* Allocate array of connections */
    g_test.conns = calloc(g_options.num_connections, sizeof(*g_test.conns));
    if (!g_test.conns) {
        LOG_ERROR("failed to allocate connections array");
        return -1;
    }

    return 0;
}

static void cleanup_test()
{
    unsigned i;

    for (i = 0; i < g_test.num_conns; ++i) {
        if (g_test.conns[i].dc_ah) {
            ibv_destroy_ah(g_test.conns[i].dc_ah);
        }
        if (g_test.conns[i].rdma_id != NULL) {
            rdma_destroy_id(g_test.conns[i].rdma_id);
        }
    }
    for (i = 0; i < NUM_DCI; ++i) {
        if (g_test.dcis[i]) {
            ibv_destroy_qp(g_test.dcis[i]);
        }
    }
    if (g_test.srq) {
        ibv_destroy_srq(g_test.srq);
    }
    if (g_test.cq) {
        ibv_destroy_cq(g_test.cq);
    }
    cleanup_buffer(&g_test.rdma_buf);
    cleanup_buffer(&g_test.recv_buf);
    if (g_test.listen_cm_id) {
        rdma_destroy_id(g_test.listen_cm_id);
    }
    free(g_test.conns);
    rdma_destroy_event_channel(g_test.event_ch);
}

static int get_address(struct sockaddr_in *in_addr)
{
    struct hostent *he = gethostbyname(g_options.dest_address);
    if (!he || !he->h_addr_list) {
        LOG_ERROR("host %s not found: %s", g_options.dest_address,
                  hstrerror(h_errno));
        return -1;
    }

    if (he->h_addrtype != AF_INET) {
        LOG_ERROR("Only IPv4 addresses are supported");
        return -1;
    }

    if (he->h_length != sizeof(struct in_addr)) {
        LOG_ERROR("Mismatching address length");
        return -1;
    }

    memset(in_addr, 0, sizeof(*in_addr));
    in_addr->sin_family = AF_INET;
    in_addr->sin_port   = htons(g_options.port_num);
    in_addr->sin_addr   = *(struct in_addr*)he->h_addr_list[0];
    return 0;
}

/****************************************************************************************
 * Modify target QP state to RTR (on the client side)
 * Return value: 0 - success, 1 - error
 ****************************************************************************************/
static int modify_target_qp_to_rtr(struct rdma_cm_id *rdma_cm_id, enum ibv_mtu mtu)
{
    struct ibv_qp_attr      qp_attr;
    enum ibv_qp_attr_mask   attr_mask;

    memset(&qp_attr, 0, sizeof qp_attr);
    qp_attr.qp_state       = IBV_QPS_RTR;
    qp_attr.path_mtu       = mtu;
    qp_attr.min_rnr_timer  = g_options.min_rnr_timer;
    qp_attr.ah_attr.port_num = rdma_cm_id->port_num;
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.hop_limit  = g_options.hop_limit;
    qp_attr.ah_attr.grh.sgid_index = g_options.gid_index;
    qp_attr.ah_attr.grh.traffic_class = g_options.traffic_class;

    attr_mask = IBV_QP_STATE          |
                IBV_QP_AV             |
                IBV_QP_PATH_MTU       |
                IBV_QP_MIN_RNR_TIMER; // for DCT

    LOG_DEBUG("ibv_modify_qp(qp = %p, qp_attr.qp_state = %d, attr_mask = 0x%x)\n",
               g_test.dct, qp_attr.qp_state, attr_mask);
    if (ibv_modify_qp(g_test.dct, &qp_attr, attr_mask)) {
        fprintf(stderr, "Failed to modify QP to RTR\n");
        return 1;
    }
    LOG_DEBUG ("ibv_modify_qp to state %d completed: qp_num = %u\n", qp_attr.qp_state, g_test.dct->qp_num);

    return 0;
}

/****************************************************************************************
 * Modify source QP state to RTR and then to RTS (on the server side)
 * Return value: 0 - success, 1 - error
 ****************************************************************************************/
static int modify_source_qp_to_rtr_and_rts(struct rdma_cm_id *rdma_cm_id, enum ibv_mtu mtu, struct ibv_qp *dci)
{
    struct ibv_qp_attr      qp_attr;
    enum ibv_qp_attr_mask   attr_mask;

    memset(&qp_attr, 0, sizeof qp_attr);
    
    /* - - - - - - -  Modify QP to RTR  - - - - - - - */
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = mtu;
    qp_attr.ah_attr.port_num = rdma_cm_id->port_num;

    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.hop_limit  = g_options.hop_limit;
    qp_attr.ah_attr.grh.sgid_index = g_options.gid_index;
    qp_attr.ah_attr.sl        = g_options.sl;

    attr_mask = IBV_QP_STATE    |
                IBV_QP_AV       |
                IBV_QP_PATH_MTU ;

    LOG_DEBUG("ibv_modify_qp(qp = %p, qp_attr.qp_state = %d, attr_mask = 0x%x)\n",
               dci, qp_attr.qp_state, attr_mask);
    if (ibv_modify_qp(dci, &qp_attr, attr_mask)) {
        fprintf(stderr, "Failed to modify QP to RTR\n");
        return 1;
    }
    LOG_DEBUG ("ibv_modify_qp to state %d completed: qp_num = %u\n", qp_attr.qp_state, dci->qp_num);

    /* - - - - - - -  Modify QP to RTS  - - - - - - - */
    qp_attr.qp_state       = IBV_QPS_RTS;
    qp_attr.timeout        = g_options.xport_timeout;
    qp_attr.retry_cnt      = g_options.xport_retry_cnt;
    qp_attr.rnr_retry      = g_options.rnr_retry;
    //qp_attr.sq_psn         = 0;
    qp_attr.max_dest_rd_atomic = g_options.max_rd_atomic;
    qp_attr.max_rd_atomic      = g_options.max_rd_atomic;
    attr_mask = IBV_QP_STATE            |
                IBV_QP_TIMEOUT          |
                IBV_QP_RETRY_CNT        |
                IBV_QP_RNR_RETRY        |
                IBV_QP_SQ_PSN           |
                IBV_QP_MAX_QP_RD_ATOMIC ;
    LOG_DEBUG("ibv_modify_qp(qp = %p, qp_attr.qp_state = %d, attr_mask = 0x%x)\n",
               dci, qp_attr.qp_state, attr_mask);
    if (ibv_modify_qp(dci, &qp_attr, attr_mask)) {
        fprintf(stderr, "Failed to modify QP to RTS\n");
        return 1;
    }
    LOG_DEBUG ("ibv_modify_qp to state %d completed: qp_num = %u\n", qp_attr.qp_state, dci->qp_num);
    
    return 0;
}

static int init_dc_qps(struct rdma_cm_id *rdma_cm_id)
{
    
    struct ibv_port_attr port_attr;
    int i, ret;
    
    if (g_test.dct) {
        return 0; /* Already initialized */
    }

    ret = ibv_query_port(rdma_cm_id->verbs, rdma_cm_id->port_num, &port_attr);
    if (ret) {
        LOG_ERROR("ibv_query_port() failed: %m");
        return ret;
    }

    if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
        LOG_ERROR("This test can run only on Ethernet (RoCE) port");
        return -1;
    }

    /* Create DCT
      * Note: For RoCE, must specify gid_index, hop_limit, traffic_class
      *       on command line.
      * */
    
    /* **********************************  Create DCT QP  ********************************** */
    struct ibv_qp_init_attr_ex attr_ex;
    struct mlx5dv_qp_init_attr attr_dv;

    memset(&attr_ex, 0, sizeof(attr_ex));
    memset(&attr_dv, 0, sizeof(attr_dv));

    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = g_test.cq;
    attr_ex.recv_cq = g_test.cq;

    attr_ex.comp_mask |= IBV_QP_INIT_ATTR_PD;
    attr_ex.pd = rdma_cm_id->pd;
    attr_ex.srq = g_test.srq; /* Should use SRQ for DCT only */

    /* create DCT */
    attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_DC;
    attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    attr_dv.dc_init_attr.dct_access_key = DC_KEY;

    LOG_DEBUG ("mlx5dv_create_qp(%p,%p,%p)\n", rdma_cm_id->verbs, &attr_ex, &attr_dv);
    g_test.dct = mlx5dv_create_qp(rdma_cm_id->verbs, &attr_ex, &attr_dv);

    if (!g_test.dct)  {
        fprintf(stderr, "Couldn't create DCT QP\n");
        return -1;
    }
    LOG_DEBUG ("mlx5dv_create_qp %p completed: qp_num = %u\n", g_test.dct, g_test.dct->qp_num);

    /* - - - - - - -  Modify QP to INIT  - - - - - - - */
    struct ibv_qp_attr qp_attr = {
        .qp_state        = IBV_QPS_INIT,
        .pkey_index      = 0,
        .port_num        = rdma_cm_id->port_num,
        .qp_access_flags = MEM_ACCESS_FLAGS
    };
    enum ibv_qp_attr_mask attr_mask = IBV_QP_STATE      |
                                      IBV_QP_PKEY_INDEX |
                                      IBV_QP_PORT       |
                                      IBV_QP_ACCESS_FLAGS;
    LOG_DEBUG ("ibv_modify_qp(qp = %p, qp_attr.qp_state = %d, attr_mask = 0x%x)\n",
               g_test.dct, qp_attr.qp_state, attr_mask);
    ret = ibv_modify_qp(g_test.dct, &qp_attr, attr_mask);
    if (ret) {
        fprintf(stderr, "Failed to modify QP to INIT, error %d\n", ret);
        return -1;
    }
    LOG_DEBUG ("ibv_modify_qp to state %d completed: qp_num = %u\n", qp_attr.qp_state, g_test.dct->qp_num);

    ret = modify_target_qp_to_rtr(rdma_cm_id, port_attr.active_mtu);
    if (ret) {
        return -1;
    }

//    struct ibv_exp_qp_init_attr qp_init_attr;
//    struct ibv_exp_dct_init_attr dct_attr;
//    struct ibv_exp_qp_attr qp_attr;

    /* Create and initialize DC initiators */
    for (i = 0; i < NUM_DCI; ++i) {
        /* **********************************  Create QP  ********************************** */
        struct ibv_qp_init_attr_ex attr_ex;
        struct mlx5dv_qp_init_attr attr_dv;
        
        memset(&attr_ex, 0, sizeof(attr_ex));
        memset(&attr_dv, 0, sizeof(attr_dv));
        
        attr_ex.qp_type = IBV_QPT_DRIVER;
        attr_ex.send_cq = g_test.cq;
        attr_ex.recv_cq = g_test.cq;
        attr_ex.comp_mask |= IBV_QP_INIT_ATTR_PD;
        attr_ex.pd = rdma_cm_id->pd;

        //attr_ex.srq                 = g_test.srq;//?
        
        attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_DC;
        attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;
        attr_dv.dc_init_attr.dct_access_key = DC_KEY;//? - don't need for DCI

        attr_ex.cap.max_send_wr  = g_options.tx_queue_len;
        //attr_ex.cap.max_recv_wr  = g_options.rx_queue_len;//?
        attr_ex.cap.max_send_sge = g_options.max_send_sge;
        //attr_ex.cap.max_recv_sge = g_options.max_recv_sge;//?
        attr_ex.cap.max_inline_data = sizeof(packet_t);

        attr_ex.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
        attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_READ;

        attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
        attr_dv.create_flags |= MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE; /*driver doesnt support scatter2cqe data-path on DCI yet*/

        LOG_DEBUG ("mlx5dv_create_qp(%p,%p,%p)\n", rdma_cm_id->verbs, &attr_ex, &attr_dv);
        g_test.dcis[i] = mlx5dv_create_qp(rdma_cm_id->verbs, &attr_ex, &attr_dv);
        if (!g_test.dcis[i])  {
            fprintf(stderr, "Couldn't create QP\n");
            return -1;
        }
        LOG_DEBUG ("mlx5dv_create_qp %p completed: qp_num = %u\n", g_test.dcis[i], g_test.dcis[i]->qp_num);

        g_test.dcis_ex[i] = ibv_qp_to_qp_ex(g_test.dcis[i]);
        if (!g_test.dcis_ex[i])  {
            fprintf(stderr, "Couldn't create QPEX\n");
            return -1;
        }
        g_test.m_dcis_ex[i] = mlx5dv_qp_ex_from_ibv_qp_ex(g_test.dcis_ex[i]);
        if (!g_test.m_dcis_ex[i])  {
            fprintf(stderr, "Couldn't create MQPEX\n");
            return -1;
        }
        
        /* - - - - - - -  Modify QP to INIT  - - - - - - - */
        struct ibv_qp_attr qp_attr = {
            .qp_state        = IBV_QPS_INIT,
            .pkey_index      = 0,
            .port_num        = rdma_cm_id->port_num,
            .qp_access_flags = 0
        };
        enum ibv_qp_attr_mask attr_mask = IBV_QP_STATE      |
                                          IBV_QP_PKEY_INDEX |
                                          IBV_QP_PORT       |
                                          0 /*IBV_QP_ACCESS_FLAGS*/; /*we must zero this bit for DCI QP*/
        LOG_DEBUG("ibv_modify_qp(qp = %p, qp_attr.qp_state = %d, attr_mask = 0x%x)\n",
                   g_test.dcis[i], qp_attr.qp_state, attr_mask);
        ret = ibv_modify_qp(g_test.dcis[i], &qp_attr, attr_mask);
        if (ret) {
            fprintf(stderr, "Failed to modify QP to INIT, error %d\n", ret);
            return -1;
        }
        LOG_DEBUG("ibv_modify_qp to state %d completed: qp_num = %u\n", qp_attr.qp_state, g_test.dcis[i]->qp_num);

        ret = modify_source_qp_to_rtr_and_rts(rdma_cm_id, port_attr.active_mtu, g_test.dcis[i]);
        if (ret) {
            return -1;
        }

        g_test.dci_outstanding[i] = 0;
        LOG_DEBUG("Created DCI[%d]=0x%x", i, g_test.dcis[i]->qp_num);
    }

    g_test.free_dci_bitmap = -1;

    return 0;
}

static int post_receives()
{
    struct ibv_recv_wr wr, *bad_wr;
    struct ibv_sge sge;
    unsigned i;
    void *ptr;
    int ret;

    /* We post receives only once, should have enough to handle a control
     * message for every connection (FIN)
     */
    for (i = 0; i < g_test.recv_available; ++i) {
        /* sge.addr points to grh, wr_id points after grh */
        ptr        = g_test.recv_buf.ptr + (i * sizeof(packet_t));
        wr.next    = NULL;
        wr.num_sge = 1;
        wr.sg_list = &sge;
        sge.addr   = (uintptr_t)ptr;
        sge.length = sizeof(packet_t);
        sge.lkey   = g_test.recv_buf.mr->lkey;
        wr.wr_id   = (uintptr_t)ptr;

        ret = ibv_post_srq_recv(g_test.srq, &wr, &bad_wr);
        if (ret) {
            LOG_ERROR("ibv_post_srq_recv() failed: %m");
            return ret;
        }
    }

    if (g_test.recv_available) {
        LOG_DEBUG("Posted %d receives", g_test.recv_available);
        g_test.recv_available = 0;
    }

    return 0;
}

static int init_transport(struct rdma_cm_event *event)
{
    struct ibv_srq_init_attr srq_init_attr;
    int ret;

    if (!g_test.cq) {
        /* Create single completion queue for both sends and receives */
        g_test.cq = ibv_create_cq(event->id->verbs,
                                  g_options.tx_queue_len + g_options.rx_queue_len, /* cqe */
                                  NULL, /* cq_context */
                                  NULL, /* comp_channel */
                                  0  /* comp_vector */ );
        if (!g_test.cq) {
            LOG_ERROR("ibv_create_cq() failed: %m");
            return -1;
        }

        LOG_DEBUG("Created CQ @%p", g_test.cq)
    }

    if (!g_test.srq) {
        /* Create a shared receive queue for control messages */
        srq_init_attr.srq_context    = NULL;
        srq_init_attr.attr.max_wr    = g_options.rx_queue_len;
        srq_init_attr.attr.max_sge   = g_options.max_recv_sge;
        srq_init_attr.attr.srq_limit = 0;
        g_test.srq = ibv_create_srq(event->id->pd, &srq_init_attr);
        if (!g_test.srq) {
            LOG_ERROR("ibv_create_srq() failed: %m");
            return -1;
        }

        g_test.recv_available = g_options.rx_queue_len;

        LOG_DEBUG("Created SRQ @%p", g_test.srq);
    }

    /* Create buffer for receives */
    ret = init_buffer(event->id->pd, g_options.rx_queue_len * sizeof(packet_t),
                      &g_test.recv_buf);
    if (ret) {
        return ret;
    }

    /* Create buffer for RDMA (one buffer which is split between connections) */
    ret = init_buffer(event->id->pd,
                      g_options.num_connections * g_options.rdma_read_size,
                      &g_test.rdma_buf);
    if (ret) {
        return ret;
    }

    ret = post_receives();
    if (ret) {
        return ret;
    }

    /* Initialize DC objects (done only once) */
    ret = init_dc_qps(event->id);
    if (ret) {
        return ret;
    }

    return 0;
}

static int get_conn_param(struct rdma_cm_id *rdma_id,
                          struct rdma_conn_param *conn_param,
                          conn_priv_t *priv, unsigned conn_index)
{
    int ret;

    memset(conn_param, 0, sizeof(*conn_param));

    priv->conn_index          = conn_index;
    priv->rkey                = g_test.rdma_buf.mr->rkey;
    priv->virt_addr           = (uint64_t)g_test.rdma_buf.ptr +
                                (conn_index * g_options.rdma_read_size);

    priv->dct_num  = g_test.dct->qp_num;

    ret = ibv_query_gid(rdma_id->verbs, rdma_id->port_num,
                        g_options.gid_index, &priv->gid);
    if (ret) {
        LOG_ERROR("ibv_query_gid(port=%d index=%d) failed: %m",
                  rdma_id->port_num, g_options.gid_index);
        return ret;
    }

    conn_param->private_data        = priv;
    conn_param->private_data_len    = sizeof(*priv);
    conn_param->responder_resources = g_options.max_rd_atomic;
    conn_param->initiator_depth     = g_options.max_rd_atomic;
    conn_param->retry_count         = g_options.xport_retry_cnt;
    conn_param->rnr_retry_count     = g_options.rnr_retry;

    return 0;
}

static int set_conn_param(connection_t *conn, struct rdma_cm_event *event)
{
    const conn_priv_t *remote_priv = event->param.conn.private_data;
    struct ibv_ah_attr ah_attr;

    conn->remote_addr = remote_priv->virt_addr;
    conn->rkey        = remote_priv->rkey;

    /* Creating Address handler */
    memset(&ah_attr, 0, sizeof(ah_attr));

    ah_attr.is_global         = 1;
    ah_attr.port_num          = event->id->port_num;
    ah_attr.sl                = g_options.sl;
    ah_attr.grh.dgid          = remote_priv->gid;
    ah_attr.grh.sgid_index    = g_options.gid_index;
    ah_attr.grh.hop_limit     = g_options.hop_limit;
    ah_attr.grh.traffic_class = g_options.traffic_class;

    conn->dc_ah = ibv_create_ah(event->id->pd, &ah_attr);
    if (!conn->dc_ah) {
        LOG_ERROR("ibv_create_ah(port_num=%d) failed: %m", ah_attr.port_num );
        return -1;
    }

    conn->remote_dctn = remote_priv->dct_num;
    LOG_DEBUG("DC ah @%p remote_dctn 0x%x", conn->dc_ah, conn->remote_dctn);

    return 0;
}

static connection_t* add_connection(struct rdma_cm_event *event)
{
    connection_t *conn;
    int ret;

    conn = &g_test.conns[g_test.num_conns];
    conn->rdma_id = event->id;

    ret = set_conn_param(conn, event);
    if (ret) {
        return NULL;
    }

    ++g_test.num_conns;
    return conn;
}

static int handle_connect_request(struct rdma_cm_event *event)
{
    struct rdma_conn_param conn_param;
    conn_priv_t priv;
    connection_t *conn;
    int ret;

    ret = init_transport(event);
    if (ret) {
        return ret;
    }

    /* add a new connection and set its parameters according to private_data
     * received from the client
     */
    conn = add_connection(event);
    if (!conn) {
        return -1;
    }

    LOG_DEBUG("calling rdma_accept()");

    ret = get_conn_param(event->id, &conn_param, &priv, conn - g_test.conns);
    if (ret) {
        return ret;
    }

    /* send accept message to the client with our own parameters in private_data */
    ret = rdma_accept(event->id, &conn_param);
    if (ret) {
        LOG_ERROR("rdma_accept() failed: %m");
        return -1;
    }

    /* For DC server, the connection is ready after getting a connect request */
    ++g_test.num_established;

    return 0;
}

static int is_client()
{
    return strlen(g_options.dest_address);
}

/* send control message */
static int send_control(connection_t *conn, enum packet_type type,
                        unsigned conn_index)
{
    packet_t packet = { .type = type,
                        .conn_index = conn_index };
    int dci, ret;

    dci = rand() % NUM_DCI;
    assert(g_test.dci_outstanding[dci] < g_options.tx_queue_len);

    LOG_TRACE("send_control: ibv_wr_start: qpex = %p\n", g_test.dcis_ex[dci]);
    ibv_wr_start(g_test.dcis_ex[dci]);

    g_test.dcis_ex[dci]->wr_id = (uint64_t)dci;
    g_test.dcis_ex[dci]->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;

    LOG_TRACE("send_control: ibv_wr_send: wr_id=0x%lx, qpex=%p\n", g_test.dcis_ex[dci]->wr_id, g_test.dcis_ex[dci]);
    ibv_wr_send(g_test.dcis_ex[dci]);

    LOG_TRACE("send_control: mlx5dv_wr_set_dc_addr: mqpex=%p, ah=%p, rem_dctn=0x%06x\n",
               g_test.m_dcis_ex[dci], conn->dc_ah, conn->remote_dctn);
    mlx5dv_wr_set_dc_addr(g_test.m_dcis_ex[dci], conn->dc_ah, conn->remote_dctn, DC_KEY);

    LOG_TRACE("send_control: ibv_wr_set_inline_data: qpex=%p, lkey=0, local_buf=%p, size=%u\n",
              g_test.dcis_ex[dci], &packet, (uint32_t)sizeof(packet));
    ibv_wr_set_inline_data(g_test.dcis_ex[dci], &packet, (uint32_t)sizeof(packet));
    
//    LOG_TRACE("send_control: ibv_wr_set_sge: qpex=%p, lkey=0, local_buf=%p, size=%u\n",
//              g_test.dcis_ex[dci], &packet, (uint32_t)sizeof(packet));
//    ibv_wr_set_sge(g_test.dcis_ex[dci], 0 /*mr->lkey*/, (uintptr_t)&packet, (uint32_t)sizeof(packet));

    LOG_TRACE("send_control: ibv_wr_complete: qpex=%p\n", g_test.dcis_ex[dci]);
    ret = ibv_wr_complete(g_test.dcis_ex[dci]);
    if (ret) {
        LOG_ERROR("send_control: ibv_wr_complete failed (error=%d)\n", ret);
        return -1;
    }
    
    ++g_test.dci_outstanding[dci];

    LOG_TRACE("Sent packet %d conn_index %d", packet.type, packet.conn_index);
    return 0;
}

static int handle_established(struct rdma_cm_event *event)
{
    connection_t *conn;

    if (is_client()) {
        /* Add new (and only) connection on client side */
        conn = add_connection(event);
        if (!conn) {
            return -1;
        }
    }

    ++g_test.num_established;
    return 0;
}

/* wait and process single event */
static int wait_and_process_one_event(uint64_t event_mask) {
    struct rdma_conn_param conn_param;
    struct rdma_cm_event *event;
    conn_priv_t priv;
    int ret;

    ret = rdma_get_cm_event(g_test.event_ch, &event);
    if (ret) {
        LOG_ERROR("rdma_get_cm_event() failed: %m");
        return -1;
    }

    if (!(BIT(event->event) & event_mask)) {
        LOG_ERROR("Unexpected event %s", rdma_event_str(event->event));
        ret = -1;
        goto out_ack_event;
    }

    LOG_DEBUG("Got rdma_cm event %s", rdma_event_str(event->event));
    switch (event->event) {
    case RDMA_CM_EVENT_ADDR_RESOLVED:
        ret = rdma_resolve_route(event->id, g_options.conn_timeout_ms);
        if (ret) {
            LOG_ERROR("rdma_resolve_route() failed: %m");
            goto out_ack_event;
        }
        break;
    case RDMA_CM_EVENT_ROUTE_RESOLVED:
        ret = init_transport(event);
        if (ret) {
            goto out_ack_event;
        }

        ret = get_conn_param(event->id, &conn_param, &priv, 0);
        if (ret) {
            return ret;
        }

        ret = rdma_connect(event->id, &conn_param);
        if (ret) {
            LOG_ERROR("rdma_connect() failed: %m");
            return -1;
        }

        break;
    case RDMA_CM_EVENT_CONNECT_REQUEST:
        ret = handle_connect_request(event);
        if (ret) {
            goto out_ack_event;
        }
        break;
    case RDMA_CM_EVENT_ESTABLISHED:
        ret = handle_established(event);
        if (ret) {
            goto out_ack_event;
        }
        break;
    case RDMA_CM_EVENT_DISCONNECTED:
        ++g_test.num_disconnect;
        break;
    default:
        break;
    }

    ret = rdma_ack_cm_event(event);
    if (ret) {
        LOG_ERROR("rdma_ack_cm_event() failed: %m");
        return ret;
    }

    return 0;

out_ack_event:
    (void)rdma_ack_cm_event(event);
    return ret;
}

static enum rdma_port_space get_port_space()
{
    return RDMA_PS_UDP; //RDMA_PS_TCP for RC
}

static void dc_send_completion(const struct ibv_wc* wc)
{
    int dci = wc->wr_id;

    /* update outstanding sends counter and free dci bitmap */
    assert(wc->qp_num == g_test.dcis[dci]->qp_num);
    if (--g_test.dci_outstanding[dci] == 0) {
        g_test.free_dci_bitmap |= BIT(dci);
    }
}

static int poll_cq()
{
    packet_t *packet;
    struct ibv_wc wc;
    int ret;

    ret = ibv_poll_cq(g_test.cq, 1, &wc);
    if (ret < 0) {
        return ret;
    } else if (ret > 0) {
        if (wc.status != IBV_WC_SUCCESS) {
            LOG_ERROR("wr_id %ld: Completion with error: %s, vendor_err 0x%x",
                      wc.wr_id, ibv_wc_status_str(wc.status), wc.vendor_err);
            return -1;
        }

        switch (wc.opcode) {
        case IBV_WC_RECV:
            /* Packet was received */
            packet = (packet_t*)wc.wr_id;
            LOG_TRACE("Received packet %d conn_index %d at %p", packet->type,
                      packet->conn_index, packet);
            switch (packet->type) {
            case PACKET_FIN:
                ++g_test.num_fin_recvd;
            default:
                break;
            }
            break;
        case IBV_WC_SEND:
            /* Outgoing send was acknowledged on transport level */
            LOG_TRACE("SEND completion wr_id %ld", wc.wr_id);
            dc_send_completion(&wc);
            break;
        case IBV_WC_RDMA_READ:
            /* Outgoing RDMA_READ was completed */
            LOG_TRACE("RDMA_READ completion wr_id %ld", wc.wr_id);
            --g_test.num_outstanding_reads;
            dc_send_completion(&wc);
            break;
        default:
            LOG_ERROR("Unexpected completion opcode %d", wc.opcode);
            return -1;
        }
    }
    return 0;
}

static int disconnect()
{
    unsigned i;
    int ret;

    LOG_INFO("Disconnecting %d connections", g_test.num_conns);

    /* Send FIN messages on all connections */
    for (i = 0; i < g_test.num_conns; ++i) {
        ret = send_control(&g_test.conns[i], PACKET_FIN, i);
        if (ret) {
            return ret;
        }
    }

    /* Wait for FIN messages on all connections */
    while (g_test.num_fin_recvd < g_test.num_conns) {
        ret = poll_cq();
        if (ret) {
            return ret;
        }
    }

    return 0;
}

static int run_client()
{
    struct sockaddr_in dest_addr;
    struct rdma_cm_id *rdma_id;
    int i, ret;

    ret = get_address(&dest_addr);
    if (ret) {
        return ret;
    }

    for (i = 0; i < g_options.num_connections; ++i) {

        LOG_INFO("Connection[%d] to %s...", i, g_options.dest_address);

        ret = rdma_create_id(g_test.event_ch, &rdma_id, NULL, get_port_space());
        if (ret) {
            LOG_ERROR("rdma_create_id() failed: %m");
            return ret;
        }

        ret = rdma_resolve_addr(rdma_id, NULL, /* src_addr */
                                (struct sockaddr *)&dest_addr,
                                g_options.conn_timeout_ms);
        if (ret) {
            LOG_ERROR("rdma_resolve_addr() failed: %m");
            return -1;
        }
    }

    while (g_test.num_established < g_options.num_connections) {
        ret = wait_and_process_one_event(BIT(RDMA_CM_EVENT_ADDR_RESOLVED) |
                                         BIT(RDMA_CM_EVENT_ROUTE_RESOLVED) |
                                         BIT(RDMA_CM_EVENT_ESTABLISHED));
        if (ret) {
            return ret;
        }
    }

    ret = disconnect();
    if (ret) {
        return ret;
    }

    return 0;
}

/* post rdma_read, set *posted to 1 if the operation was posted, 0 if no
 * available QP is found
 */
static int post_rdma_read(connection_t *conn, size_t offset, size_t length,
                          int *posted)
{
    uintptr_t remote_addr,
              local_addr;
    int       ret,
              dci;

    if (!g_test.free_dci_bitmap) {
        *posted = 0;
        goto out;
    }

    /* find an available DCI */
    dci = __builtin_ffsl(g_test.free_dci_bitmap) - 1;
    assert(g_test.dci_outstanding[dci] == 0);

    remote_addr = conn->remote_addr + offset;
    local_addr  = (uintptr_t)g_test.rdma_buf.ptr + offset;

    LOG_TRACE("post_rdma_read: ibv_wr_start: qpex = %p\n", g_test.dcis_ex[dci]);
    ibv_wr_start(g_test.dcis_ex[dci]);

    g_test.dcis_ex[dci]->wr_id = (uint64_t)dci;
    g_test.dcis_ex[dci]->wr_flags = IBV_SEND_SIGNALED;

    LOG_TRACE("post_rdma_read: ibv_wr_rdma_read: wr_id=0x%lx, qpex=%p, rkey=0x%x, remote_addr=%p\n",
              g_test.dcis_ex[dci]->wr_id, g_test.dcis_ex[dci], conn->rkey, (void*)remote_addr);
    ibv_wr_rdma_read(g_test.dcis_ex[dci], conn->rkey, remote_addr);

    LOG_TRACE("post_rdma_read: mlx5dv_wr_set_dc_addr: mqpex=%p, ah=%p, rem_dctn=0x%06x\n",
               g_test.m_dcis_ex[dci], conn->dc_ah, conn->remote_dctn);
    mlx5dv_wr_set_dc_addr(g_test.m_dcis_ex[dci], conn->dc_ah, conn->remote_dctn, DC_KEY);

    LOG_TRACE("post_rdma_read: ibv_wr_set_sge: qpex=%p, lkey=0x%x, local_buf=%p, size=%lu\n",
              g_test.dcis_ex[dci], g_test.rdma_buf.mr->lkey, (void*)local_addr, length);
    ibv_wr_set_sge(g_test.dcis_ex[dci], g_test.rdma_buf.mr->lkey, local_addr, length);

    LOG_TRACE("post_rdma_read: ibv_wr_complete: qpex=%p\n", g_test.dcis_ex[dci]);
    ret = ibv_wr_complete(g_test.dcis_ex[dci]);
    if (ret) {
        LOG_ERROR("post_rdma_read: ibv_wr_complete failed (error=%d)\n", ret);
        return -1;
    }
    
    ++g_test.dci_outstanding[dci];
    g_test.free_dci_bitmap &= ~BIT(dci);
    *posted = 1;

    /* Update counters */
    if (*posted) {
        ++g_test.num_outstanding_reads;
        conn->read_offset += length;
    }

out:
    return 0;
}

static int do_rdma_reads()
{
    struct timeval tv_start, tv_end;
    connection_t *conn;
    double sec_elapsed;
    unsigned i, iter;
    size_t size, bytes_transferred;
    int posted;
    int ret;
    LIST_HEAD(sched);

    g_test.num_outstanding_reads = 0;

    /* measure the time over several iterations */
    gettimeofday(&tv_start, NULL);
    for (iter = 0; iter < g_options.num_iterations; ++iter) {
        /* insert all connections to the schedule queue */
        for (i = 0; i < g_test.num_conns; ++i) {
            g_test.conns[i].read_offset = 0;
            list_add_tail(&sched, &g_test.conns[i].list);
        }

        /* As long as not all read operations are completed, and the list is not
         * empty, continue working
         */
        while (g_test.num_outstanding_reads || !list_is_empty(&sched)) {

            //MB:
            if (!list_is_empty(&sched) &&
                (g_test.num_outstanding_reads == g_options.max_outstanding_reads)) {
                LOG_TRACE("num_outstanding_reads (%ld) reached maximum\n",
                          g_test.num_outstanding_reads);
            }
            /* If there are more connections with pending data, and we have not
             * exceeded the total number of outstanding reads, post the next
             * read operation
             */
            if (!list_is_empty(&sched) &&
                (g_test.num_outstanding_reads < g_options.max_outstanding_reads))
            {
                LOG_TRACE("num_outstanding_reads = %ld\n", g_test.num_outstanding_reads);
                /* take the first connection from the schedule queue */
                conn = list_extract_head(&sched, connection_t, list);

                /* see how many bytes are left to read */
                size = MIN(g_options.max_read_size,
                           g_options.rdma_read_size - conn->read_offset);
                assert(size > 0);

                /* issue rdma read on the next segment */
                posted = 0;
                ret = post_rdma_read(conn, conn->read_offset, size, &posted);
                if (ret < 0) {
                    return ret;
                }

                if (posted) {
                    /* if this connection is not done, reinsert it to the tail of
                     * the schedule queue
                     */
                    if (conn->read_offset < g_options.rdma_read_size) {
                        list_add_tail(&sched, &conn->list);
                    }
                } else {
                    /* If we could not sent, reinsert to the head of the queue */
                    list_add_head(&sched, &conn->list);
                }
            }

            ret = poll_cq();
            if (ret) {
                return ret;
            }
        }
    }
    gettimeofday(&tv_end, NULL);

    /* make sure everything was read */
    for (i = 0; i < g_test.num_conns; ++i) {
        assert(g_test.conns[i].read_offset == g_options.rdma_read_size);
    }

    /* Calculate and report total read bandwidth */
    sec_elapsed = (tv_end.tv_sec - tv_start.tv_sec) +
                  (tv_end.tv_usec - tv_start.tv_usec) * 1e-6;
    bytes_transferred = g_test.num_conns * 
                        g_options.num_iterations * g_options.rdma_read_size;
    LOG_INFO("Total read bandwidth: %.2f MB/s",
             bytes_transferred / sec_elapsed / BIT(20));
    return 0;
}

static int run_server()
{
    struct sockaddr_in in_addr;
    int ret;

    /* Make TX/RX queue lengths are large enough to send/recv control messages
     * on all connections without extra CQ polling (this is done for simplicity)
     */
    g_options.rx_queue_len = MAX(g_options.rx_queue_len,
                                 g_options.num_connections * 2);
    g_options.tx_queue_len = MAX(g_options.tx_queue_len,
                                 g_options.num_connections);

    ret = rdma_create_id(g_test.event_ch, &g_test.listen_cm_id, NULL,
                         get_port_space());
    if (ret) {
        LOG_ERROR("rdma_create_id() failed: %m");
        return ret;
    }

    /* Listen on INADDR_ANY */
    memset(&in_addr, 0, sizeof(in_addr));
    in_addr.sin_family      = AF_INET;
    in_addr.sin_addr.s_addr = INADDR_ANY;
    in_addr.sin_port        = htons(g_options.port_num);
    ret = rdma_bind_addr(g_test.listen_cm_id, (struct sockaddr*)&in_addr);
    if (ret) {
        LOG_ERROR("rdma_bind_addr() failed: %m");
        return ret;
    }

    ret = rdma_listen(g_test.listen_cm_id, g_options.conn_backlog);
    if (ret) {
        LOG_ERROR("rdma_listen() failed: %m");
        return ret;
    }

    LOG_INFO("Waiting for %d connections...", g_options.num_connections);
    while (g_test.num_established < g_options.num_connections) {
        /* For RC, wait until all connections got ESTABLISHED event */
        ret = wait_and_process_one_event(BIT(RDMA_CM_EVENT_CONNECT_REQUEST) |
                                         BIT(RDMA_CM_EVENT_ESTABLISHED) |
                                         BIT(RDMA_CM_EVENT_DISCONNECTED));
        if (ret) {
            return ret;
        }
    }

    ret = do_rdma_reads();
    if (ret) {
        return ret;
    }

    ret = disconnect();
    if (ret) {
        return ret;
    }

    return 0;
}

int main(int argc, char **argv)
{
    int ret;

    ret = parse_opts(argc, argv);
    if (ret) {
        return ret;
    }

    ret = init_test();
    if (ret) {
        return ret;
    }

    if (is_client()) {
        ret = run_client();
    } else {
        ret = run_server();
    }

    cleanup_test();

    return ret;
}
