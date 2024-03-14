#ifndef IBMSG_UTILITY_H
#define IBMSG_UTILITY_H

#include <fcntl.h>
#include <string.h>


#define IBMSG_DEBUG

#ifdef IBMSG_DEBUG
#  include <stdio.h>
#  define LOG(...) do { fprintf(stderr, "LOG: "__VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#else
#  define LOG(...)
#endif

#define CHECK_CALL(x, r)      \
  do {                        \
    int result = x;           \
    if(0 != result) return r; \
  } while(0)


#define IBMSG_MAX_EVENTS          (64)
#define IBMSG_TIMEOUT_MS         (100)
#define IBMSG_RESPONDER_RESOURCES  (1)
#define IBMSG_HW_FLOW_CONTROL      (0)
#define IBMSG_RETRY_COUNT          (7)
#define IBMSG_RNR_RETRY_COUNT      (7)
#define IBMSG_MIN_CQE              (1)
#define IBMSG_MAX_WR             (128)
#define IBMSG_MAX_SGE              (2)
#define IBMSG_MAX_INLINE         (256)
#define IBMSG_HOP_LIMIT          (255)
#define IBMSG_MAX_DEST_RD_ATOMIC   (1)
#define IBMSG_MAX_RD_ATOMIC        (1)


static void
init_qp_param(struct ibv_qp_init_attr* qp_init_attr)
{
    qp_init_attr->send_cq = NULL;
    qp_init_attr->recv_cq = NULL;
    qp_init_attr->srq = NULL;
    qp_init_attr->cap.max_send_wr = IBMSG_MAX_WR;
    qp_init_attr->cap.max_recv_wr = IBMSG_MAX_WR;
    qp_init_attr->cap.max_send_sge = IBMSG_MAX_SGE;
    qp_init_attr->cap.max_recv_sge = IBMSG_MAX_SGE;
    qp_init_attr->cap.max_inline_data = IBMSG_MAX_INLINE;
    qp_init_attr->qp_type = IBV_QPT_RC;
    qp_init_attr->sq_sig_all = 1;
}


static void
init_conn_param(struct rdma_conn_param* conn_param)
{
    memset(conn_param, 0, sizeof *conn_param);
    conn_param->responder_resources = IBMSG_RESPONDER_RESOURCES;
    conn_param->flow_control = IBMSG_HW_FLOW_CONTROL;
    conn_param->retry_count = IBMSG_RETRY_COUNT;
    conn_param->rnr_retry_count = IBMSG_RNR_RETRY_COUNT;
}


static int
make_nonblocking(int fd)
{
    /* change the blocking mode of the completion channel */
    int flags = fcntl(fd, F_GETFL);
    int rc = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    if (rc < 0) {
        LOG("Failed to change file descriptor of Completion Event Channel\n");
        return -1;
    }

    return 0;
}

#endif
