#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <malloc.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <time.h>
#include <inttypes.h>
#include <infiniband/verbs.h>
#include "pingpong.h"
#define PP_VERB_OPCODE_CLIENT IBV_WR_SEND_WITH_IMM
enum {
        PINGPONG_RECV_WRID = 1,
        PINGPONG_SEND_WRID = 2,
};

int min( int a, int b )
{
	    return a < b ? a : b;
}

int max( int a, int b )
{
	    return a > b ? a : b;
}
#if 0
enum ibv_mtu pp_mtu_to_enum(int mtu)
{
        switch (mtu) {
        case 256:  return IBV_MTU_256;
        case 512:  return IBV_MTU_512;
        case 1024: return IBV_MTU_1024;
        case 2048: return IBV_MTU_2048;
        case 4096: return IBV_MTU_4096;
        default:   return 0;
        }
}

int pp_get_port_info(struct ibv_context *context, int port,
                     struct ibv_port_attr *attr)
{
        return ibv_query_port(context, port, attr);
}

void wire_gid_to_gid(const char *wgid, union ibv_gid *gid)
{
        char tmp[9];
        __be32 v32;
        int i;
        uint32_t tmp_gid[4];

        for (tmp[8] = 0, i = 0; i < 4; ++i) {
                memcpy(tmp, wgid + i * 8, 8);
                sscanf(tmp, "%x", &v32);
                tmp_gid[i] = be32toh(v32);
        }
        memcpy(gid, tmp_gid, sizeof(*gid));
}

void gid_to_wire_gid(const union ibv_gid *gid, char wgid[])
{
        uint32_t tmp_gid[4];
        int i;

        memcpy(tmp_gid, gid, sizeof(tmp_gid));
        for (i = 0; i < 4; ++i)
                sprintf(&wgid[i * 8], "%08x", htobe32(tmp_gid[i]));
}

#endif
static int page_size;
static int use_odp;
static int implicit_odp;
static int prefetch_mr;
static int use_ts = 1;
static int validate_buf;
static int use_dm;
static int use_new_send;

struct pingpong_context {
        struct ibv_context      *context;
        struct ibv_comp_channel *channel;
        struct ibv_pd           *pd;
        struct ibv_mr           *mr;
        struct ibv_dm           *dm;
        union {
                struct ibv_cq           *cq;
                struct ibv_cq_ex        *cq_ex;
        } cq_s;
        struct ibv_qp           *qp;
        struct ibv_qp_ex        *qpx;
        char                    *buf;
        int                      size;
        int                      send_flags;
        int                      rx_depth;
        int                      pending;
        struct ibv_port_attr     portinfo;
        uint64_t                 completion_timestamp_mask;
};

static struct ibv_cq *pp_cq(struct pingpong_context *ctx)
{
        return use_ts ? ibv_cq_ex_to_cq(ctx->cq_s.cq_ex) :
                ctx->cq_s.cq;
}

struct pingpong_dest {
        int lid;
        int qpn;
        int psn;
        union ibv_gid gid;
};

#if 0
static int pp_connect_ctx(struct pingpong_context *ctx, int port, int my_psn,
                          enum ibv_mtu mtu, int sl,
                          struct pingpong_dest *dest, int sgid_idx)
{
        struct ibv_qp_attr attr = {
                .qp_state               = IBV_QPS_RTR,
                .path_mtu               = mtu,
                .dest_qp_num            = dest->qpn,
                .rq_psn                 = dest->psn,
                .max_dest_rd_atomic     = 1,
                .min_rnr_timer          = 12,
                .ah_attr                = {
                        .is_global      = 0,
                        .dlid           = dest->lid,
                        .sl             = sl,
                        .src_path_bits  = 0,
                        .port_num       = port
                }
        };

        if (dest->gid.global.interface_id) {
                attr.ah_attr.is_global = 1;
                attr.ah_attr.grh.hop_limit = 1;
                attr.ah_attr.grh.dgid = dest->gid;
                attr.ah_attr.grh.sgid_index = sgid_idx;
        }
        if (ibv_modify_qp(ctx->qp, &attr,
                          IBV_QP_STATE              |
                          IBV_QP_AV                 |
                          IBV_QP_PATH_MTU           |
                          IBV_QP_DEST_QPN           |
                          IBV_QP_RQ_PSN             |
                          IBV_QP_MAX_DEST_RD_ATOMIC |
                          IBV_QP_MIN_RNR_TIMER)) {
                fprintf(stderr, "Failed to modify QP to RTR\n");
                return 1;
        }

        attr.qp_state       = IBV_QPS_RTS;
        attr.timeout        = 14;
        attr.retry_cnt      = 7;
        attr.rnr_retry      = 7;
        attr.sq_psn         = my_psn;
        attr.max_rd_atomic  = 1;
        if (ibv_modify_qp(ctx->qp, &attr,
                          IBV_QP_STATE              |
                          IBV_QP_TIMEOUT            |
                          IBV_QP_RETRY_CNT          |
                          IBV_QP_RNR_RETRY          |
                          IBV_QP_SQ_PSN             |
                          IBV_QP_MAX_QP_RD_ATOMIC)) {
                fprintf(stderr, "Failed to modify QP to RTS\n");
                return 1;
        }

        return 0;
}
#else
static int pp_connect_ctx(struct pingpong_context *ctx, int port, int my_psn,
			  enum ibv_mtu mtu, int sl, struct pingpong_dest *dest,
			  int sgid_idx)
{
	struct ibv_qp_attr attr = { .qp_state = IBV_QPS_RTR,
				    .path_mtu = mtu,
				    .dest_qp_num = dest->qpn,
				    .rq_psn = dest->psn,
				    .ah_attr = { .is_global = 0,
						 .dlid = dest->lid,
						 .sl = sl,
						 .src_path_bits = 0,
						 .port_num = port } };

	if (dest->gid.global.interface_id) {
		attr.ah_attr.is_global = 1;
		attr.ah_attr.grh.hop_limit = 1;
		attr.ah_attr.grh.dgid = dest->gid;
		attr.ah_attr.grh.sgid_index = sgid_idx;
	}
	if (ibv_modify_qp(ctx->qp, &attr,
			  IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
				  IBV_QP_DEST_QPN | IBV_QP_RQ_PSN)) {
		fprintf(stderr, "Failed to modify QP to RTR\n");
		return 1;
	}

	attr.qp_state = IBV_QPS_RTS;
	attr.sq_psn = my_psn;
	if (ibv_modify_qp(ctx->qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
		fprintf(stderr, "Failed to modify QP to RTS\n");
		return 1;
	}

	return 0;
}

#endif

static struct pingpong_dest *pp_client_exch_dest(const char *servername, int port,
                                                 const struct pingpong_dest *my_dest)
{
        struct addrinfo *res, *t;
        struct addrinfo hints = {
                .ai_family   = AF_UNSPEC,
                .ai_socktype = SOCK_STREAM
        };
        char *service;
        char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
        int n;
        int sockfd = -1;
        struct pingpong_dest *rem_dest = NULL;
        char gid[33];

        if (asprintf(&service, "%d", port) < 0)
                return NULL;

        n = getaddrinfo(servername, service, &hints, &res);

        if (n < 0) {
                fprintf(stderr, "%s for %s:%d\n", gai_strerror(n), servername, port);
                free(service);
                return NULL;
        }

        for (t = res; t; t = t->ai_next) {
                sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
                if (sockfd >= 0) {
                        if (!connect(sockfd, t->ai_addr, t->ai_addrlen))
                                break;
                        close(sockfd);
                        sockfd = -1;
                }
        }

        freeaddrinfo(res);
        free(service);

        if (sockfd < 0) {
                fprintf(stderr, "Couldn't connect to %s:%d\n", servername, port);
                return NULL;
        }

        gid_to_wire_gid(&my_dest->gid, gid);
        sprintf(msg, "%04x:%06x:%06x:%s", my_dest->lid, my_dest->qpn,
                                                        my_dest->psn, gid);
        if (write(sockfd, msg, sizeof msg) != sizeof msg) {
                fprintf(stderr, "Couldn't send local address\n");
                goto out;
        }

        if (read(sockfd, msg, sizeof msg) != sizeof msg ||
            write(sockfd, "done", sizeof "done") != sizeof "done") {
                perror("client read/write");
                fprintf(stderr, "Couldn't read/write remote address\n");
                goto out;
        }

        rem_dest = malloc(sizeof *rem_dest);
        if (!rem_dest)
                goto out;

        sscanf(msg, "%x:%x:%x:%s", &rem_dest->lid, &rem_dest->qpn,
                                                &rem_dest->psn, gid);
        wire_gid_to_gid(gid, &rem_dest->gid);

out:
        close(sockfd);
        return rem_dest;
}

static struct pingpong_dest *pp_server_exch_dest(struct pingpong_context *ctx,
                                                 int ib_port, enum ibv_mtu mtu,
                                                 int port, int sl,
                                                 const struct pingpong_dest *my_dest,
                                                 int sgid_idx)
{
        struct addrinfo *res, *t;
        struct addrinfo hints = {
                .ai_flags    = AI_PASSIVE,
                .ai_family   = AF_UNSPEC,
                .ai_socktype = SOCK_STREAM
        };
        char *service;
        char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
        int n;
        int sockfd = -1, connfd;
        struct pingpong_dest *rem_dest = NULL;
        char gid[33];

        if (asprintf(&service, "%d", port) < 0)
                return NULL;

        n = getaddrinfo(NULL, service, &hints, &res);

        if (n < 0) {
                fprintf(stderr, "%s for port %d\n", gai_strerror(n), port);
                free(service);
                return NULL;
        }

        for (t = res; t; t = t->ai_next) {
                sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
                if (sockfd >= 0) {
                        n = 1;

                        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &n, sizeof n);

                        if (!bind(sockfd, t->ai_addr, t->ai_addrlen))
                                break;
                        close(sockfd);
                        sockfd = -1;
                }
        }

        freeaddrinfo(res);
        free(service);

        if (sockfd < 0) {
                fprintf(stderr, "Couldn't listen to port %d\n", port);
                return NULL;
        }

        listen(sockfd, 1);
        connfd = accept(sockfd, NULL, NULL);
        close(sockfd);
        if (connfd < 0) {
                fprintf(stderr, "accept() failed\n");
                return NULL;
        }

        n = read(connfd, msg, sizeof msg);
        if (n != sizeof msg) {
                perror("server read");
                fprintf(stderr, "%d/%d: Couldn't read remote address\n", n, (int) sizeof msg);
                goto out;
        }

        rem_dest = malloc(sizeof *rem_dest);
        if (!rem_dest)
                goto out;

        sscanf(msg, "%x:%x:%x:%s", &rem_dest->lid, &rem_dest->qpn,
                                                        &rem_dest->psn, gid);
        wire_gid_to_gid(gid, &rem_dest->gid);

        if (pp_connect_ctx(ctx, ib_port, my_dest->psn, mtu, sl, rem_dest,
                                                                sgid_idx)) {
                fprintf(stderr, "Couldn't connect to remote QP\n");
                free(rem_dest);
                rem_dest = NULL;
                goto out;
        }


        gid_to_wire_gid(&my_dest->gid, gid);
        sprintf(msg, "%04x:%06x:%06x:%s", my_dest->lid, my_dest->qpn,
                                                        my_dest->psn, gid);
        if (write(connfd, msg, sizeof msg) != sizeof msg ||
            read(connfd, msg, sizeof msg) != sizeof "done") {
                fprintf(stderr, "Couldn't send/recv local address\n");
                free(rem_dest);
                rem_dest = NULL;
                goto out;
        }


out:
        close(connfd);
        return rem_dest;
}

static struct pingpong_context *pp_init_ctx(struct ibv_device *ib_dev, int size,
                                            int rx_depth, int port,
                                            int use_event)
{
        struct pingpong_context *ctx;
        int access_flags = IBV_ACCESS_LOCAL_WRITE;

        ctx = calloc(1, sizeof *ctx);
        if (!ctx)
                return NULL;

        ctx->size       = size;
        ctx->send_flags = IBV_SEND_SIGNALED;
        ctx->rx_depth   = rx_depth;

        ctx->buf = memalign(page_size, size);
        if (!ctx->buf) {
                fprintf(stderr, "Couldn't allocate work buf.\n");
                goto clean_ctx;
        }

        /* FIXME memset(ctx->buf, 0, size); */
        memset(ctx->buf, 0x7b, size);

        ctx->context = ibv_open_device(ib_dev);
        if (!ctx->context) {
                fprintf(stderr, "Couldn't get context for %s\n",
                        ibv_get_device_name(ib_dev));
                goto clean_buffer;
        }

        if (use_event) {
                ctx->channel = ibv_create_comp_channel(ctx->context);
                if (!ctx->channel) {
                        fprintf(stderr, "Couldn't create completion channel\n");
                        goto clean_device;
                }
        } else
                ctx->channel = NULL;

        ctx->pd = ibv_alloc_pd(ctx->context);
        if (!ctx->pd) {
                fprintf(stderr, "Couldn't allocate PD\n");
                goto clean_comp_channel;
        }

        if (use_odp || use_ts || use_dm) {
                const uint32_t rc_caps_mask = IBV_ODP_SUPPORT_SEND |
                                              IBV_ODP_SUPPORT_RECV;
                struct ibv_device_attr_ex attrx;

                if (ibv_query_device_ex(ctx->context, NULL, &attrx)) {
                        fprintf(stderr, "Couldn't query device for its features\n");
                        goto clean_pd;
                }

                if (use_odp) {
                        if (!(attrx.odp_caps.general_caps & IBV_ODP_SUPPORT) ||
                            (attrx.odp_caps.per_transport_caps.rc_odp_caps & rc_caps_mask) != rc_caps_mask) {
                                fprintf(stderr, "The device isn't ODP capable or does not support RC send and receive with ODP\n");
                                goto clean_pd;
                        }
                        if (implicit_odp &&
                            !(attrx.odp_caps.general_caps & IBV_ODP_SUPPORT_IMPLICIT)) {
                                fprintf(stderr, "The device doesn't support implicit ODP\n");
                                goto clean_pd;
                        }
                        access_flags |= IBV_ACCESS_ON_DEMAND;
                }

                if (use_ts) {
                        if (!attrx.completion_timestamp_mask) {
                                fprintf(stderr, "The device isn't completion timestamp capable\n");
                                goto clean_pd;
                        }
                        ctx->completion_timestamp_mask = attrx.completion_timestamp_mask;
                }

                if (use_dm) {
                        struct ibv_alloc_dm_attr dm_attr = {};

                        if (!attrx.max_dm_size) {
                                fprintf(stderr, "Device doesn't support dm allocation\n");
                                goto clean_pd;
                        }

                        if (attrx.max_dm_size < size) {
                                fprintf(stderr, "Device memory is insufficient\n");
                                goto clean_pd;
                        }

                        dm_attr.length = size;
                        ctx->dm = ibv_alloc_dm(ctx->context, &dm_attr);
                        if (!ctx->dm) {
                                fprintf(stderr, "Dev mem allocation failed\n");
                                goto clean_pd;
                        }

                        access_flags |= IBV_ACCESS_ZERO_BASED;
                }
        }

        if (implicit_odp) {
                ctx->mr = ibv_reg_mr(ctx->pd, NULL, SIZE_MAX, access_flags);
        } else {
                ctx->mr = use_dm ? ibv_reg_dm_mr(ctx->pd, ctx->dm, 0,
                                                 size, access_flags) :
                        ibv_reg_mr(ctx->pd, ctx->buf, size, access_flags);
        }

        if (!ctx->mr) {
                fprintf(stderr, "Couldn't register MR\n");
                goto clean_dm;
        }

        if (prefetch_mr) {
                struct ibv_sge sg_list;
                int ret;

                sg_list.lkey = ctx->mr->lkey;
                sg_list.addr = (uintptr_t)ctx->buf;
                sg_list.length = size;

                ret = ibv_advise_mr(ctx->pd, IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE,
                                    IB_UVERBS_ADVISE_MR_FLAG_FLUSH,
                                    &sg_list, 1);

                if (ret)
                        fprintf(stderr, "Couldn't prefetch MR(%d). Continue anyway\n", ret);
        }

        if (use_ts) {
                struct ibv_cq_init_attr_ex attr_ex = {
                        .cqe = rx_depth + 1,
                        .cq_context = NULL,
                        .channel = ctx->channel,
                        .comp_vector = 0,
                        .wc_flags = IBV_WC_EX_WITH_COMPLETION_TIMESTAMP  |  IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM | IBV_WC_EX_WITH_QP_NUM |
				          IBV_WC_EX_WITH_SRC_QP
                };

                ctx->cq_s.cq_ex = ibv_create_cq_ex(ctx->context, &attr_ex);
        } else {
                ctx->cq_s.cq = ibv_create_cq(ctx->context, rx_depth + 1, NULL,
                                             ctx->channel, 0);
        }

        if (!pp_cq(ctx)) {
                fprintf(stderr, "Couldn't create CQ\n");
                goto clean_mr;
        }

        {
                struct ibv_qp_attr attr;
                struct ibv_qp_init_attr init_attr = {
                        .send_cq = pp_cq(ctx),
                        .recv_cq = pp_cq(ctx),
                        .cap     = {
                                .max_send_wr  = 1,
                                .max_recv_wr  = rx_depth,
                                .max_send_sge = 1,
                                .max_recv_sge = 1
                        },
                        .qp_type = IBV_QPT_UC
                };

                if (use_new_send) {
                        struct ibv_qp_init_attr_ex init_attr_ex = {};

                        init_attr_ex.send_cq = pp_cq(ctx);
                        init_attr_ex.recv_cq = pp_cq(ctx);
                        init_attr_ex.cap.max_send_wr = 1;
                        init_attr_ex.cap.max_recv_wr = rx_depth;
                        init_attr_ex.cap.max_send_sge = 1;
                        init_attr_ex.cap.max_recv_sge = 1;
                        init_attr_ex.qp_type = IBV_QPT_UC;

                        init_attr_ex.comp_mask |= IBV_QP_INIT_ATTR_PD |
                                                  IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
                        init_attr_ex.pd = ctx->pd;
                        init_attr_ex.send_ops_flags = IBV_QP_EX_WITH_SEND;

                        ctx->qp = ibv_create_qp_ex(ctx->context, &init_attr_ex);
                } else {
                        ctx->qp = ibv_create_qp(ctx->pd, &init_attr);
                }

                if (!ctx->qp)  {
                        fprintf(stderr, "Couldn't create QP\n");
                        goto clean_cq;
                }

                if (use_new_send)
                        ctx->qpx = ibv_qp_to_qp_ex(ctx->qp);

                ibv_query_qp(ctx->qp, &attr, IBV_QP_CAP, &init_attr);
                if (init_attr.cap.max_inline_data >= size && !use_dm)
                        ctx->send_flags |= IBV_SEND_INLINE;
        }

        {
                struct ibv_qp_attr attr = {
                        .qp_state        = IBV_QPS_INIT,
                        .pkey_index      = 0,
                        .port_num        = port,
                        .qp_access_flags = 0
                };

                if (ibv_modify_qp(ctx->qp, &attr,
                                  IBV_QP_STATE              |
                                  IBV_QP_PKEY_INDEX         |
                                  IBV_QP_PORT               |
                                  IBV_QP_ACCESS_FLAGS)) {
                        fprintf(stderr, "Failed to modify QP to INIT\n");
                        goto clean_qp;
                }
        }

        return ctx;

clean_qp:
        ibv_destroy_qp(ctx->qp);

clean_cq:
        ibv_destroy_cq(pp_cq(ctx));

clean_mr:
        ibv_dereg_mr(ctx->mr);

clean_dm:
        if (ctx->dm)
                ibv_free_dm(ctx->dm);

clean_pd:
        ibv_dealloc_pd(ctx->pd);

clean_comp_channel:
        if (ctx->channel)
                ibv_destroy_comp_channel(ctx->channel);

clean_device:
        ibv_close_device(ctx->context);

clean_buffer:
        free(ctx->buf);

clean_ctx:
        free(ctx);

        return NULL;
}

static int pp_close_ctx(struct pingpong_context *ctx)
{
        if (ibv_destroy_qp(ctx->qp)) {
                fprintf(stderr, "Couldn't destroy QP\n");
                return 1;
        }

        if (ibv_destroy_cq(pp_cq(ctx))) {
                fprintf(stderr, "Couldn't destroy CQ\n");
                return 1;
        }

        if (ibv_dereg_mr(ctx->mr)) {
                fprintf(stderr, "Couldn't deregister MR\n");
                return 1;
        }

        if (ctx->dm) {
                if (ibv_free_dm(ctx->dm)) {
                        fprintf(stderr, "Couldn't free DM\n");
                        return 1;
                }
        }

        if (ibv_dealloc_pd(ctx->pd)) {
                fprintf(stderr, "Couldn't deallocate PD\n");
                return 1;
        }

        if (ctx->channel) {
                if (ibv_destroy_comp_channel(ctx->channel)) {
                        fprintf(stderr, "Couldn't destroy completion channel\n");
                        return 1;
                }
        }

        if (ibv_close_device(ctx->context)) {
                fprintf(stderr, "Couldn't release context\n");
                return 1;
        }

        free(ctx->buf);
        free(ctx);

        return 0;
}

static int pp_post_recv(struct pingpong_context *ctx, int n)
{
        struct ibv_sge list = {
                .addr   = use_dm ? 0 : (uintptr_t) ctx->buf,
                .length = ctx->size,
                .lkey   = ctx->mr->lkey
        };
        struct ibv_recv_wr wr = {
                .wr_id      = PINGPONG_RECV_WRID,
                .sg_list    = &list,
                .num_sge    = 1,
        };
        struct ibv_recv_wr *bad_wr;
        int i;

        for (i = 0; i < n; ++i)
                if (ibv_post_recv(ctx->qp, &wr, &bad_wr))
                        break;

        return i;
}

static int pp_post_send(struct pingpong_context *ctx)
{
        struct ibv_sge list = {
                .addr   = use_dm ? 0 : (uintptr_t) ctx->buf,
                .length = ctx->size,
                .lkey   = ctx->mr->lkey
        };
        struct ibv_send_wr wr = {
                .wr_id      = PINGPONG_SEND_WRID,
                .sg_list    = &list,
		.imm_data =  htonl(0x10203040),
                .num_sge    = 1,
                .opcode     = PP_VERB_OPCODE_CLIENT,
                .send_flags = ctx->send_flags,
        };
#if 0
	if (wr.opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
		wr.wr.rdma.remote_addr = (uint64_t)peer->addr[i];
		wr.wr.rdma.rkey = peer->mrkey[i];
	}
#endif
        struct ibv_send_wr *bad_wr;

        if (use_new_send) {
                ibv_wr_start(ctx->qpx);

                ctx->qpx->wr_id = PINGPONG_SEND_WRID;
                ctx->qpx->wr_flags = ctx->send_flags;

                ibv_wr_send(ctx->qpx);
                ibv_wr_set_sge(ctx->qpx, list.lkey, list.addr, list.length);

                return ibv_wr_complete(ctx->qpx);
        } else {
                return ibv_post_send(ctx->qp, &wr, &bad_wr);
        }
}

struct ts_params {
        uint64_t                 comp_recv_max_time_delta;
        uint64_t                 comp_recv_min_time_delta;
        uint64_t                 comp_recv_total_time_delta;
        uint64_t                 comp_recv_prev_time;
        int                      last_comp_with_ts;
        unsigned int             comp_with_time_iters;
};

static inline int parse_single_wc(struct pingpong_context *ctx, int *scnt,
                                  int *rcnt, int *routs, int iters,
                                  uint64_t wr_id, uint32_t imm, enum ibv_wc_status status,
                                  uint64_t completion_timestamp,
                                  struct ts_params *ts)
{
        if (status != IBV_WC_SUCCESS) {
                fprintf(stderr, "Failed status %s (%d) for wr_id %d\n",
                        ibv_wc_status_str(status),
                        status, (int)wr_id);
                return 1;
        }

        switch ((int)wr_id) {
        case PINGPONG_SEND_WRID:
                ++(*scnt);
                break;

        case PINGPONG_RECV_WRID:
		printf("imm data is 0x%x \n",imm);
                if (--(*routs) <= 1) {
                        *routs += pp_post_recv(ctx, ctx->rx_depth - *routs);
                        if (*routs < ctx->rx_depth) {
                                fprintf(stderr,
                                        "Couldn't post receive (%d)\n",
                                        *routs);
                                return 1;
                        }
                }

                ++(*rcnt);
                if (use_ts) {
                        if (ts->last_comp_with_ts) {
                                uint64_t delta;

                                /* checking whether the clock was wrapped around */
                                if (completion_timestamp >= ts->comp_recv_prev_time)
                                        delta = completion_timestamp - ts->comp_recv_prev_time;
                                else
                                        delta = ctx->completion_timestamp_mask - ts->comp_recv_prev_time +
                                                completion_timestamp + 1;

                                ts->comp_recv_max_time_delta = max(ts->comp_recv_max_time_delta, delta);
                                ts->comp_recv_min_time_delta = min(ts->comp_recv_min_time_delta, delta);
                                ts->comp_recv_total_time_delta += delta;
                                ts->comp_with_time_iters++;
                        }

                        ts->comp_recv_prev_time = completion_timestamp;
                        ts->last_comp_with_ts = 1;
                } else {
                        ts->last_comp_with_ts = 0;
                }

                break;

        default:
                fprintf(stderr, "Completion for unknown wr_id %d\n",
                        (int)wr_id);
                return 1;
        }

        ctx->pending &= ~(int)wr_id;
        if (*scnt < iters && !ctx->pending) {
                if (pp_post_send(ctx)) {
                        fprintf(stderr, "Couldn't post send\n");
                        return 1;
                }
                ctx->pending = PINGPONG_RECV_WRID |
                        PINGPONG_SEND_WRID;
        }

        return 0;
}

static void usage(const char *argv0)
{
        printf("Usage:\n");
        printf("  %s            start a server and wait for connection\n", argv0);
        printf("  %s <host>     connect to server at <host>\n", argv0);
        printf("\n");
        printf("Options:\n");
        printf("  -p, --port=<port>      listen on/connect to port <port> (default 18515)\n");
        printf("  -d, --ib-dev=<dev>     use IB device <dev> (default first device found)\n");
        printf("  -i, --ib-port=<port>   use port <port> of IB device (default 1)\n");
        printf("  -s, --size=<size>      size of message to exchange (default 4096)\n");
        printf("  -m, --mtu=<size>       path MTU (default 1024)\n");
        printf("  -r, --rx-depth=<dep>   number of receives to post at a time (default 500)\n");
        printf("  -n, --iters=<iters>    number of exchanges (default 1000)\n");
        printf("  -l, --sl=<sl>          service level value\n");
        printf("  -e, --events           sleep on CQ events (default poll)\n");
        printf("  -g, --gid-idx=<gid index> local port gid index\n");
        printf("  -o, --odp                 use on demand paging\n");
        printf("  -O, --iodp                use implicit on demand paging\n");
        printf("  -P, --prefetch            prefetch an ODP MR\n");
        printf("  -t, --ts                  get CQE with timestamp\n");
        printf("  -c, --chk                 validate received buffer\n");
        printf("  -j, --dm                  use device memory\n");
        printf("  -N, --new_send            use new post send WR API\n");
}

int main(int argc, char *argv[])
{
        struct ibv_device      **dev_list;
        struct ibv_device       *ib_dev;
        struct pingpong_context *ctx;
        struct pingpong_dest     my_dest;
        struct pingpong_dest    *rem_dest;
        struct timeval           start, end;
        char                    *ib_devname = NULL;
        char                    *servername = NULL;
        unsigned int             port = 18515;
        int                      ib_port = 1;
        unsigned int             size = 4096;
        enum ibv_mtu             mtu = IBV_MTU_1024;
        unsigned int             rx_depth = 500;
        unsigned int             iters = 1000;
        int                      use_event = 0;
        int                      routs;
        int                      rcnt, scnt;
        int                      num_cq_events = 0;
        int                      sl = 0;
        int                      gidx = -1;
        char                     gid[33];
        struct ts_params         ts;

        srand48(getpid() * time(NULL));

        while (1) {
                int c;

                static struct option long_options[] = {
                        { .name = "port",     .has_arg = 1, .val = 'p' },
                        { .name = "ib-dev",   .has_arg = 1, .val = 'd' },
                        { .name = "ib-port",  .has_arg = 1, .val = 'i' },
                        { .name = "size",     .has_arg = 1, .val = 's' },
                        { .name = "mtu",      .has_arg = 1, .val = 'm' },
                        { .name = "rx-depth", .has_arg = 1, .val = 'r' },
                        { .name = "iters",    .has_arg = 1, .val = 'n' },
                        { .name = "sl",       .has_arg = 1, .val = 'l' },
                        { .name = "events",   .has_arg = 0, .val = 'e' },
                        { .name = "gid-idx",  .has_arg = 1, .val = 'g' },
                        { .name = "odp",      .has_arg = 0, .val = 'o' },
                        { .name = "iodp",     .has_arg = 0, .val = 'O' },
                        { .name = "prefetch", .has_arg = 0, .val = 'P' },
                        { .name = "ts",       .has_arg = 0, .val = 't' },
                        { .name = "chk",      .has_arg = 0, .val = 'c' },
                        { .name = "dm",       .has_arg = 0, .val = 'j' },
                        { .name = "new_send", .has_arg = 0, .val = 'N' },
                        {}
                };

                c = getopt_long(argc, argv, "p:d:i:s:m:r:n:l:eg:oOPtcjN",
                                long_options, NULL);

                if (c == -1)
                        break;

                switch (c) {
                case 'p':
                        port = strtoul(optarg, NULL, 0);
                        if (port > 65535) {
                                usage(argv[0]);
                                return 1;
                        }
                        break;

                case 'd':
                        ib_devname = strdupa(optarg);
                        break;

                case 'i':
                        ib_port = strtol(optarg, NULL, 0);
                        if (ib_port < 1) {
                                usage(argv[0]);
                                return 1;
                        }
                        break;

                case 's':
                        size = strtoul(optarg, NULL, 0);
                        break;

                case 'm':
                        mtu = pp_mtu_to_enum(strtol(optarg, NULL, 0));
                        if (mtu == 0) {
                                usage(argv[0]);
                                return 1;
                        }
                        break;

                case 'r':
                        rx_depth = strtoul(optarg, NULL, 0);
                        break;

                case 'n':
                        iters = strtoul(optarg, NULL, 0);
                        break;

                case 'l':
                        sl = strtol(optarg, NULL, 0);
                        break;

                case 'e':
                        ++use_event;
                        break;

                case 'g':
                        gidx = strtol(optarg, NULL, 0);
                        break;

                case 'o':
                        use_odp = 1;
                        break;
                case 'P':
                        prefetch_mr = 1;
                        break;
                case 'O':
                        use_odp = 1;
                        implicit_odp = 1;
                        break;
                case 't':
                        use_ts = 1;
                        break;
                case 'c':
                        validate_buf = 1;
                        break;

                case 'j':
                        use_dm = 1;
                        break;

                case 'N':
                        use_new_send = 1;
                        break;

                default:
                        usage(argv[0]);
                        return 1;
                }
        }

        if (optind == argc - 1)
                servername = strdupa(argv[optind]);
        else if (optind < argc) {
                usage(argv[0]);
                return 1;
        }

        if (use_odp && use_dm) {
                fprintf(stderr, "DM memory region can't be on demand\n");
                return 1;
        }

        if (!use_odp && prefetch_mr) {
                fprintf(stderr, "prefetch is valid only with on-demand memory region\n");
                return 1;
        }

        if (use_ts) {
                ts.comp_recv_max_time_delta = 0;
                ts.comp_recv_min_time_delta = 0xffffffff;
                ts.comp_recv_total_time_delta = 0;
                ts.comp_recv_prev_time = 0;
                ts.last_comp_with_ts = 0;
                ts.comp_with_time_iters = 0;
        }

        page_size = sysconf(_SC_PAGESIZE);

        dev_list = ibv_get_device_list(NULL);
        if (!dev_list) {
                perror("Failed to get IB devices list");
                return 1;
        }

        if (!ib_devname) {
                ib_dev = *dev_list;
                if (!ib_dev) {
                        fprintf(stderr, "No IB devices found\n");
                        return 1;
                }
        } else {
                int i;
                for (i = 0; dev_list[i]; ++i)
                        if (!strcmp(ibv_get_device_name(dev_list[i]), ib_devname))
                                break;
                ib_dev = dev_list[i];
                if (!ib_dev) {
                        fprintf(stderr, "IB device %s not found\n", ib_devname);
                        return 1;
                }
        }

        ctx = pp_init_ctx(ib_dev, size, rx_depth, ib_port, use_event);
        if (!ctx)
                return 1;

	if(NULL == servername) {
            memset(ctx->buf, 0x7a, size);
        }
        routs = pp_post_recv(ctx, ctx->rx_depth);
        if (routs < ctx->rx_depth) {
                fprintf(stderr, "Couldn't post receive (%d)\n", routs);
                return 1;
        }

        if (use_event)
                if (ibv_req_notify_cq(pp_cq(ctx), 0)) {
                        fprintf(stderr, "Couldn't request CQ notification\n");
                        return 1;
                }


        if (pp_get_port_info(ctx->context, ib_port, &ctx->portinfo)) {
                fprintf(stderr, "Couldn't get port info\n");
                return 1;
        }

        my_dest.lid = ctx->portinfo.lid;
        if (ctx->portinfo.link_layer != IBV_LINK_LAYER_ETHERNET &&
                                                        !my_dest.lid) {
                fprintf(stderr, "Couldn't get local LID\n");
                return 1;
        }

        if (gidx >= 0) {
                if (ibv_query_gid(ctx->context, ib_port, gidx, &my_dest.gid)) {
                        fprintf(stderr, "can't read sgid of index %d\n", gidx);
                        return 1;
                }
        } else
                memset(&my_dest.gid, 0, sizeof my_dest.gid);

        my_dest.qpn = ctx->qp->qp_num;
        my_dest.psn = lrand48() & 0xffffff;
        inet_ntop(AF_INET6, &my_dest.gid, gid, sizeof gid);
        printf("  local address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
               my_dest.lid, my_dest.qpn, my_dest.psn, gid);


        if (servername)
                rem_dest = pp_client_exch_dest(servername, port, &my_dest);
        else
                rem_dest = pp_server_exch_dest(ctx, ib_port, mtu, port, sl,
                                                                &my_dest, gidx);

        if (!rem_dest)
                return 1;

        inet_ntop(AF_INET6, &rem_dest->gid, gid, sizeof gid);
        printf("  remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
               rem_dest->lid, rem_dest->qpn, rem_dest->psn, gid);

        if (servername)
                if (pp_connect_ctx(ctx, ib_port, my_dest.psn, mtu, sl, rem_dest,
                                        gidx))
                        return 1;

        ctx->pending = PINGPONG_RECV_WRID;

        if (servername) {
                if (validate_buf)
                        for (int i = 0; i < size; i += page_size)
                                ctx->buf[i] = i / page_size % sizeof(char);

                if (use_dm)
                        if (ibv_memcpy_to_dm(ctx->dm, 0, (void *)ctx->buf, size)) {
                                fprintf(stderr, "Copy to dm buffer failed\n");
                                return 1;
                        }

                if (pp_post_send(ctx)) {
                        fprintf(stderr, "Couldn't post send\n");
                        return 1;
                }
                ctx->pending |= PINGPONG_SEND_WRID;
        }

        if (gettimeofday(&start, NULL)) {
                perror("gettimeofday");
                return 1;
        }

        rcnt = scnt = 0;
        while (rcnt < iters || scnt < iters) {
                int ret;

                if (use_event) {
                        struct ibv_cq *ev_cq;
                        void          *ev_ctx;

                        if (ibv_get_cq_event(ctx->channel, &ev_cq, &ev_ctx)) {
                                fprintf(stderr, "Failed to get cq_event\n");
                                return 1;
                        }

                        ++num_cq_events;

                        if (ev_cq != pp_cq(ctx)) {
                                fprintf(stderr, "CQ event for unknown CQ %p\n", ev_cq);
                                return 1;
                        }

                        if (ibv_req_notify_cq(pp_cq(ctx), 0)) {
                                fprintf(stderr, "Couldn't request CQ notification\n");
                                return 1;
                        }
                }

                if (use_ts) {
                        struct ibv_poll_cq_attr attr = {};

                        do {
                                ret = ibv_start_poll(ctx->cq_s.cq_ex, &attr);
                        } while (!use_event && ret == ENOENT);

                        if (ret) {
                                fprintf(stderr, "poll CQ failed %d\n", ret);
                                return ret;
                        }
                        ret = parse_single_wc(ctx, &scnt, &rcnt, &routs,
                                              iters,
                                              ctx->cq_s.cq_ex->wr_id,
					      ntohl(ibv_wc_read_imm_data(ctx->cq_s.cq_ex)),
                                              ctx->cq_s.cq_ex->status,
                                              ibv_wc_read_completion_ts(ctx->cq_s.cq_ex),
                                              &ts);
                        if (ret) {
                                ibv_end_poll(ctx->cq_s.cq_ex);
                                return ret;
                        }
                        ret = ibv_next_poll(ctx->cq_s.cq_ex);
                        if (!ret)
                                ret = parse_single_wc(ctx, &scnt, &rcnt, &routs,
                                                      iters,
                                                      ctx->cq_s.cq_ex->wr_id,
					              ntohl(ibv_wc_read_imm_data(ctx->cq_s.cq_ex)),
                                                      ctx->cq_s.cq_ex->status,
                                                      ibv_wc_read_completion_ts(ctx->cq_s.cq_ex),
                                                      &ts);
                        ibv_end_poll(ctx->cq_s.cq_ex);
                        if (ret && ret != ENOENT) {
                                fprintf(stderr, "poll CQ failed %d\n", ret);
                                return ret;
                        }
                } else {
                        int ne, i;
                        struct ibv_wc wc[2];

                        do {
                                ne = ibv_poll_cq(pp_cq(ctx), 2, wc);
                                if (ne < 0) {
                                        fprintf(stderr, "poll CQ failed %d\n", ne);
                                        return 1;
                                }
                        } while (!use_event && ne < 1);

                        for (i = 0; i < ne; ++i) {
                                ret = parse_single_wc(ctx, &scnt, &rcnt, &routs,
                                                      iters,
                                                      wc[i].wr_id,
						      wc[i].imm_data,
                                                      wc[i].status,
                                                      0, &ts);
                                if (ret) {
                                        fprintf(stderr, "parse WC failed %d\n", ne);
                                        return 1;
                                }
                        }
                }
        }

        if (gettimeofday(&end, NULL)) {
                perror("gettimeofday");
                return 1;
        }

        {
                float usec = (end.tv_sec - start.tv_sec) * 1000000 +
                        (end.tv_usec - start.tv_usec);
                long long bytes = (long long) size * iters * 2;

                printf("%lld bytes in %.2f seconds = %.2f Mbit/sec\n",
                       bytes, usec / 1000000., bytes * 8. / usec);
                printf("%d iters in %.2f seconds = %.2f usec/iter\n",
                       iters, usec / 1000000., usec / iters);

                if (use_ts && ts.comp_with_time_iters) {
                        printf("Max receive completion clock cycles = %" PRIu64 "\n",
                               ts.comp_recv_max_time_delta);
                        printf("Min receive completion clock cycles = %" PRIu64 "\n",
                               ts.comp_recv_min_time_delta);
                        printf("Average receive completion clock cycles = %f\n",
                               (double)ts.comp_recv_total_time_delta / ts.comp_with_time_iters);
                }

                if ((!servername) && (validate_buf)) {
                        if (use_dm)
                                if (ibv_memcpy_from_dm(ctx->buf, ctx->dm, 0, size)) {
                                        fprintf(stderr, "Copy from DM buffer failed\n");
                                        return 1;
                                }
                        for (int i = 0; i < size; i += page_size)
                                if (ctx->buf[i] != i / page_size % sizeof(char))
                                        printf("invalid data in page %d\n",
                                               i / page_size);
                }
        }

        ibv_ack_cq_events(pp_cq(ctx), num_cq_events);

        if (pp_close_ctx(ctx))
                return 1;

        ibv_free_device_list(dev_list);
        free(rem_dest);

        return 0;
}

