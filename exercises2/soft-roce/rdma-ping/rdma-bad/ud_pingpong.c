// Require information: Device name, Port GID Index, Server IP
#include <malloc.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common/bitset.h"
#include "common/common.h"
#include "common/net.h"
#include "common/persistence.h"
#include "src/args.h"
#include "src/ib_net.h"
#include "src/pingpong.h"

#define IB_MTU (pp_mtu_to_enum (PACKET_SIZE))

#define QUEUE_SIZE 128
#define DEFAULT_PORT 18515
#define IB_PORT 1

// Priority (i.e. service level) for the traffic
#define PRIORITY 0

persistence_agent_t *persistence_agent;

static volatile bool global_exit = false;

/**
 * Work Request IDs.
 * Range [0, QUEUE_SIZE) is used for send WRs, [QUEUE_SIZE, 2*QUEUE_SIZE) is used for receive WRs.
 * The different IDs in the reception are used to determine which queue index was used to receive the packet.
 * The IDs in the send are used only by the server, in order to remember which queue index is now available to receive a new packet, i.e. to be used in a new recv WR.
 */
enum {
    PINGPONG_SEND_WRID = 0,
    PINGPONG_RECV_WRID = QUEUE_SIZE,
};

static int page_size;

struct pingpong_context {
    /**
     * Bitset to keep track of the send WRs that are still pending.
     * On the client only the first bit is used, since there is only one buffer used to send (`send_buf`).
     * On the server, each bit represents a different queue index, keeping track of the buffers that have a pending send request.
     */
    BITSET_DECLARE (pending_send, QUEUE_SIZE);
    int send_flags;

    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_mr *send_mr;
    struct ibv_mr *recv_mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_ah *ah;

    struct ib_node_info remote_info;

    uint8_t *send_buf;
    struct pingpong_payload *send_payload;

    uint8_t *recv_bufs;
    struct pingpong_payload *recv_payloads[QUEUE_SIZE];
};

int init_pp_buffer (void **buffer, size_t size)
{
    *buffer = memalign (page_size, size);
    if (!*buffer)
    {
        fprintf (stderr, "Couldn't allocate work buffer\n");
        return 1;
    }

    memset (*buffer, 0, size);
    return 0;
}

struct pingpong_context *pp_init_context (struct ibv_device *ib_dev)
{
    struct pingpong_context *ctx = malloc (sizeof (struct pingpong_context));
    if (!ctx)
        return NULL;

    BITSET_INIT (ctx->pending_send, QUEUE_SIZE);

    ctx->send_flags = IBV_SEND_SIGNALED;

    if (init_pp_buffer ((void **) &ctx->send_buf, PACKET_SIZE))
    {
        LOG (stderr, "Couldn't allocate send_buf\n");
        goto clean_ctx;
    }
    ctx->send_payload = (struct pingpong_payload *) (ctx->send_buf + 40);

    if (init_pp_buffer ((void **) &ctx->recv_bufs, PACKET_SIZE * QUEUE_SIZE))
    {
        LOG (stderr, "Couldn't allocate recv_buf\n");
        goto clean_send_buf;
    }

    for (unsigned i = 0; i < QUEUE_SIZE; ++i)
    {
        ctx->recv_payloads[i] = (struct pingpong_payload *) (ctx->recv_bufs + i * PACKET_SIZE + 40);
    }

    ctx->context = ibv_open_device (ib_dev);
    if (!ctx->context)
    {
        LOG (stderr, "Couldn't get context for %s\n", ibv_get_device_name (ib_dev));
        goto clean_recv_buf;
    }

    ctx->pd = ibv_alloc_pd (ctx->context);
    if (!ctx->pd)
    {
        LOG (stderr, "Couldn't allocate PD\n");
        goto clean_context;
    }

    ctx->send_mr = ibv_reg_mr (ctx->pd, ctx->send_buf, PACKET_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->send_mr)
    {
        LOG (stderr, "Couldn't register MR for send_buf\n");
        goto clean_pd;
    }
    ctx->recv_mr = ibv_reg_mr (ctx->pd, ctx->recv_bufs, QUEUE_SIZE * PACKET_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!ctx->recv_mr)
    {
        LOG (stderr, "Couldn't register MR for recv_buf\n");
        goto clean_send_mr;
    }

    ctx->cq = ibv_create_cq (ctx->context, QUEUE_SIZE, NULL, NULL, 0);
    if (!ctx->cq)
    {
        LOG (stderr, "Couldn't create CQ\n");
        goto clean_recv_mr;
    }

    {
        struct ibv_qp_attr attr;
        struct ibv_qp_init_attr init_attr = {
            .send_cq = ctx->cq,
            .recv_cq = ctx->cq,
            .cap = {
                // Since RECV requests are posted for each queue index when the corresponding send WR is completed, there could be at most `QUEUE_SIZE` SEND WRs in flight.
                .max_send_wr = QUEUE_SIZE,
                .max_recv_wr = QUEUE_SIZE,
                .max_send_sge = 1,
                .max_recv_sge = 1},
            .qp_type = IBV_QPT_UD,
        };

        ctx->qp = ibv_create_qp (ctx->pd, &init_attr);
        if (!ctx->qp)
        {
            LOG (stderr, "Couldn't create QP\n");
            goto clean_cq;
        }

        ibv_query_qp (ctx->qp, &attr, IBV_QP_CAP, &init_attr);
        if (init_attr.cap.max_inline_data >= PACKET_SIZE)
        {
            ctx->send_flags |= IBV_SEND_INLINE;
        }
        else
        {
            LOG (stdout, "Device doesn't support IBV_SEND_INLINE, using sge. Max inline size: %d\n", init_attr.cap.max_inline_data);
        }
    }

    {
        struct ibv_qp_attr attr = {
            .qp_state = IBV_QPS_INIT,
            .pkey_index = 0,
            .port_num = IB_PORT,
            .qkey = 0x11111111};
        if (ibv_modify_qp (ctx->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY))
        {
            LOG (stderr, "Failed to modify QP to INIT\n");
            goto clean_qp;
        }
    }

    return ctx;

clean_qp:
    ibv_destroy_qp (ctx->qp);

clean_cq:
    ibv_destroy_cq (ctx->cq);

clean_recv_mr:
    ibv_dereg_mr (ctx->recv_mr);

clean_send_mr:
    ibv_dereg_mr (ctx->send_mr);

clean_pd:
    ibv_dealloc_pd (ctx->pd);

clean_context:
    ibv_close_device (ctx->context);

clean_recv_buf:
    free (ctx->recv_bufs);

clean_send_buf:
    free (ctx->send_buf);

clean_ctx:
    free (ctx);
    return NULL;
}

int pp_close_context (struct pingpong_context *ctx)
{
    if (ibv_destroy_qp (ctx->qp))
    {
        LOG (stderr, "Couldn't destroy QP\n");
        return 1;
    }

    if (ibv_destroy_cq (ctx->cq))
    {
        LOG (stderr, "Couldn't destroy CQ\n");
        return 1;
    }

    if (ibv_dereg_mr (ctx->recv_mr))
    {
        LOG (stderr, "Couldn't deregister MR for recv_buf\n");
        return 1;
    }

    if (ibv_dereg_mr (ctx->send_mr))
    {
        LOG (stderr, "Couldn't deregister MR for send_buf\n");
        return 1;
    }

    if (ctx->ah && ibv_destroy_ah (ctx->ah))
    {
        LOG (stderr, "Couldn't destroy AH\n");
        return 1;
    }

    if (ibv_dealloc_pd (ctx->pd))
    {
        LOG (stderr, "Couldn't deallocate PD\n");
        return 1;
    }

    if (ibv_close_device (ctx->context))
    {
        LOG (stderr, "Couldn't close device context\n");
        return 1;
    }

    free (ctx->recv_bufs);
    free (ctx->send_buf);
    free (ctx);
    return 0;
}

int pp_post_recv (struct pingpong_context *ctx, int queue_idx)
{
    struct ibv_sge list = {
        .addr = (uintptr_t) ctx->recv_bufs + queue_idx * PACKET_SIZE,
        .length = PACKET_SIZE,
        .lkey = ctx->recv_mr->lkey};
    struct ibv_recv_wr wr = {
        .wr_id = PINGPONG_RECV_WRID + queue_idx,
        .sg_list = &list,
        .num_sge = 1};
    struct ibv_recv_wr *bad_wr;

    if (ibv_post_recv (ctx->qp, &wr, &bad_wr))
    {
        LOG (stderr, "Couldn't post receive for queue_idx %d\n", queue_idx);
        return 1;
    }

    return 0;
}

/**
 * Post a send WR to the queue.
 * This function can be called only when the `buffer` is not being used by a pending WR.
 *
 * @param ctx the pingpong context
 * @param buffer the buffer containing the packet to send
 * @param lkey the local key of the buffer, obtained from the MR
 * @param queue_idx if the packet being sent is the response to a received packet, this is the queue index used to receive the packet. Otherwise, -1.
 * @return 0 on success, -1 on failure
 */
int pp_post_send (struct pingpong_context *ctx, uintptr_t buffer, uint32_t lkey, int queue_idx)
{
    const struct ib_node_info *remote = &ctx->remote_info;
    struct ibv_sge list = {
        .addr = buffer + 40,
        .length = PACKET_SIZE - 40,
        .lkey = lkey};

    struct ibv_send_wr wr = {
        .wr_id = queue_idx == -1 ? PINGPONG_SEND_WRID : PINGPONG_SEND_WRID + queue_idx,
        .sg_list = &list,
        .num_sge = 1,
        .opcode = IBV_WR_SEND,
        .send_flags = ctx->send_flags,
        .wr = {
            .ud = {
                .ah = ctx->ah,
                .remote_qpn = remote->qpn,
                .remote_qkey = 0x11111111}}};
    struct ibv_send_wr *bad_wr;
    BITSET_SET (ctx->pending_send, queue_idx != -1 ? queue_idx : 0);
    return ibv_post_send (ctx->qp, &wr, &bad_wr);
}

int parse_single_wc (struct pingpong_context *ctx, struct ibv_wc wc)
{
    const uint64_t ts = get_time_ns ();
    if (wc.status != IBV_WC_SUCCESS)
    {
        LOG (stderr, "Failed status %s (%d) for wr_id %d\n", ibv_wc_status_str (wc.status), wc.status, (int) wc.wr_id);
        return 1;
    }

    // Unknown WRID
    if (wc.wr_id >= PINGPONG_RECV_WRID + QUEUE_SIZE)
    {
        LOG (stderr, "Completion for unknown wr_id %d\n", (int) wc.wr_id);
        return 1;
    }

    // Send WRID
    if (wc.wr_id < PINGPONG_SEND_WRID + QUEUE_SIZE)
    {
        // Received a completion for a send WR, the corresponding bit can be unset.
        BITSET_CLEAR (ctx->pending_send, wc.wr_id - PINGPONG_SEND_WRID);

        // Only the server needs to wait for the completion of a send operation in order to post the receive.
        // Since the client reads the content immediately and only once, the recv is posted immediatly because overwriting the buffer is not a problem.
#if SERVER
        if (UNLIKELY (pp_post_recv (ctx, wc.wr_id - PINGPONG_SEND_WRID)))
        {
            LOG (stderr, "Couldn't post receive on queue_idx %lu\n", wc.wr_id - PINGPONG_SEND_WRID);
            return 1;
        }
#endif
        return 0;
    }

    // Recv WRID
    uint32_t queue_idx = wc.wr_id - PINGPONG_RECV_WRID;
    // LOG (stdout, "Received packet on queue_idx %d\n", queue_idx);
#if SERVER
    // Make sure the send buffer is not being used by an outgoing packet.
    BUSY_WAIT (BITSET_TEST (ctx->pending_send, queue_idx));
    ctx->recv_payloads[queue_idx]->ts[1] = ts;
    ctx->recv_payloads[queue_idx]->ts[2] = get_time_ns ();
    LOG (stdout, "Sending back packet %llu from queue %d\n", ctx->recv_payloads[queue_idx]->id, queue_idx);
    if (pp_post_send (ctx, (uintptr_t) ctx->recv_bufs + queue_idx * PACKET_SIZE, ctx->recv_mr->lkey, queue_idx))
    {
        LOG (stderr, "Couldn't post send\n");
        return 1;
    }
#else
    ctx->recv_payloads[queue_idx]->ts[3] = get_time_ns ();
    persistence_agent->write (persistence_agent, ctx->recv_payloads[queue_idx]);

    // The client can post the receive for the used queue index immediately, since the packet is not used anymore.
    if (UNLIKELY (pp_post_recv (ctx, queue_idx)))
    {
        LOG (stderr, "Couldn't post receive on queue_idx %d\n", queue_idx);
        return 1;
    }
#endif

    return 0;
}

int pp_ib_connect (struct pingpong_context *ctx, int gidx, struct ib_node_info *local, struct ib_node_info *dest)
{
    struct ibv_ah_attr ah_attr = {
        .is_global = 0,
        .dlid = dest->lid,
        .sl = PRIORITY,
        .src_path_bits = 0,
        .port_num = IB_PORT};
    struct ibv_qp_attr attr = {.qp_state = IBV_QPS_RTR};

    if (ibv_modify_qp (ctx->qp, &attr, IBV_QP_STATE))
    {
        LOG (stderr, "Failed to modify QP to RTR\n");
        return 1;
    }

    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = local->psn;

    if (ibv_modify_qp (ctx->qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN))
    {
        LOG (stderr, "Failed to modify QP to RTS\n");
        return 1;
    }

    if (dest->gid.global.interface_id)
    {
        ah_attr.is_global = 1;
        ah_attr.grh.dgid = dest->gid;
        ah_attr.grh.sgid_index = gidx;
        ah_attr.grh.hop_limit = 1;
    }

    ctx->ah = ibv_create_ah (ctx->pd, &ah_attr);
    if (!ctx->ah)
    {
        LOG (stderr, "Failed to create AH\n");
        return 1;
    }

    return 0;
}

int pp_send_single_packet (char *buf __unused, const uint64_t packet_id, struct sockaddr_ll *dest_addr __unused, void *aux)
{
    struct pingpong_context *ctx = (struct pingpong_context *) aux;

    // Make sure the buffer is available before writing on it.
    BUSY_WAIT (BITSET_TEST (ctx->pending_send, 0));

    *ctx->send_payload = new_pingpong_payload (packet_id);
    ctx->send_payload->ts[0] = get_time_ns ();

    return pp_post_send (ctx, (uintptr_t) ctx->send_buf, ctx->send_mr->lkey, -1);
}

void sigint_handler (int sig __unused)
{
    global_exit = true;
}

int main (int argc, char **argv)
{
    char *ib_devname = NULL;
    int port_gid_idx = 0;
    uint64_t iters = 0;

    char *server_ip = NULL;
#if SERVER
    if (!ib_parse_args (argc, argv, &ib_devname, &port_gid_idx, &iters))
    {
        ib_print_usage (argv[0]);
        return 1;
    }
#else
    uint64_t interval = 0;
    uint32_t persistence_flags = 0;
    if (!ib_parse_args (argc, argv, &ib_devname, &port_gid_idx, &iters, &interval, &server_ip, &persistence_flags))
    {
        ib_print_usage (argv[0]);
        return 1;
    }

    persistence_agent = persistence_init ("ud.dat", persistence_flags, &interval);
    if (!persistence_agent)
    {
        LOG (stderr, "Failed to initialize persistence agent\n");
        return 1;
    }
#endif

    srand48 (getpid () * time (NULL));

    page_size = sysconf (_SC_PAGESIZE);

    struct ibv_device *ib_dev = ib_device_find_by_name (ib_devname);
    if (!ib_dev)
    {
        fprintf (stderr, "IB device %s not found\n", ib_devname);
        return 1;
    }

    struct pingpong_context *ctx = pp_init_context (ib_dev);
    if (!ctx)
    {
        fprintf (stderr, "Couldn't initialize context\n");
        return 1;
    }

    struct ib_node_info local_info;
    if (ib_get_local_info (ctx->context, IB_PORT, port_gid_idx, ctx->qp, &local_info))
    {
        fprintf (stderr, "Couldn't get local info\n");
        pp_close_context (ctx);
        return 1;
    }
    ib_print_node_info (&local_info);

    if (exchange_data (server_ip, SERVER, sizeof (local_info), (uint8_t *) &local_info, (uint8_t *) &ctx->remote_info))
    {
        fprintf (stderr, "Couldn't exchange data\n");
        pp_close_context (ctx);
        return 1;
    }
    ib_print_node_info (&ctx->remote_info);

    if (pp_ib_connect (ctx, port_gid_idx, &local_info, &ctx->remote_info))
    {
        fprintf (stderr, "Couldn't connect\n");
        pp_close_context (ctx);
        return 1;
    }

    LOG (stdout, "Connected\n");

    for (int i = 0; i < QUEUE_SIZE; ++i)
    {
        if (pp_post_recv (ctx, i))
        {
            fprintf (stderr, "Couldn't post receive\n");
            pp_close_context (ctx);
            return 1;
        }
    }

    signal (SIGINT, sigint_handler);

#if !SERVER
    start_sending_packets (iters, interval, (char *) ctx->send_buf, NULL, pp_send_single_packet, ctx);
#endif

    uint64_t recv_idx = 0;
    while (LIKELY (recv_idx < iters && !global_exit))
    {
        struct ibv_wc wc[2];
        int ne;

        do
        {
            ne = ibv_poll_cq (ctx->cq, 2, wc);
            if (ne < 0)
            {
                fprintf (stderr, "Poll CQ failed %d\n", ne);
                return 1;
            }
        } while (!ne && !global_exit);

        if (UNLIKELY (global_exit))
            break;

        for (int i = 0; i < ne; ++i)
        {
            if (UNLIKELY (parse_single_wc (ctx, wc[i])))
            {
                fprintf (stderr, "Couldn't parse WC\n");
                goto done;
            }

            if (wc[i].wr_id >= PINGPONG_RECV_WRID)
                recv_idx = max (recv_idx, ctx->recv_payloads[wc[i].wr_id - PINGPONG_RECV_WRID]->id);
        }
    }
done:
    LOG (stdout, "Received all packets\n");

    if (persistence_agent)
    {
        persistence_agent->close (persistence_agent);
    }
    if (pp_close_context (ctx))
    {
        fprintf (stderr, "Couldn't close context\n");
        return 1;
    }
    return 0;
}
