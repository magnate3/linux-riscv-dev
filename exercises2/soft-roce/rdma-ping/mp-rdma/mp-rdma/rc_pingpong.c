/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <netdb.h>
#include <stdlib.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <time.h>
#include <fcntl.h>

#include "pingpong.h"

#define MP_REBALANCE_SEND_FRACTION (1000)
#define MP_REBALANCE_INTERVAL      (1e6)
#define MAX_MP_CONN                (8)
#define MIN_ROUTS_THRESHOLD        (8)
#define MAX_RATIO                  (100)

enum {
	PINGPONG_RECV_WRID = 0,
	PINGPONG_SEND_WRID = 1,
	PINGPONG_MAX_WRID  = 2
};

static int page_size;


struct pingpong_ecn {
	int _rp_cnp_handled;
	int _rp_cnp_ignored;
	int _np_cnp_sent;
	int _np_ecn_marked_roce_packets;
};

struct pingpong_context {
	struct ibv_context	*context;
	struct ibv_comp_channel *channel;
	struct ibv_pd		*pd;
	struct ibv_mr		*mr;
	struct ibv_cq		*cq;
	struct ibv_qp		*qp[MAX_MP_CONN];
	struct pingpong_ecn	ecn_fds;
	void			*buf;
	int			 size;
	int			 rx_depth;
	int			 pending[MAX_MP_CONN];
	unsigned	 ratio[MAX_MP_CONN];
	struct ibv_port_attr     portinfo;
	int                      random_seed;
};

#define ECN_COUNTER_PATH ("/sys/class/infiniband/%s/ports/%i/hw_counters/%s")
#define MAX_ECN_PATH_LEN (100)
#define MAX_ECN_VALUE_LEN (10)
#define ECN_COUNTER(ecn, counter)                                      \
	(*((int*)((char*)(ecn) +                                           \
		offsetof(struct pingpong_ecn, _ ## counter))))

#define ECN_OPEN_ONE(ecn_fds, dev_name, port_num, counter)             \
{                                                                      \
	char path[MAX_ECN_PATH_LEN];                                       \
	snprintf(path, MAX_ECN_PATH_LEN, ECN_COUNTER_PATH,                 \
		dev_name, port_num, # counter );                               \
	int fd = open(path, O_RDONLY);                                     \
	if (fd < 0) {                                                      \
		fprintf(stderr, "Couldn't open the ECN counter files\n");      \
		return 1;                                                      \
	}                                                                  \
	ECN_COUNTER(ecn_fds, counter) = fd;                                \
}

#define ECN_OPEN(ecn_fds, dev_name, port_num)                          \
{                                                                      \
	ECN_OPEN_ONE(ecn_fds, dev_name, port_num, rp_cnp_handled);         \
	ECN_OPEN_ONE(ecn_fds, dev_name, port_num, rp_cnp_ignored);         \
	ECN_OPEN_ONE(ecn_fds, dev_name, port_num, np_cnp_sent);            \
	ECN_OPEN_ONE(ecn_fds, dev_name, port_num,                          \
		np_ecn_marked_roce_packets);                                   \
}

#define ECN_CLOSE_ONE(ecn_fds, counter)                                \
	close(ECN_COUNTER(ecn_fds, counter))

#define ECN_CLOSE(ecn_fds)                                             \
{                                                                      \
	ECN_CLOSE_ONE(ecn_fds, rp_cnp_handled);                            \
	ECN_CLOSE_ONE(ecn_fds, rp_cnp_ignored);                            \
	ECN_CLOSE_ONE(ecn_fds, np_cnp_sent);                               \
	ECN_CLOSE_ONE(ecn_fds, np_ecn_marked_roce_packets);                \
}

#define ECN_READ_ONE(ecn_fds, out, counter)                            \
{                                                                      \
	char read_buf[MAX_ECN_VALUE_LEN];                                  \
	lseek(ECN_COUNTER(ecn_fds, counter), 0, SEEK_SET);                 \
	int ret = read(ECN_COUNTER(ecn_fds, counter),                      \
		&read_buf, MAX_ECN_VALUE_LEN);                                 \
	if (ret < 0) {                                                     \
		return ret;                                                    \
	}                                                                  \
	read_buf[ret] = 0;                                                 \
	ECN_COUNTER(out, counter) = atoi(read_buf);                        \
}

#define ECN_READ(ecn_fds, out)                                         \
{                                                                      \
	ECN_READ_ONE(ecn_fds, out, rp_cnp_handled);                        \
	ECN_READ_ONE(ecn_fds, out, rp_cnp_ignored);                        \
	ECN_READ_ONE(ecn_fds, out, np_cnp_sent);                           \
	ECN_READ_ONE(ecn_fds, out, np_ecn_marked_roce_packets);            \
}

#define ECN_SUBTRACT_ONE(ecn_a, ecn_b, ecn_c, counter)                 \
{                                                                      \
	ECN_COUNTER(ecn_a, counter) = ECN_COUNTER(ecn_b, counter) -        \
		ECN_COUNTER(ecn_c, counter);                                   \
}

#define ECN_SUBTRACT(ecn_a, ecn_b, ecn_c)                              \
{                                                                      \
	ECN_SUBTRACT_ONE(ecn_a, ecn_b, ecn_c, rp_cnp_handled);             \
	ECN_SUBTRACT_ONE(ecn_a, ecn_b, ecn_c, rp_cnp_ignored);             \
	ECN_SUBTRACT_ONE(ecn_a, ecn_b, ecn_c, np_cnp_sent);                \
	ECN_SUBTRACT_ONE(ecn_a, ecn_b, ecn_c, np_ecn_marked_roce_packets); \
}

// TODO: "calibrate" the difference - to know when ECN read "improves"
#define ECN_DIFF(ecn_a, ecn_b)                                         \
	((ECN_COUNTER(ecn_a, np_cnp_sent) +                                \
		ECN_COUNTER(ecn_a, np_ecn_marked_roce_packets)) >              \
	 (ECN_COUNTER(ecn_b, np_cnp_sent) +                                \
		ECN_COUNTER(ecn_b, np_ecn_marked_roce_packets)))

#define ECN_PRINT_ONE(out, counter)                                    \
	printf(#counter "=%i\n", (ECN_COUNTER(out, counter)))

#define ECN_PRINT(ecn_fds)                                             \
{                                                                      \
	struct pingpong_ecn temp;          	                               \
	ECN_READ(ecn_fds, &temp);                                          \
	ECN_PRINT_ONE(&temp, rp_cnp_handled);                              \
	ECN_PRINT_ONE(&temp, rp_cnp_ignored);                              \
	ECN_PRINT_ONE(&temp, np_cnp_sent);                                 \
	ECN_PRINT_ONE(&temp, np_ecn_marked_roce_packets);                  \
}

struct pingpong_dest {
	int lid;
	int psn;
	union ibv_gid gid;
	int qpn[MAX_MP_CONN];
};

static int pp_connect_ctx(struct pingpong_context *ctx, int port, int my_psn,
			  enum ibv_mtu mtu, int sl,
			  struct pingpong_dest *dest, int sgid_idx, int qp_idx)
{
	struct ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR,
		.path_mtu		= mtu,
		.dest_qp_num		= dest->qpn[qp_idx],
		.rq_psn			= dest->psn,
		.max_dest_rd_atomic	= 1,
		.min_rnr_timer		= 12,
		.ah_attr		= {
			.is_global	= 0,
			.dlid		= dest->lid,
			.sl		= sl,
			.src_path_bits	= 0,
			.port_num	= port
		}
	};

	if (dest->gid.global.interface_id) {
		attr.ah_attr.is_global = 1;
		attr.ah_attr.grh.hop_limit = 1;
		attr.ah_attr.grh.dgid = dest->gid;
		attr.ah_attr.grh.sgid_index = sgid_idx;
	}
	if (ibv_modify_qp(ctx->qp[qp_idx], &attr,
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

	attr.qp_state	    = IBV_QPS_RTS;
	attr.timeout	    = 14;
	attr.retry_cnt	    = 7;
	attr.rnr_retry	    = 7;
	attr.sq_psn	    = my_psn;
	attr.max_rd_atomic  = 1;
	if (ibv_modify_qp(ctx->qp[qp_idx], &attr,
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

static struct pingpong_dest *pp_client_exch_dest(const char *servername, int port,
						 const struct pingpong_dest *my_dest)
{
	struct addrinfo *res, *t;
	struct addrinfo hints = {
		.ai_family   = AF_INET,
		.ai_socktype = SOCK_STREAM
	};
	char *service;
	int n;
	int sockfd = -1;
	struct pingpong_dest *rem_dest = malloc(sizeof *rem_dest);
	if (!rem_dest)
		goto out;

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

	if (write(sockfd, my_dest, sizeof *my_dest) != sizeof *my_dest) {
		fprintf(stderr, "Couldn't send local address\n");
		goto out;
	}


	if (read(sockfd, rem_dest, sizeof *rem_dest) != sizeof *rem_dest) {
		perror("client read");
		fprintf(stderr, "Couldn't read remote address\n");
		goto out;
	}

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
		.ai_family   = AF_INET,
		.ai_socktype = SOCK_STREAM
	};
	char *service;
	int n;
	int sockfd = -1, connfd;
	struct pingpong_dest *rem_dest = malloc(sizeof *rem_dest);
	if (!rem_dest)
		goto out;

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
	connfd = accept(sockfd, NULL, 0);
	close(sockfd);
	if (connfd < 0) {
		fprintf(stderr, "accept() failed\n");
		return NULL;
	}

	n = read(connfd, rem_dest, sizeof *rem_dest);
	if (n != sizeof *rem_dest) {
		perror("server  read");
		fprintf(stderr, "%d/%d: Couldn't read remote address\n", n, (int) sizeof *rem_dest);
		goto out;
	}

	for(n = 0; n < MAX_MP_CONN && rem_dest->qpn[n]; n++) {
		if (pp_connect_ctx(ctx, ib_port, my_dest->psn, mtu, sl, rem_dest, sgid_idx, n)) {
			fprintf(stderr, "Couldn't connect to remote QP\n");
			free(rem_dest);
			rem_dest = NULL;
			goto out;
		}
	}

	if (write(connfd, my_dest, sizeof *my_dest) != sizeof *my_dest) {
		fprintf(stderr, "Couldn't send local address\n");
		free(rem_dest);
		rem_dest = NULL;
		goto out;
	}

out:
	close(connfd);
	return rem_dest;
}

#include <sys/param.h>

static struct pingpong_context *pp_init_ctx(struct ibv_device *ib_dev, int size,
					    int conns, int rx_depth, int port,
					    int use_event, int is_server)
{
	struct pingpong_context *ctx;
	int qp_idx;

	ctx = calloc(1, sizeof *ctx);
	if (!ctx)
		return NULL;

	ctx->size     = size;
	ctx->rx_depth = rx_depth;

	ctx->buf = malloc(roundup(size, page_size));
	if (!ctx->buf) {
		fprintf(stderr, "Couldn't allocate work buf.\n");
		return NULL;
	}

	memset(ctx->buf, 0x7b + is_server, size);

	ctx->context = ibv_open_device(ib_dev);
	if (!ctx->context) {
		fprintf(stderr, "Couldn't get context for %s\n",
			ibv_get_device_name(ib_dev));
		return NULL;
	}

	if (use_event) {
		ctx->channel = ibv_create_comp_channel(ctx->context);
		if (!ctx->channel) {
			fprintf(stderr, "Couldn't create completion channel\n");
			return NULL;
		}
	} else
		ctx->channel = NULL;

	ctx->pd = ibv_alloc_pd(ctx->context);
	if (!ctx->pd) {
		fprintf(stderr, "Couldn't allocate PD\n");
		return NULL;
	}

	ctx->mr = ibv_reg_mr(ctx->pd, ctx->buf, size, IBV_ACCESS_LOCAL_WRITE);
	if (!ctx->mr) {
		fprintf(stderr, "Couldn't register MR\n");
		return NULL;
	}

	ctx->cq = ibv_create_cq(ctx->context, rx_depth + 1, NULL,
				ctx->channel, 0);
	if (!ctx->cq) {
		fprintf(stderr, "Couldn't create CQ\n");
		return NULL;
	}

	for (qp_idx = 0; qp_idx < conns; qp_idx++) {
		struct ibv_qp_init_attr attr = {
			.send_cq = ctx->cq,
			.recv_cq = ctx->cq,
			.cap     = {
				.max_send_wr  = 1,
				.max_recv_wr  = rx_depth,
				.max_send_sge = 1,
				.max_recv_sge = 1
			},
			.qp_type = IBV_QPT_RC
		};

		ctx->qp[qp_idx] = ibv_create_qp(ctx->pd, &attr);
		if (!ctx->qp[qp_idx])  {
			fprintf(stderr, "Couldn't create QP\n");
			return NULL;
		}
	}

	for (qp_idx = 0; qp_idx < conns; qp_idx++) {
		struct ibv_qp_attr attr = {
			.qp_state        = IBV_QPS_INIT,
			.pkey_index      = 0,
			.port_num        = port,
			.qp_access_flags = 0
		};

		if (ibv_modify_qp(ctx->qp[qp_idx], &attr,
				  IBV_QP_STATE              |
				  IBV_QP_PKEY_INDEX         |
				  IBV_QP_PORT               |
				  IBV_QP_ACCESS_FLAGS)) {
			fprintf(stderr, "Failed to modify QP to INIT\n");
			return NULL;
		}
	}

	return ctx;
}

int pp_close_ctx(struct pingpong_context *ctx)
{
	int qp_idx;
	for (qp_idx = 0; ctx->qp[qp_idx]; qp_idx++) {
		if (ibv_destroy_qp(ctx->qp[qp_idx])) {
			fprintf(stderr, "Couldn't destroy QP\n");
			return 1;
		}
	}

	if (ibv_destroy_cq(ctx->cq)) {
		fprintf(stderr, "Couldn't destroy CQ\n");
		return 1;
	}

	if (ibv_dereg_mr(ctx->mr)) {
		fprintf(stderr, "Couldn't deregister MR\n");
		return 1;
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

static int pp_post_recv(struct pingpong_context *ctx, int n, int qp_idx)
{
	struct ibv_sge list = {
		.addr	= (uintptr_t) ctx->buf,
		.length = ctx->size,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_recv_wr wr = {
		.wr_id	    = PINGPONG_RECV_WRID,
		.sg_list    = &list,
		.num_sge    = 1,
	};
	struct ibv_recv_wr *bad_wr;
	int i;

	for (i = 0; i < n; ++i)
		if (ibv_post_recv(ctx->qp[qp_idx], &wr, &bad_wr))
			break;

	return i;
}

static int pp_post_send(struct pingpong_context *ctx, int n, int qp_idx)
{
	struct ibv_sge list = {
		.addr	= (uintptr_t) ctx->buf,
		.length = ctx->size,
		.lkey	= ctx->mr->lkey
	};
	struct ibv_send_wr wr = {
		.wr_id	    = PINGPONG_SEND_WRID,
		.sg_list    = &list,
		.num_sge    = 1,
		.opcode     = IBV_WR_SEND,
		.send_flags = IBV_SEND_SIGNALED,
	};
	struct ibv_send_wr *bad_wr;
	int i;

	for (i = 0; i < n; ++i)
		if (ibv_post_send(ctx->qp[qp_idx], &wr, &bad_wr))
			break;

	return i;
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
	printf("  -c, --conns=<conns>    number of concurrent connections (default 1)\n");
	printf("  -b, --bandwidth        measure the bandwidth instead of the latency\n");
	printf("  -q, --query-ecn        query ECN counter for each connection\n");
	printf("  -o, --random-seed      set the random seed for the path ratio rebalancer\n");
}

static inline uint64_t pp_x86_timestamp()
{
    uint32_t low, high;
    asm volatile ("rdtsc" : "=a" (low), "=d" (high));
    return ((uint64_t)high << 32) | (uint64_t)low;
}

static uint64_t last_timestamp = 0;
struct pingpong_ecn last_ecn_value;
struct pingpong_ecn last_ecn_change;
static unsigned last_incremented_index;
static int last_incremented_value;

static inline int pp_multipath_rebalance(struct pingpong_context *ctx, int num_connections)
{
	uint64_t now = pp_x86_timestamp();
	if (now - last_timestamp < MP_REBALANCE_INTERVAL) {
		last_timestamp = now;
		return 0;
	}

	struct pingpong_ecn ecn_value, ecn_change;
	ECN_READ(&ctx->ecn_fds, &ecn_value);
	ECN_SUBTRACT(&ecn_change, &ecn_value, &last_ecn_value);
	last_timestamp = now;

	/* Use ECN to decide how to send packets */
	int has_improved = ECN_DIFF(&last_ecn_change, &ecn_change);
	int tried_incrementing = (last_incremented_index != -1);
	if (tried_incrementing) {
		if (has_improved) {
			last_incremented_value *= 2; /* Note: can be negative! */
		} else {
			last_incremented_value = -1;
		}
	} else {
		last_incremented_index = rand_r(&ctx->random_seed);
		last_incremented_value = 1;
	}

	/* Increment by using the selected path index and value */
	ctx->ratio[last_incremented_index] += last_incremented_value;
	if (ctx->ratio[last_incremented_index] > MAX_RATIO) {
		/* Normalize the values - divide all by two */
		int i;
		for (i = 0; i < num_connections; i++) {
			ctx->ratio[i] >>= 1;
		}
	}

	/* If we reached the peak - try a new path to balance */
	if (last_incremented_value < 0) {
		last_incremented_index = -1;
	}

	/* store values for the next time we rebalance */
	memcpy(&last_ecn_change, &ecn_change, sizeof(ecn_change));
	memcpy(&last_ecn_value, &ecn_value, sizeof(ecn_change));
	return 0;
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
	int                      port = 18515;
	int                      ib_port = 1;
	int                      size = 4096;
	enum ibv_mtu             mtu = IBV_MTU_1024;
	int                      rx_depth = 500;
	int                      iters = 1000;
	int                      use_event = 0;
	int                      routs[MAX_MP_CONN];
	int                      souts[MAX_MP_CONN];
	int                      rcnt, scnt;
	int                      num_cq_events = 0;
	int                      sl = 0;
	int                      conns = 1;
	int                      gidx = -1;
	char                     gid[33];
	int                      qp_idx;
	int                      wr_id_qp;
	int                      wr_id_type;
	int                      use_bandwidth;
	int                      query_ecn;
	int                      random_seed;

	srand48(getpid() * time(NULL));

	while (1) {
		int c;

		static struct option long_options[] = {
			{ .name = "port",       .has_arg = 1, .val = 'p' },
			{ .name = "ib-dev",     .has_arg = 1, .val = 'd' },
			{ .name = "ib-port",    .has_arg = 1, .val = 'i' },
			{ .name = "size",       .has_arg = 1, .val = 's' },
			{ .name = "mtu",        .has_arg = 1, .val = 'm' },
			{ .name = "rx-depth",   .has_arg = 1, .val = 'r' },
			{ .name = "iters",      .has_arg = 1, .val = 'n' },
			{ .name = "sl",         .has_arg = 1, .val = 'l' },
			{ .name = "events",     .has_arg = 0, .val = 'e' },
			{ .name = "gid-idx",    .has_arg = 1, .val = 'g' },
			{ .name = "conns",      .has_arg = 1, .val = 'c' },
			{ .name = "bandwidth",  .has_arg = 0, .val = 'b' },
			{ .name = "query-ecn",  .has_arg = 0, .val = 'q' },
			{ .name = "random-seed",.has_arg = 1, .val = 'o' },
			{ 0 }
		};

		c = getopt_long(argc, argv, "p:d:i:s:m:r:n:l:eg:c:b", long_options, NULL);
		if (c == -1)
			break;

		switch (c) {
		case 'p':
			port = strtol(optarg, NULL, 0);
			if (port < 0 || port > 65535) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 'd':
			ib_devname = strdup(optarg);
			break;

		case 'i':
			ib_port = strtol(optarg, NULL, 0);
			if (ib_port < 0) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 's':
			size = strtol(optarg, NULL, 0);
			break;

		case 'm':
			mtu = pp_mtu_to_enum(strtol(optarg, NULL, 0));
			if (mtu < 0) {
				usage(argv[0]);
				return 1;
			}
			break;

		case 'r':
			rx_depth = strtol(optarg, NULL, 0);
			break;

		case 'n':
			iters = strtol(optarg, NULL, 0);
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

		case 'c':
			conns = strtol(optarg, NULL, 0);
			break;

		case 'b':
			++use_bandwidth;
			break;

		case 'q':
			++query_ecn;
			break;

		case 'o':
			random_seed = strtol(optarg, NULL, 0);
			break;

		default:
			usage(argv[0]);
			return 1;
		}
	}

	if (optind == argc - 1)
		servername = strdup(argv[optind]);
	else if (optind < argc) {
		usage(argv[0]);
		return 1;
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

	ctx = pp_init_ctx(ib_dev, size, conns, rx_depth, ib_port, use_event, !servername);
	if (!ctx)
		return 1;

	if (query_ecn) {
		ECN_OPEN(&ctx->ecn_fds, ib_dev->dev_name, port);
		ctx->random_seed = random_seed;
	}

	for (qp_idx = 0; qp_idx < conns; qp_idx++) {
		routs[qp_idx] = pp_post_recv(ctx, ctx->rx_depth, qp_idx);
		if (routs[qp_idx] < ctx->rx_depth) {
			fprintf(stderr, "Couldn't post receive (%d)\n", routs[qp_idx]);
			return 1;
		}
	}

	if (use_event)
		if (ibv_req_notify_cq(ctx->cq, 0)) {
			fprintf(stderr, "Couldn't request CQ notification\n");
			return 1;
		}


	if (pp_get_port_info(ctx->context, ib_port, &ctx->portinfo)) {
		fprintf(stderr, "Couldn't get port info\n");
		return 1;
	}

	my_dest.lid = ctx->portinfo.lid;
	if (ctx->portinfo.link_layer == IBV_LINK_LAYER_INFINIBAND && !my_dest.lid) {
		fprintf(stderr, "Couldn't get local LID\n");
		return 1;
	}

	if (gidx >= 0) {
		if (ibv_query_gid(ctx->context, ib_port, gidx, &my_dest.gid)) {
			fprintf(stderr, "Could not get local gid for gid index %d\n", gidx);
			return 1;
		}
	} else
		memset(&my_dest.gid, 0, sizeof my_dest.gid);

	for (qp_idx = 0; qp_idx < conns; qp_idx++) {
		my_dest.qpn[qp_idx] = ctx->qp[qp_idx]->qp_num;
	}
	my_dest.psn = lrand48() & 0xffffff;
	inet_ntop(AF_INET6, &my_dest.gid, gid, sizeof gid);
	printf("  local address:  LID 0x%04x, QPN0 0x%06x, PSN 0x%06x, GID %s\n",
	       my_dest.lid, my_dest.qpn[0], my_dest.psn, gid);


	if (servername)
		rem_dest = pp_client_exch_dest(servername, port, &my_dest);
	else
		rem_dest = pp_server_exch_dest(ctx, ib_port, mtu, port, sl, &my_dest, gidx);

	if (!rem_dest)
		return 1;

	inet_ntop(AF_INET6, &rem_dest->gid, gid, sizeof gid);
	printf("  remote address: LID 0x%04x, QPN0 0x%06x, PSN 0x%06x, GID %s\n",
	       rem_dest->lid, rem_dest->qpn[0], rem_dest->psn, gid);

	if (servername)
		for (qp_idx = 0; qp_idx < conns; qp_idx++) {
			if (pp_connect_ctx(ctx, ib_port, my_dest.psn, mtu, sl, rem_dest, gidx, qp_idx))
				return 1;
		}

	for (qp_idx = 0; qp_idx < conns; qp_idx++) {
		ctx->pending[qp_idx] = use_bandwidth ? 0 : (1 << PINGPONG_RECV_WRID);
	}

	if (servername) {
		scnt = use_bandwidth ? ctx->rx_depth : 1;
		for (qp_idx = 0; qp_idx < conns; qp_idx++) {
			souts[qp_idx] = pp_post_send(ctx, scnt, qp_idx);
			if (souts[qp_idx] < scnt) {
				fprintf(stderr, "Couldn't post send (%d)\n", souts[qp_idx]);
				return 1;
			}
			ctx->pending[qp_idx] |= 1 << PINGPONG_SEND_WRID;
		}
	}

	if (query_ecn) {
		printf("ECN counter - before starting:\n");
		ECN_PRINT(&ctx->ecn_fds);
	}

	if (gettimeofday(&start, NULL)) {
		perror("gettimeofday");
		return 1;
	}

	rcnt = scnt = 0;
	while (rcnt < iters || scnt < iters) {
		if (use_event) {
			struct ibv_cq *ev_cq;
			void          *ev_ctx;

			if (ibv_get_cq_event(ctx->channel, &ev_cq, &ev_ctx)) {
				fprintf(stderr, "Failed to get cq_event\n");
				return 1;
			}

			++num_cq_events;

			if (ev_cq != ctx->cq) {
				fprintf(stderr, "CQ event for unknown CQ %p\n", ev_cq);
				return 1;
			}

			if (ibv_req_notify_cq(ctx->cq, 0)) {
				fprintf(stderr, "Couldn't request CQ notification\n");
				return 1;
			}
		}

		{
			struct ibv_wc wc[2];
			int ne, i;

			do {
				ne = ibv_poll_cq(ctx->cq, 2, wc);
				if (ne < 0) {
					fprintf(stderr, "poll CQ failed %d\n", ne);
					return 1;
				}

			} while (!use_event && ne < 1);

			for (i = 0; i < ne; ++i) {
				if (wc[i].status != IBV_WC_SUCCESS) {
					fprintf(stderr, "Failed status %s (%d) for wr_id %d\n",
						ibv_wc_status_str(wc[i].status),
						wc[i].status, (int) wc[i].wr_id);
					return 1;
				}

				wr_id_type = (int) wc[i].wr_id % PINGPONG_MAX_WRID;
				wr_id_qp   = (int) wc[i].wr_id / PINGPONG_MAX_WRID;
				switch (wr_id_type) {
				case PINGPONG_SEND_WRID:
					if (use_bandwidth && --souts[wr_id_qp] <= MIN_ROUTS_THRESHOLD) {
						souts[wr_id_qp] += pp_post_send(ctx, ctx->rx_depth - souts[wr_id_qp], wr_id_qp);
						if (souts[wr_id_qp] < ctx->rx_depth) {
							fprintf(stderr,
									"Couldn't post send (%d)\n",
									routs[wr_id_qp]);
							return 1;
						}
					}

					++scnt;
					if (scnt % MP_REBALANCE_SEND_FRACTION == 0) {
						if (pp_multipath_rebalance(ctx, conns)) {
							return 1;
						}
					}
					break;

				case PINGPONG_RECV_WRID:
					if (--routs[wr_id_qp] <= MIN_ROUTS_THRESHOLD) {
						routs[wr_id_qp] += pp_post_recv(ctx, ctx->rx_depth - routs[wr_id_qp], wr_id_qp);
						if (routs[wr_id_qp] < ctx->rx_depth) {
							fprintf(stderr,
								"Couldn't post receive (%d)\n",
								routs[wr_id_qp]);
							return 1;
						}
					}

					++rcnt;
					break;

				default:
					fprintf(stderr, "Completion for unknown wr_id %d\n",
						(int) wc[i].wr_id);
					return 1;
				}

				ctx->pending[wr_id_qp] &= ~(1<<wr_id_type);
				if (scnt < iters && !ctx->pending) {
					if (pp_post_send(ctx, 1, wr_id_qp)) {
						fprintf(stderr, "Couldn't post send\n");
						return 1;
					}
					ctx->pending[wr_id_qp] = use_bandwidth ?
							(1 << PINGPONG_SEND_WRID) :
							((1 << PINGPONG_RECV_WRID) |
							 (1 << PINGPONG_SEND_WRID));
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

		if (query_ecn) {
			printf("ECN counter - after finishing:\n");
			ECN_PRINT(&ctx->ecn_fds);
		}
	}

	ibv_ack_cq_events(ctx->cq, num_cq_events);

	if (query_ecn) {
		ECN_CLOSE(&ctx->ecn_fds);
	}
	if (pp_close_ctx(ctx))
		return 1;

	ibv_free_device_list(dev_list);
	free(rem_dest);

	return 0;
}
