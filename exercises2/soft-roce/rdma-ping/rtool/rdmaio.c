/*
 *  rdmaio -- rdma io generation tool
 *
 *  Copyright (c) 2017, Mellanox Technologies. All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VERSION
#define VERSION "0.1 alpha"
#endif

#define _GNU_SOURCE
#include <endian.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <limits.h>
#include <unistd.h>
#include <semaphore.h>
#include <hugetlbfs.h>
#include <sys/time.h>
#include <malloc.h>
#include <inttypes.h>
#include <infiniband/verbs.h>
#include <infiniband/sa.h>
#include <rdma/rdma_cma.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/resource.h>
#include <sys/capability.h>
#include<signal.h>
#include<unistd.h>

#include "options.h"
#include "ts.h"

#define RDMAIO_Q_DEPTH 1024

enum rdmaio_msg_opc {
	RDMA_IO_OPC_REG_MR_REQ = 0x1,
	RDMA_IO_OPC_REG_MR_RSP = 0x81
};

enum rdmaio_cmd_rsp_type {
	RDMAIO_CMD = 1,
	RDMAIO_RSP = 2,
};

struct rdmaio_alloc_mr_cmd {
	uint32_t mr_len;
	uint32_t rsvd;
};

struct rdmaio_alloc_mr_rsp {
	uint32_t mr_len;
	uint32_t mr_key;
	uint64_t addr;
};

struct rdmaio_msg {
	uint8_t cmd_or_rsp;	/* enum rdmaio_cmd_rsp_type */
	uint8_t opc;		/* enum rdmaio_msg_opc */
	uint16_t rsvd2;
	uint32_t rsvd3;
	uint32_t io_handle;
	union {
		struct rdmaio_alloc_mr_cmd alloc_mr_cmd;
		struct rdmaio_alloc_mr_rsp alloc_mr_rsp;
		uint8_t cmd[56];
	} u;
};

struct rdmaio_rx_wr {
	struct rdmaio_msg *cmd;
	struct ibv_recv_wr wr;
	struct ibv_sge sge;
};

struct rdmaio_tx_wr {
	struct rdmaio_msg *cmd;
	struct ibv_send_wr rdma_wr;
	struct ibv_send_wr send_wr;
	struct ibv_sge sge;
};

enum rdmaio_cm_state {
	RDMAIO_CM_STATE_ADDR_RESOLVED,
	RDMAIO_CM_STATE_ROUTE_RESOLVED,
	RDMAIO_CM_CONNECTED,
	RDMAIO_CM_DISCONNECTED,
	RDMAIO_CM_FATAL_ERROR,
};

struct run_ctx;
struct rdma_connection;

enum rdmaio_si_state {
	RDMAIO_CLIENT_STATE_START_IO,
	RDMAIO_CLIENT_STATE_MR_REQ_SENT,
	RDMAIO_CLIENT_STATE_MR_RSP_RECV,
	RDMAIO_CLIENT_STATE_SI_SENT,
	RDMAIO_CLIENT_STATE_RW_SENT = RDMAIO_CLIENT_STATE_SI_SENT
};

struct rdma_rw_ctx {
	struct ibv_mr *local_mr;
	uint64_t length;	/* length of the IO can be lot smaller than
				 * the MR length.
				 */
	/* address and rkey advertized by server */
	uint64_t raddr;
	uint32_t rkey;
};

struct rdma_connection {
	struct ibv_comp_channel *cq_channel;
	struct ibv_cq *cq;
	struct ibv_pd *pd;
	long msg_count;

	struct {
		struct rdmaio_msg *cmds;
		struct rdmaio_rx_wr *wrs;
		struct ibv_mr *cmds_mr;
		uint64_t rqe_post_cnt;
		struct ibv_mr *io_mr;
	} rx;
	struct {
		struct rdmaio_msg *cmds;
		struct rdmaio_tx_wr *wrs;
		struct ibv_mr *cmds_mr;
		uint64_t send_cnt;
		struct time_stats slat;
		struct time_stats clat;
		struct ts_time clat_ts;
		enum rdmaio_si_state si_state;
	} tx;
	pthread_t io_thread;
	int (*io_handler)(struct rdma_connection *q);
	int id;

	enum rdmaio_cm_state  state;
	struct rdma_cm_id *cm_id;	/* client id or server child id */
	struct rdmacm_client_ctx *c_ctx;
	struct run_ctx *run_ctx;
};

struct rdmacm_client_ctx {
	pthread_t cm_thread;
	sem_t sem;
	struct rdma_connection *q;
	struct rdma_rw_ctx rw_ctx;
};

struct rdmacm_run_ctx {
	struct rdma_event_channel *channel;
	struct rdma_cm_event *event;
	struct sockaddr_storage local_addr;

	struct {
		struct rdma_cm_id *listen_cm_id;

		int next_free_client_index;
		struct rdma_connection clients[128];
	} s_ctx;
	struct rdmacm_client_ctx c_ctx;
};

struct run_ctx {
	struct ibv_context *context;
	char *ibdev_name;
	uint64_t size;
	uint64_t page_size;
	uint64_t align;
	uint64_t count;
	uint64_t offset;
	void *buf;
	int access_flags;

	int huge;
	int odp;
	int si;	/* Send with invalidate test */
	int read; /* Read test, mutually exclusive with SI test */
	int write; /* Write test, mutually exclusive with SI test */
	int write_pattern;
	char pattern;
	int server;
	char *host;
	uint16_t port;	/* network byte order */
	int connections;
	int skip_route_resolve;
	long long wait_time;
	int drop_ipc_lock_cap;
	uint64_t rlimit;
	uint64_t rlimit_set;

	struct rdmacm_run_ctx r_ctx;
 	struct sockaddr_storage sockaddr;
	struct sockaddr_storage src_sockaddr;
};

#define HUGE_PAGE_KPATH "/proc/sys/vm/nr_hugepages"

static int parse_address(struct sockaddr_storage *sock_addr,
			 const char *input_addr)
{
	struct addrinfo *info;
	int ret;

	ret = getaddrinfo(input_addr, NULL, NULL, &info);
	if (ret) {
		printf("err (%s) - invalid hostname or IP address\n",
		       gai_strerror(ret));
		return ret;
	}

	if (info->ai_family == PF_INET)
		memcpy(sock_addr, info->ai_addr, sizeof(struct sockaddr_in));
	else if (info->ai_family == PF_INET6)
		memcpy(sock_addr, info->ai_addr, sizeof(struct sockaddr_in6));
	else
		ret = -1;
	
	freeaddrinfo(info);
	return ret;
}

static void usage(const char *argv0)
{
	printf("Usage:\n");
	printf("%s\n", argv0);
	printf("Options:\n");
	printf("  -a --address=<ip_address>	ip address to connect to\n");
	printf("  -S --size=<size>		size of mr in bytes (default 4096)\n");
	printf("  -l --align=<align_size>	align memory allocation to this size\n");
	printf("  -n --count=<count>		number of wr operations\n");
	printf("  -u --huge			use huge pages\n");
	printf("  -j --skip			Skip route lookup for subsequent connections\n");
	printf("  -s --server			run in server mode\n");
	printf("  -c --client			run in client mode, connecting to <host>\n");
	printf("  -C --connections		number of connections\n");
	printf("  -B --bind			bind src addr from which to originate traffic\n");
	printf("  -o --odp			use ODP registration\n");
	printf("  -I --sendinvalidate		send invalidate operation test\n");
	printf("  -p --port			server port to listen on/connect to\n");
	printf("  -f --offset			use offset in registered MR for data transfer\n");
	printf("  -w --wait			wait time before starting the IOs such as 1hour or 2min\n");
	printf("  -h				display this help message\n");
	printf("  -r --rlimit=<bytes>      	memory resource hard limit in bytes\n");
	printf("  -v				display program version\n");
}

void version(const char *argv0)
{
	printf("%s %s\n", argv0, VERSION);
}

void parse_options(struct run_ctx *ctx, int argc, char **argv)
{
	int opt;
	int ret = 0;

	static struct option long_options[] = {
		{ .name = "address",  .has_arg = 1, .val = 'a' },
		{ .name = "bind",     .has_arg = 1, .val = 'B' },
		{ .name = "size",     .has_arg = 1, .val = 'S' },
		{ .name = "align",    .has_arg = 1, .val = 'l' },
		{ .name = "pattern",  .has_arg = 1, .val = 'P' },
		{ .name = "count",    .has_arg = 1, .val = 'n' },
		{ .name = "server",   .has_arg = 0, .val = 's' },
		{ .name = "sendinvalidate",   .has_arg = 0, .val = 'I' },
		{ .name = "rdmaread",   .has_arg = 0, .val = 'R' },
		{ .name = "rdmawrite",   .has_arg = 0, .val = 'W' },
		{ .name = "client",   .has_arg = 1, .val = '0' },
		{ .name = "connections",   .has_arg = 1, .val = 'C' },
		{ .name = "huge",     .has_arg = 0, .val = 'u' },
		{ .name = "port",     .has_arg = 1, .val = 'p' },
		{ .name = "skip",     .has_arg = 0, .val = 'j' },
		{ .name = "odp",      .has_arg = 0, .val = 'o' },
		{ .name = "wait",     .has_arg = 1, .val = 'w' },
		{ .name = "offset",   .has_arg = 1, .val = 'f' },
		{ .name = "rlimit",   .has_arg = 1, .val = 'r' },
		{ .name = NULL }
	};

	if (argc < 2) {
		usage(argv[0]);
		exit(1);
	}

	while ((opt = getopt_long(argc, argv, "hv:d:P:S:a:B:C:p:r:w:n:f:l:juoscIRWD", long_options, NULL)) != -1) {
		switch (opt) {
		case 'v':
			version(argv[0]);
			exit(0);
		case 'h':
			usage(argv[0]);
			exit(0);
		case 'a':
			ret = parse_address(&ctx->sockaddr, optarg);
			if (ret)
				goto err;
			break;
		case 'S':
			ctx->size = parse_size(optarg);
			break;
		case 'l':
			ctx->align = parse_size(optarg);
			break;
		case 'n':
			ctx->count = parse_size(optarg);
			break;
		case 'f':
			ctx->offset = parse_size(optarg);
			break;
		case 'C':
			ctx->connections = parse_size(optarg);
			break;
		case 'B':
			ret = parse_address(&ctx->src_sockaddr, optarg);
			if (ret)
				goto err;
			break;
		case 'P':
			ctx->write_pattern = 1;
			ctx->pattern = *((char*)optarg);
			break;
		case 'p':
			ctx->port = atoi(optarg);
			break;
		case 'u':
			ctx->huge = 1;
			break;
		case 'o':
			ctx->odp = 1;
			break;
		case 'I':
			ctx->si = 1;
			break;
		case 'R':
			ctx->read = 1;
			break;
		case 'W':
			ctx->write = 1;
			break;
		case 'j':
			ctx->skip_route_resolve = 1;
			break;
		case 'r':
			ctx->rlimit = parse_size(optarg);
			ctx->rlimit_set = 1;
			break;
		case 's':
			ctx->server = 1;
			break;
		case 'c':
			ctx->host = (char*)optarg;
			break;
		case 'w':
			ctx->wait_time = parse_time_seconds(optarg);
			printf("Wait time = %lld\n", ctx->wait_time);
			break;
		case 'D':
			ctx->drop_ipc_lock_cap = 1;
			break;
		}
	}
	return;

err:
	exit(1);
}

struct statistics {
	long long start, finish, load_time;
	long long min, max;
};

static int config_hugetlb_pages(uint64_t num_hpages)
{
	char hpages_str[128] = {0};
	size_t s;
	int err = 0;
	int fd;

	fd = open(HUGE_PAGE_KPATH, O_RDWR, 0);
	if (fd < 0)
		return fd;
	sprintf(hpages_str, "%ld", num_hpages);
	s = write(fd, hpages_str, strlen(hpages_str));
	if (s != strlen(hpages_str))
		err = -EINVAL;

	close(fd);
	return err;
}

static void reset_huge_tlb_pages(uint64_t num_pages)
{
	config_hugetlb_pages(num_pages);
}

static int config_hugetlb_kernel(struct run_ctx *ctx)
{
	long hpage_size = gethugepagesize();
	uint64_t num_hpages;

	if (hpage_size == 0)
		return -EINVAL;

	num_hpages = ctx->size / hpage_size;
	if (num_hpages == 0)
		num_hpages = 1;

	return config_hugetlb_pages(num_hpages);
}

static int alloc_mem(struct run_ctx *ctx)
{
	int err = 0;

	if (ctx->huge) {
		err = config_hugetlb_kernel(ctx);
		if (err) {
			fprintf(stderr,"fail to configure hugetlb\n");
			err = -EINVAL;
			return err;
		}
		ctx->buf = get_hugepage_region(ctx->size, GHR_STRICT | GHR_COLOR);
		if (!ctx->buf) {
			perror("mmap");
			err = -ENOMEM;
			return err;
		}
	} else {
		ctx->buf = memalign(ctx->align, ctx->size);
		if (!ctx->buf) {
			fprintf(stderr, "Couldn't allocate work buf.\n");
			err = -ENOMEM;
			return err;
		}
	}

	return err;
}

static void free_mem(struct run_ctx *ctx)
{
	if (ctx->huge) {
		free_hugepage_region(ctx->buf);
		reset_huge_tlb_pages(0);
	} else {
		free(ctx->buf);
	}
}

static struct rdma_cm_event *wait_for_event(struct run_ctx *ctx)
{
	int ret;

	ret = rdma_get_cm_event(ctx->r_ctx.channel, &ctx->r_ctx.event);
	if (ret) {
		fprintf(stderr,"%s null event.\n", __func__);
		return NULL;
	}

	fprintf(stderr,"rdmacm event: %s status = %d id = %p\n",
		rdma_event_str(ctx->r_ctx.event->event),
		ctx->r_ctx.event->status, ctx->r_ctx.event->id);

	if (ctx->r_ctx.event->event == RDMA_CM_EVENT_CONNECT_REQUEST ||
	    ctx->r_ctx.event->event == RDMA_CM_EVENT_ESTABLISHED) {
		fprintf(stderr,"listen_id = %p, id = %p\n",
			ctx->r_ctx.event->listen_id, ctx->r_ctx.event->id);
	}
	return ctx->r_ctx.event;
}

static int
rdma_ack_event(struct run_ctx *ctx)
{
	int ret;

	ret = rdma_ack_cm_event(ctx->r_ctx.event);
	return ret;
}

static int set_rlimit(struct run_ctx *ctx)
{
	struct rlimit rlim, after_rlim;
	int ret;

	ret = getrlimit(RLIMIT_MEMLOCK, &rlim);
	if (ret)
		return ret;

	if (ctx->rlimit_set) {
		rlim.rlim_cur = ctx->rlimit;
		rlim.rlim_max = ctx->rlimit;
		ret = setrlimit(RLIMIT_MEMLOCK, &rlim);
		if (ret)
			return ret;

		ret = getrlimit(RLIMIT_MEMLOCK, &after_rlim);
		if (after_rlim.rlim_max != ctx->rlimit) {
			fprintf(stderr, "Set rlimit %ld, Got %ld\n",
				ctx->rlimit, after_rlim.rlim_max);
			return -EINVAL;
		}
	}
	return ret;
}

static int drop_ipc_lock_cap(void)
{
	cap_value_t capList[1];
	cap_t caps;
	int ret;

	/* Retrieve caller's current capabilities */
	caps = cap_get_proc();
	if (caps == NULL)
		return -EINVAL;

	/* Change setting of 'capability' in the effective set of 'caps'. The
	 * third argument, 1, is the number of items in the array 'capList'.
	 */
	capList[0] = CAP_IPC_LOCK;
	ret = cap_set_flag(caps, CAP_EFFECTIVE, 1, capList, CAP_CLEAR);
	if (ret)
		goto err;

	ret = cap_set_proc(caps);
	if (ret)
		goto err;

	ret = cap_set_flag(caps, CAP_PERMITTED, 1, capList, CAP_CLEAR);
	if (ret)
		goto err;

	ret = cap_set_proc(caps);
err:
	cap_free(caps);
	return ret;
}

static int setup_ipc_lock_cap(struct run_ctx *ctx)
{
	int ret = 0;

	if (ctx->drop_ipc_lock_cap)
		ret = drop_ipc_lock_cap();
	return ret;
}

static int create_q(struct rdma_connection *q)
{
	struct ibv_qp_init_attr qp_attr;
	int ret;

	memset(&qp_attr, 0, sizeof(qp_attr));

	q->pd = ibv_alloc_pd(q->cm_id->verbs);
	if (!q->pd) {
		fprintf(stderr,"%s fail to create pd\n", __func__);
		return -ENOMEM;
	}

	q->cq_channel = ibv_create_comp_channel(q->cm_id->verbs);
	if (!q->cq_channel) {
		fprintf(stderr,"%s fail to create cq channel\n", __func__);
		return -ENOMEM;
	}

	q->cq = ibv_create_cq(q->cm_id->verbs,
			      RDMAIO_Q_DEPTH * 2, q,
			      q->cq_channel,
			      0);
	if (!q->cq) {
		fprintf(stderr,"%s fail to create cq\n", __func__);
		return -ENOMEM;
	}
	ret = ibv_req_notify_cq(q->cq, 0);
	if (ret)
		return ret;

	qp_attr.qp_context = q;
	qp_attr.send_cq = q->cq;
	qp_attr.recv_cq = q->cq;
	qp_attr.qp_type = IBV_QPT_RC;
	qp_attr.sq_sig_all = 1;
	qp_attr.cap.max_send_wr = RDMAIO_Q_DEPTH;
	qp_attr.cap.max_send_sge = 1;
	qp_attr.cap.max_recv_wr = RDMAIO_Q_DEPTH;
	qp_attr.cap.max_recv_sge = 1;

	ret = rdma_create_qp(q->cm_id, q->pd, &qp_attr);
	if (ret) {
		fprintf(stderr,"%s fail to create qp ret = %d\n", __func__, ret);
		goto err;
	}
	return ret;

err:
	ibv_destroy_cq(q->cq);
	ibv_destroy_comp_channel(q->cq_channel);
	ibv_dealloc_pd(q->pd);
	return ret;
}

static void init_rx_cmd_wrs(struct rdma_connection *q,
			    int rx_msg_size, int rx_msg_count)
{
	int i;

	fprintf(stderr,"%s rx_msg size = %d, count = %d\n", __func__,
		rx_msg_size, rx_msg_count);

	for (i = 0; i < rx_msg_count; i++) {
		q->rx.wrs[i].cmd = &q->rx.cmds[i];
		q->rx.wrs[i].sge.addr = (uintptr_t)&q->rx.cmds[i];
		q->rx.wrs[i].sge.length = rx_msg_size;
		q->rx.wrs[i].sge.lkey = q->rx.cmds_mr->lkey;

		q->rx.wrs[i].wr.wr_id = (uintptr_t)q->rx.wrs[i].cmd;
		q->rx.wrs[i].wr.sg_list = &q->rx.wrs[i].sge;
		q->rx.wrs[i].wr.num_sge = 1;
	}
}

static int post_recv_wr(struct rdma_connection *q, int index)
{
	struct ibv_recv_wr *bad_wr;
	int ret;

	ret = ibv_post_recv(q->cm_id->qp, &q->rx.wrs[index].wr, &bad_wr);
	if (!ret)
		q->rx.rqe_post_cnt++;
	return ret;
}

static int
setup_rx_cmds_buffers(struct run_ctx *ctx, struct rdma_connection *q,
		      int msg_size, int msg_count)
{
	int ret;
	int i;

	q->rx.cmds = calloc(msg_count, msg_size);
	if (!q->rx.cmds)
		return -ENOMEM;

	memset(q->rx.cmds, 0, msg_size * msg_count);

	q->rx.cmds_mr = ibv_reg_mr(q->pd, q->rx.cmds,
				   msg_size * msg_count,
				   ctx->access_flags);
	if (!q->rx.cmds_mr)
		return -ENOMEM;
	fprintf(stdout,"%s rx cmds mr lkey = 0x%x\n", __func__, q->rx.cmds_mr->lkey);

	q->rx.wrs = calloc(msg_count, sizeof(*q->rx.wrs));
	if (!q->rx.wrs)
		return -ENOMEM;

	init_rx_cmd_wrs(q, msg_size, msg_count);

	for (i = 0; i < msg_count; i++) {
		ret = post_recv_wr(q, i);
		if (ret)
			break;
	}
	fprintf(stdout,"%s i= %d ret = %d\n", __func__, i, ret);
	return ret;
}

static int
setup_tx_cmds_buffers(struct run_ctx *ctx, struct rdma_connection *q,
		      int msg_size, int msg_count)
{
	q->tx.cmds = calloc(msg_count, msg_size);
	if (!q->tx.cmds)
		return -ENOMEM;

	memset(q->tx.cmds, 0, msg_size * msg_count);

	q->tx.cmds_mr = ibv_reg_mr(q->pd, q->tx.cmds,
				   msg_size * msg_count,
				   ctx->access_flags);
	if (!q->tx.cmds_mr)
		return -ENOMEM;

	fprintf(stdout,"%s tx cmds mr key = 0x%x\n", __func__, q->tx.cmds_mr->lkey);

	q->tx.wrs = calloc(msg_count, sizeof(*q->tx.wrs));
	if (!q->tx.wrs)
		return -ENOMEM;

	return 0;
}

static void init_tx_cmd_wrs(struct rdma_connection *q,
			    int msg_size, int msg_count)
{
	int i;

	fprintf(stdout,"%s tx msg_size = %d, count = %d\n", __func__,
		msg_size, msg_count);

	for (i = 0; i < msg_count; i++) {
		q->tx.wrs[i].cmd = &q->tx.cmds[i];
		q->tx.wrs[i].sge.addr = (uintptr_t)&q->tx.cmds[i];
		q->tx.wrs[i].sge.length = msg_size;
		q->tx.wrs[i].sge.lkey = q->tx.cmds_mr->lkey;

		q->tx.wrs[i].send_wr.wr_id = (uintptr_t)q->tx.wrs[i].cmd;
		q->tx.wrs[i].send_wr.sg_list = &q->tx.wrs[i].sge;
		q->tx.wrs[i].send_wr.num_sge = 1;

		q->tx.wrs[i].send_wr.send_flags = IBV_SEND_SIGNALED;
	}
}

static int setup_msg_buffers(struct run_ctx *ctx, struct rdma_connection *q)
{
	int ret;

	ret = setup_rx_cmds_buffers(ctx, q, sizeof(struct rdmaio_msg),
				    RDMAIO_Q_DEPTH);
	if (ret)
		return ret;

	ret = setup_tx_cmds_buffers(ctx, q, sizeof(struct rdmaio_msg),
				    RDMAIO_Q_DEPTH);
	if (ret)
		return ret;
	
	init_tx_cmd_wrs(q, sizeof(struct rdmaio_msg), RDMAIO_Q_DEPTH);
	return 0;
}

static int get_tx_index(const struct rdma_connection *q)
{
	return q->tx.send_cnt % RDMAIO_Q_DEPTH;
}

static struct rdmaio_msg *get_tx_msg(const struct rdma_connection *q)
{
	int index = get_tx_index(q);

	return q->tx.wrs[index].cmd;
}

static struct ibv_send_wr *get_tx_wr(const struct rdma_connection *q)
{
	int index = get_tx_index(q);

	return &q->tx.wrs[index].send_wr;
}

static int get_rx_index(struct rdma_connection *q)
{
	return q->rx.rqe_post_cnt % RDMAIO_Q_DEPTH;
}

static int post_sq_wr(struct rdma_connection *q, struct ibv_send_wr *sq_wr)
{
	struct ts_time sstat = { 0 };
	struct ibv_send_wr *bad_wr;
	int ret;

	ts_log_start_time(&sstat);
	ret = ibv_post_send(q->cm_id->qp, sq_wr, &bad_wr);
	if (!ret)
		q->tx.send_cnt++;
	else
		fprintf(stderr,"%s sq opcode = %d, ret = %d\n", __func__,
		       sq_wr->opcode, ret);
	ts_log_end_time(&sstat);
	ts_update_time_stats(&sstat, &q->tx.slat);
	ts_log_start_time(&q->tx.clat_ts);
	return ret;
}

static int post_rdma_wr(struct rdma_connection *q)
{
	struct ibv_send_wr *rdma_wr = get_tx_wr(q);
	struct ibv_sge sge;

	memset(rdma_wr, 0, sizeof(*rdma_wr));
	memset(&sge, 0, sizeof(sge));

	if (q->run_ctx->read)
		rdma_wr->opcode = IBV_WR_RDMA_READ;
	else if (q->run_ctx->write)
		rdma_wr->opcode = IBV_WR_RDMA_WRITE;

	if (q->run_ctx->r_ctx.c_ctx.rw_ctx.length) {
		rdma_wr->wr.rdma.rkey = q->run_ctx->r_ctx.c_ctx.rw_ctx.rkey;
		rdma_wr->wr.rdma.remote_addr = q->run_ctx->r_ctx.c_ctx.rw_ctx.raddr;
		rdma_wr->num_sge = 1;
		rdma_wr->sg_list = &sge;
	}
	rdma_wr->send_flags = IBV_SEND_SIGNALED;

	sge.addr = (uintptr_t)q->run_ctx->r_ctx.c_ctx.rw_ctx.local_mr->addr;
	sge.length = q->run_ctx->r_ctx.c_ctx.rw_ctx.length;
	sge.lkey = q->run_ctx->r_ctx.c_ctx.rw_ctx.local_mr->lkey;

	return post_sq_wr(q, rdma_wr);
}

/*
 * client-server ladder diagram:
 * client:			server:
 * Send MR alloc req:
 * 				Recv MR alloc req:
 * 				allocate MR
 * 				Send MR allocated resp.
 * Recv MR alloc resp:
 * Do send_with_invalidate:
 * 				dealloc_MR:
 * Reset state to for new IO.
 */

static int prepare_send_alloc_mr_req(struct rdma_connection *q)
{
	struct ibv_send_wr *tx_wr = get_tx_wr(q);
	struct rdmaio_msg *msg = get_tx_msg(q);

	msg->cmd_or_rsp = RDMAIO_CMD;
	msg->opc = RDMA_IO_OPC_REG_MR_REQ;
	msg->io_handle = (uintptr_t)msg;
	msg->u.alloc_mr_cmd.mr_len = q->run_ctx->size;

	tx_wr->opcode = IBV_WR_SEND;
	return post_sq_wr(q, tx_wr);
}

static int client_si_ops_handler(struct rdma_connection *q)
{
	int ret = 0;

restart_state:
	switch (q->tx.si_state) {
	case RDMAIO_CLIENT_STATE_START_IO:
		ret = prepare_send_alloc_mr_req(q);
		q->tx.si_state = RDMAIO_CLIENT_STATE_MR_REQ_SENT;
		break;
	case RDMAIO_CLIENT_STATE_MR_REQ_SENT:
		/* np for mr req send completion */
		break;
	case RDMAIO_CLIENT_STATE_SI_SENT:
		/* Send with invalidate completion arrived,
		 * Restart the sequence.
		 */
		q->tx.si_state = RDMAIO_CLIENT_STATE_START_IO;
		goto restart_state;
		break;
	default:
		break;
	};
	return ret;
}

static void client_send_handler(struct rdma_connection *q)
{
	struct run_ctx *ctx = q->run_ctx;
	struct ibv_send_wr *tx_wr = get_tx_wr(q);
	int ret = 0;

	ts_log_end_time(&q->tx.clat_ts);
	ts_update_time_stats(&q->tx.clat_ts, &q->tx.clat);

	if (ctx->si) {
		ret = client_si_ops_handler(q);
	} else if (ctx->read || ctx->write) {
		/* nop */
	} else {
		tx_wr->opcode = IBV_WR_SEND;
		ret = post_sq_wr(q, tx_wr);
	}
	if (ret)
		fprintf(stderr,"%s ret = %d\n", __func__, ret);
}

static void client_rw_handler(struct rdma_connection *q)
{
	ts_log_end_time(&q->tx.clat_ts);
	ts_update_time_stats(&q->tx.clat_ts, &q->tx.clat);
	post_rdma_wr(q);
}

static int client_setup_rw_buffer(struct rdma_connection *q)
{
	struct run_ctx *ctx = q->run_ctx;
	size_t length = q->run_ctx->size;
	struct ibv_mr *mr;
	void *buffer;

	if (!ctx->read && !ctx->write)
		return 0;

	buffer = malloc(length);
	if (!buffer)
		return -ENOMEM;

	mr = ibv_reg_mr(q->pd, buffer, length,
			IBV_ACCESS_LOCAL_WRITE |
			IBV_ACCESS_REMOTE_READ |
			IBV_ACCESS_REMOTE_WRITE);
	if (!mr)
		return -ENOMEM;

	ctx->r_ctx.c_ctx.rw_ctx.local_mr = mr;
	ctx->r_ctx.c_ctx.rw_ctx.length = 0;
	return 0;
}

static int client_io_warmup(struct rdma_connection *q)
{
	struct run_ctx *ctx = q->run_ctx;
	struct ibv_send_wr *tx_wr = get_tx_wr(q);
	int ret;

	if (ctx->si)
		ret = client_si_ops_handler(q);
	else if (ctx->read || ctx->write) {
		ret = prepare_send_alloc_mr_req(q);
		q->tx.si_state = RDMAIO_CLIENT_STATE_MR_REQ_SENT;
	} else {
		tx_wr->opcode = IBV_WR_SEND;
		ret = post_sq_wr(q, tx_wr);
	}
	if (ret)
		fprintf(stderr,"ret = %d\n", ret);
	return ret;
}

static void
client_recv_resp_handler(struct rdma_connection *q,
			 const struct rdmaio_msg *msg)
{
	struct run_ctx *ctx = q->run_ctx;
	struct ibv_send_wr *tx_wr = get_tx_wr(q);
	int rx_index;
	int ret = 0;

	rx_index = get_rx_index(q);

	if (ctx->si) {
		tx_wr->invalidate_rkey = msg->u.alloc_mr_rsp.mr_key;
		tx_wr->opcode = IBV_WR_SEND_WITH_INV;

		ret = post_sq_wr(q, tx_wr);
		q->tx.si_state = RDMAIO_CLIENT_STATE_SI_SENT;

		fprintf(stdout,"%s mr len=%d, key=0x%x\n", __func__,
			msg->u.alloc_mr_rsp.mr_len, msg->u.alloc_mr_rsp.mr_key);
	} else if (ctx->read || ctx->write) {
		fprintf(stdout,"%s mr len=%d, key=0x%x\n", __func__,
			msg->u.alloc_mr_rsp.mr_len, msg->u.alloc_mr_rsp.mr_key);
		q->run_ctx->r_ctx.c_ctx.rw_ctx.raddr = msg->u.alloc_mr_rsp.addr;
		q->run_ctx->r_ctx.c_ctx.rw_ctx.rkey = msg->u.alloc_mr_rsp.mr_key;

		ret = post_rdma_wr(q);

		q->tx.si_state = RDMAIO_CLIENT_STATE_RW_SENT;
	} else {
		tx_wr->opcode = IBV_WR_SEND;
		ret = post_sq_wr(q, tx_wr);
	}

	if (ret)
		fprintf(stderr,"%s ret = %d\n", __func__, ret);
	post_recv_wr(q, rx_index);
}

static void
client_recv_wc_handler(struct rdma_connection *q, struct ibv_wc *wc)
{
	struct rdmaio_msg *msg = (void*)wc->wr_id;

	switch (msg->cmd_or_rsp) {
	case RDMAIO_CMD:
		break;
	case RDMAIO_RSP:
		client_recv_resp_handler(q, msg);
		break;
	};
}

static int client_io_handler(struct rdma_connection *q)
{
	struct ibv_wc wc;
	int ret = 0;

	ret = ibv_poll_cq(q->cq, 1, &wc);
	if (ret <= 0)
		return ret;

	fprintf(stdout, "cqe opcode = %d, status = %d %s\n",
		wc.opcode, wc.status,
		ibv_wc_status_str(wc.status));
	if (wc.status) {
		fprintf(stderr, "cqe opcode = %d, status = %d %s\n",
			wc.opcode, wc.status,
			ibv_wc_status_str(wc.status));
		return -1;
	}

	switch (wc.opcode) {
	case IBV_WC_SEND:
		client_send_handler(q);
		break;
	case IBV_WC_RDMA_WRITE:
	case IBV_WC_RDMA_READ:
		client_rw_handler(q);
		break;
	case IBV_WC_RECV:
		client_recv_wc_handler(q, &wc);
		break;
	default:
		fprintf(stderr,"unknown completion status=%d opcode=%d\n",
			wc.status, wc.opcode);
		ret = -1;
		break;
	}
	if (ret < 0)
		fprintf(stderr, "%s cq poll error=%d\n", __func__, ret);
	return ret;
}

static void
server_reg_mr_handler(struct rdma_connection *q,
			 const struct rdmaio_msg *msg, int index)
{
	struct rdmaio_msg *mr_rsp = get_tx_msg(q);
	struct ibv_send_wr *tx_wr = get_tx_wr(q);
	struct ibv_mr *mr;
	void *buffer;

	buffer = malloc(msg->u.alloc_mr_cmd.mr_len);
	
	mr = ibv_reg_mr(q->pd, buffer, msg->u.alloc_mr_cmd.mr_len,
			IBV_ACCESS_LOCAL_WRITE |
			IBV_ACCESS_REMOTE_READ |
			IBV_ACCESS_REMOTE_WRITE);

	q->rx.io_mr = mr;
	mr_rsp->cmd_or_rsp = RDMAIO_RSP;
	mr_rsp->opc = RDMA_IO_OPC_REG_MR_RSP;
	mr_rsp->io_handle = msg->io_handle;
	mr_rsp->u.alloc_mr_rsp.mr_len = msg->u.alloc_mr_cmd.mr_len;
	mr_rsp->u.alloc_mr_rsp.mr_key = mr->rkey;
	mr_rsp->u.alloc_mr_rsp.addr = (uintptr_t)buffer;
	
	fprintf(stdout,"%s mr len=%d, key=0x%x, addr=%p\n", __func__,
		msg->u.alloc_mr_cmd.mr_len, mr->rkey, buffer);

	post_recv_wr(q, index);

	tx_wr->opcode = IBV_WR_SEND;
	post_sq_wr(q, tx_wr);
}

static void
server_cmd_handler(struct rdma_connection *q,
		      const struct rdmaio_msg *msg, int index)
{
	switch (msg->opc) {
	case RDMA_IO_OPC_REG_MR_REQ:
		server_reg_mr_handler(q, msg, index);
		break;
	};
}

static void
server_recv_cqe_handler(struct rdma_connection *q, const struct ibv_wc *wc,
			int index)
{
	struct rdmaio_msg *msg = (void*)wc->wr_id;

	switch (msg->cmd_or_rsp) {
	case RDMAIO_CMD:
		server_cmd_handler(q, msg, index);
		break;
	default:
		break;
	};
}

static void server_si_recv_cqe_handler(struct rdma_connection *q, int index)
{
	post_recv_wr(q, index);
	ibv_dereg_mr(q->rx.io_mr);
	q->rx.io_mr = NULL;
}

static void server_receiver(struct rdma_connection *q,
			    const struct ibv_wc *wc)
{
	struct run_ctx *ctx = q->run_ctx;
	int index;

	index = get_rx_index(q);

	if (ctx->si || ctx->read || ctx->write) {
		if (wc->wc_flags & IBV_WC_WITH_INV)
			server_si_recv_cqe_handler(q, index);
		else
			server_recv_cqe_handler(q, wc, index);
	} else {
		post_recv_wr(q, index);
	}
}

static int server_io_handler(struct rdma_connection *q)
{
	struct ibv_wc wc;
	int ret;

	while (1) {
		ret = ibv_poll_cq(q->cq, 1, &wc);
		if (ret <= 0)
			break;

		fprintf(stderr, "cqe opcode = %d, status = %d %s\n",
			wc.opcode, wc.status,
			ibv_wc_status_str(wc.status));
		if (wc.status) {
			fprintf(stderr, "cqe opcode = %d, status = %d %s\n",
				wc.opcode, wc.status,
				ibv_wc_status_str(wc.status));
			ret = -1;
			goto err;
		}

		switch (wc.opcode) {
		case IBV_WC_SEND:
			break;
		case IBV_WC_RDMA_WRITE:
			break;
		case IBV_WC_RDMA_READ:
			break;
		case IBV_WC_RECV:
			server_receiver(q, &wc);
			break;
		default:
			fprintf(stderr,"unknown completion status=%d opcode=%d\n",
				wc.status, wc.opcode);
			ret = -1;
			goto err;
		}
	}
	if (ret) {
		fprintf(stderr, "cq poll error=%d\n", ret);
		goto err;
	}
	return 0;

err:
	return ret;
}

static void wait_for_cqe_event(struct rdma_connection *q)
{

	struct ibv_cq *event_cq;
	void *event_ctx;
	int ret = 0;

	ret = ibv_get_cq_event(q->cq_channel, &event_cq, &event_ctx);
	if (ret) {
		fprintf(stderr, "Failed to get cq event.\n");
		pthread_exit(NULL);
	}
	if (event_cq != q->cq) {
		fprintf(stderr, "Invalid cq =%p\n", event_cq);
		pthread_exit(NULL);
	}
	ret = ibv_req_notify_cq(q->cq, 0);
	if (ret) {
		fprintf(stderr, "fail to arm cq=%p\n", event_cq);
		pthread_exit(NULL);
	}
}

static int io_allowed(struct rdma_connection *q)
{
	if (q->run_ctx->count > q->tx.send_cnt)
		return 1;

	return 0;
}

static int thread_loop(struct rdma_connection *q)
{
	int ret = 0;

	while (1) {
		pthread_testcancel();

		if (!io_allowed(q))
			pthread_exit(NULL);

		wait_for_cqe_event(q);

		ret = q->io_handler(q);

		ibv_ack_cq_events(q->cq, 1);
		if (ret < 0)
			pthread_exit(NULL);
	}
	return ret;
}

static void *server_io_thread(void *arg)
{
	struct rdma_connection *q = arg;

	thread_loop(q);
	return NULL;
}

static int server_start_io_thread(struct rdma_connection *q)
{
	return pthread_create(&q->io_thread, NULL, server_io_thread, q);
}

static void *client_io_thread(void *arg)
{
	struct rdma_connection *q = arg;
	int ret;

	ret = client_setup_rw_buffer(q);
	if (ret) {
		fprintf(stderr, "Failed to post warmup io requests\n");
		pthread_exit(NULL);
	}
	ret = client_io_warmup(q);
	if (ret) {
		fprintf(stderr, "Failed to post warmup io requests\n");
		pthread_exit(NULL);
	}
	thread_loop(q);
	return NULL;
}

static int
client_start_io_thread(struct run_ctx *ctx, struct rdma_connection *q)
{
	ctx->r_ctx.c_ctx.q = q;

	if (ctx->wait_time)
		sleep(ctx->wait_time);

	return pthread_create(&q->io_thread, NULL, client_io_thread, q);
}

static struct rdma_connection* alloc_q(struct run_ctx *ctx,
			int (*io_handler)(struct rdma_connection *q))
{
	struct rdma_connection *q;
	int free_idx;

	free_idx = ctx->r_ctx.s_ctx.next_free_client_index;

	q = &ctx->r_ctx.s_ctx.clients[free_idx];
	memset(q, 0, sizeof(*q));
	q->id = free_idx;
	q->msg_count = RDMAIO_Q_DEPTH;
	ctx->r_ctx.s_ctx.next_free_client_index++;
	q->run_ctx = ctx;
	q->io_handler = io_handler;
	ts_init(&q->tx.slat);
	ts_init(&q->tx.clat);
	return q;
}

static int server_handle_connect(struct run_ctx *ctx,
				 struct rdma_cm_id *child_id)
{
	struct rdma_conn_param param = { 0 };
	struct rdma_connection *q;
	char resp[20] = { 0 };
	int ret;

	fprintf(stdout,"%s allocating q memory\n", __func__);
	q = alloc_q(ctx, server_io_handler);
	if (!q)
		return -EINVAL;

	fprintf(stdout,"%s creating q id = %p\n", __func__, child_id);
	q->cm_id = child_id;
	q->cm_id->context = q;
	ret = create_q(q);
	if (ret)
		return ret;

	fprintf(stdout,"%s qp = %d\n", __func__, q->cm_id->qp->qp_num);
	param.rnr_retry_count = 7;
	param.flow_control = 1;
	param.initiator_depth = 16;
	param.responder_resources = 16;
	param.private_data = &resp[0];
	param.private_data_len = sizeof(resp);

	ret = rdma_accept(q->cm_id, &param);
	if (ret)
		goto err;

	fprintf(stdout,"%s accepted connection = %p\n", __func__, child_id);
	ret = setup_msg_buffers(ctx, q);
	if (ret)
		goto err;

	fprintf(stdout,"%s completed msg buffers setup = %p\n",
		__func__, child_id);
	ret = server_start_io_thread(q);
	if (ret)
		goto io_err;

	fprintf(stdout,"%s io thread started = %p\n", __func__, child_id);
	return 0;

io_err:
	rdma_disconnect(q->cm_id);
err:
	rdma_destroy_qp(q->cm_id);
	return ret;
}

static int server_handle_disconnect(struct run_ctx *ctx,
				    struct rdma_cm_id *child_id)
{
	struct rdma_connection *q;

	if (!ctx)
		return -EINVAL;

	q = child_id->context;

	fprintf(stdout,"%s qpn = %d, rqe post count = %d\n", __func__,
	       q->cm_id->qp->qp_num, q->rx.rqe_post_cnt);
	return 0;
}

static int setup_server(struct run_ctx *ctx)
{
	struct rdma_cm_event *event;
	struct sockaddr_in *addr_in;
	struct sockaddr_in6 *addr_in6;
	struct rdma_cm_id *child_id;
	int ret;

	ret = rdma_create_id(ctx->r_ctx.channel, &ctx->r_ctx.s_ctx.listen_cm_id,
			     NULL, RDMA_PS_TCP);
	if (ret)
		return ret;

	if (ctx->src_sockaddr.ss_family) {
		if (ctx->src_sockaddr.ss_family == AF_INET) {
			addr_in = (struct sockaddr_in *)&ctx->src_sockaddr;
			addr_in->sin_port = htons(ctx->port);
		} else {
			addr_in6 = (struct sockaddr_in6 *)&ctx->src_sockaddr;
			addr_in6->sin6_port = htons(ctx->port);
		}
	} else {
		addr_in = (struct sockaddr_in *)&ctx->src_sockaddr;
		addr_in->sin_family = AF_INET;
		addr_in->sin_addr.s_addr = htonl(INADDR_ANY);
		addr_in->sin_port = htons(ctx->port);
	}

	ret = rdma_bind_addr(ctx->r_ctx.s_ctx.listen_cm_id,
			     (struct sockaddr *)addr_in);
	if (ret) {
		perror("Fail to bind: ");
		fprintf(stdout,"err = %s\n", gai_strerror(ret));
		return ret;
	}

	ret = rdma_listen(ctx->r_ctx.s_ctx.listen_cm_id, 1024);
	if (ret)
		return ret;

	while (1) {
		event = wait_for_event(ctx);
		if (!event)
			break;

		child_id = ctx->r_ctx.event->id;
		switch (event->event) {
		case RDMA_CM_EVENT_CONNECT_REQUEST:
			ret = rdma_ack_event(ctx);
			ret = server_handle_connect(ctx, child_id);
			break;
		case RDMA_CM_EVENT_ESTABLISHED:
			ret = rdma_ack_event(ctx);
			break;
		case RDMA_CM_EVENT_DISCONNECTED:
			ret = server_handle_disconnect(ctx, child_id);
			ret = rdma_ack_event(ctx);
			break;
		default:
			ret = rdma_ack_event(ctx);
			break;
		}
	}

	fprintf(stdout,"%s status = %d\n", __func__, ret);
	return ret;
}

static int rdmacm_client_cm_event_handler(struct rdma_cm_id *cm_id,
					  struct rdma_cm_event *event)
{
	struct rdma_connection *q = cm_id->context;
	struct rdmacm_client_ctx *c_ctx;

	fprintf(stdout,"%s q = %p\n", __func__, q);

	c_ctx = q->c_ctx;

	switch (event->event) {
	case RDMA_CM_EVENT_ADDR_RESOLVED:
		q->state = RDMAIO_CM_STATE_ADDR_RESOLVED;
		break;
	case RDMA_CM_EVENT_ROUTE_RESOLVED:
		q->state = RDMAIO_CM_STATE_ROUTE_RESOLVED;
		break;
	case RDMA_CM_EVENT_ESTABLISHED:
		q->state = RDMAIO_CM_CONNECTED;
		break;
	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
	case RDMA_CM_EVENT_REJECTED:
		q->state = RDMAIO_CM_DISCONNECTED;
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		q->state = RDMAIO_CM_DISCONNECTED;
		break;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		q->state = RDMAIO_CM_FATAL_ERROR;
		break;
	default:
		fprintf(stderr, "unknown event: %s\n",
			rdma_event_str(event->event));
		break;
	}
	sem_post(&c_ctx->sem);
	return 0;
}

static void* client_cm_thread(void *arg)
{
	struct run_ctx *ctx = arg;
	struct rdma_cm_event *event;

	while (1) {
		event = wait_for_event(ctx);
		if (!event)
			continue;

		rdmacm_client_cm_event_handler(event->id, event);
		rdma_ack_event(ctx);
	}
	return NULL;
}

static int client_setup_cm_thread(struct run_ctx *ctx)
{
	int ret;

	ret = pthread_create(&ctx->r_ctx.c_ctx.cm_thread, NULL,
			     client_cm_thread, ctx);
	return ret;
}

static void setup_path_record(struct ibv_path_data *out,
			     struct ibv_sa_path_rec *cached)
{
	memset(out, 0, sizeof(*out));

	out->flags = IBV_PATH_FLAG_PRIMARY | IBV_PATH_FLAG_GMP |
		     IBV_PATH_FLAG_BIDIRECTIONAL;
	out->path.service_id = 0;
	out->path.dgid = cached->dgid;
	out->path.sgid = cached->sgid;
	out->path.flowlabel_hoplimit = 0xff;
	out->path.reversible_numpath = (1 << 7) | 0x1; /* reversible-7:7 num path-6:0 */
	out->path.pkey = 0xffff;
	out->path.mtu = (cached->mtu_selector << 6) | cached->mtu;
	out->path.preference = cached->preference;
}

static int client_setup_one_connection(struct run_ctx *ctx)
{
	struct rdma_conn_param conn_param;
	struct rdma_connection *q; 
	struct sockaddr_in *in4;
	int ret;

	q = alloc_q(ctx, client_io_handler);
	if (!q)
		return -ENOMEM;

	ret = rdma_create_id(ctx->r_ctx.channel, &q->cm_id, NULL, 
			     RDMA_PS_TCP);
	if (ret)
		return ret;
	q->c_ctx = &ctx->r_ctx.c_ctx;
	q->cm_id->context = q;

	in4 = (struct sockaddr_in *)&ctx->sockaddr;
	in4->sin_family = AF_INET;
	in4->sin_port = htons(ctx->port);
	if (ctx->src_sockaddr.ss_family) 
		ret = rdma_resolve_addr(q->cm_id,
				(struct sockaddr *)&ctx->src_sockaddr,
				(struct sockaddr *)&ctx->sockaddr, 5000);
	else
		ret = rdma_resolve_addr(q->cm_id,
				NULL,
				(struct sockaddr *)&ctx->sockaddr, 5000);
	if (ret) {
		perror("rdma_resolve_addr");
		return ret;
	}
	sem_wait(&ctx->r_ctx.c_ctx.sem);
	if (q->state != RDMAIO_CM_STATE_ADDR_RESOLVED)
		return -EINVAL;

	/* Check if route resolve needs to be skipped for multiple
	 * connections.
	 * First connection needs to have the route resolved to reuse
	 * for subsequent connections.
	 */
	if (!ctx->skip_route_resolve || !q->id) {
		fprintf(stdout,"%s resolving route q = %p\n", __func__, q);
		ret = rdma_resolve_route(q->cm_id, 5000);
		if (ret) {
			q->state = RDMAIO_CM_FATAL_ERROR;
			perror("rdma_resolve_route");
		}
	} else {
		struct ibv_path_data path_record;
		fprintf(stdout,"num paths for first connection=%d\n",
			ctx->r_ctx.s_ctx.clients[0].cm_id->route.num_paths);

		setup_path_record(&path_record,
				  &ctx->r_ctx.s_ctx.clients[0].cm_id->route.path_rec[0]);

		fprintf(stdout,"%s setting options q = %p\n", __func__, q);

		ret = rdma_set_option(q->cm_id, RDMA_OPTION_IB,
				      RDMA_OPTION_IB_PATH,
				      &path_record,
				      sizeof(path_record));
		if (ret) {
			fprintf(stderr,"%s fail to set path record option ret = %d\n", __func__, ret);
			return ret;
		}
	}
	sem_wait(&ctx->r_ctx.c_ctx.sem);
	if (q->state != RDMAIO_CM_STATE_ROUTE_RESOLVED) {
		fprintf(stderr,"%s q = %p, state = %d\n", __func__, q, q->state);
		return -EINVAL;
	}

	fprintf(stdout,"%s creating q =%p\n", __func__, q);
	ret = create_q(q);
	if (ret) {
		fprintf(stderr,"%s fail to create q = %d\n", __func__, ret);
		return ret;
	}

	memset(&conn_param, 0, sizeof(conn_param));
	conn_param.responder_resources = 16;
	conn_param.initiator_depth = 16;
	conn_param.retry_count = 7;

	ret = rdma_connect(q->cm_id, &conn_param);
	if (ret) {
		fprintf(stderr,"%s fail to connect ret = %d\n", __func__, ret);
		return ret;
	}
	sem_wait(&ctx->r_ctx.c_ctx.sem);
	if (q->state != RDMAIO_CM_CONNECTED)
		return -EINVAL;
	
	ret = setup_msg_buffers(ctx, q);
	if (ret)
		return ret;

	ret = client_start_io_thread(ctx, q);
	return ret;
}

static int setup_client(struct run_ctx *ctx)
{
	int ret;
	int i;

	sem_init(&ctx->r_ctx.c_ctx.sem, 0, 0);

	ret = client_setup_cm_thread(ctx);
	if (ret)
		return ret;

	for (i = 0; i < ctx->connections; i++) {
		ret = client_setup_one_connection(ctx);
		if (ret)
			break;
	}
	if (ret)
		return ret;

	pthread_join(ctx->r_ctx.c_ctx.q->io_thread, NULL);
	return 0;
}

static int setup_test(struct run_ctx *ctx)
{
	int err = 0;

	err = set_rlimit(ctx);
	if (err) {
		fprintf(stderr, "Couldn't change rlimit size %ld\n",
			ctx->rlimit);
		goto err;
	}

	err = setup_ipc_lock_cap(ctx);
	if (err) {
		fprintf(stderr, "Couldn't drop ipc lock capability\n");
		goto err;
	}

	err = alloc_mem(ctx);
	if (err) {
		fprintf(stderr, "Couldn't allocate memory of size %ld\n",
			ctx->size);
		err = -ENODEV;
		goto err;
	}
	ctx->r_ctx.channel = rdma_create_event_channel();
	if (!ctx->r_ctx.channel) {
		err = -ENODEV;
		goto err;
	}
	ctx->access_flags = IBV_ACCESS_LOCAL_WRITE;
	if (ctx->odp)
		ctx->access_flags |= IBV_ACCESS_ON_DEMAND;

	if (ctx->server) {
		err = setup_server(ctx);
		if (err)
			goto err;
	} else {
		err = setup_client(ctx);
		if (err)
			goto err;
	}

	fprintf(stdout,"Configuration\n");
	fprintf(stdout,"size = ");
	print_size(ctx->size);
	fprintf(stdout,"\n");
	fprintf(stdout,"align = ");
	print_size(ctx->align);
	fprintf(stdout,"\n");
	fprintf(stdout,"count = ");
	print_size(ctx->count);
	fprintf(stdout,"\n");
	fprintf(stdout,"hugetlb = %s\n", ctx->huge ? "enabled" : "disabled");
	fprintf(stdout,"odp = %s\n", ctx->odp ? "enabled" : "disabled");
err:
	return err;
}

static void cleanup_test(struct run_ctx *ctx)
{
	if (ctx)
		free_mem(ctx);
}

static struct run_ctx *ctx;

static void print_client_stats(void)
{
	fprintf(stdout,"send cnt = %d\n", ctx->r_ctx.c_ctx.q->tx.send_cnt);
	ts_print_lat_stats(&ctx->r_ctx.c_ctx.q->tx.slat, "slat");
	ts_print_lat_stats(&ctx->r_ctx.c_ctx.q->tx.clat, "clat");
}

static void sig_int_handler(void)
{
	if (ctx && !ctx->server)
		print_client_stats();
}

void sig_handler(int signo)
{
	switch (signo) {
	case SIGINT:
		sig_int_handler();
		break;
	default:
		break;
	};
	exit(1);
}

int main(int argc, char **argv)
{
	sighandler_t sig_err;
	int err;

	setvbuf(stdout, NULL, _IOLBF, 0);

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx) {
		err = -ENOMEM;
		goto err;
	}
	ctx->size = sysconf(_SC_PAGESIZE);
	ctx->page_size = sysconf(_SC_PAGESIZE);
	ctx->align = sysconf(_SC_PAGESIZE);
	ctx->count = 1024;
	ctx->connections = 1;

	parse_options(ctx, argc, argv);

	sig_err = signal(SIGINT, sig_handler);
	if (sig_err == SIG_ERR)
		return -1;

	err = setup_test(ctx);
	if (err)
		goto err;

	cleanup_test(ctx);

	print_client_stats();
	return 0;

err:
	cleanup_test(ctx);
	return err;
}
