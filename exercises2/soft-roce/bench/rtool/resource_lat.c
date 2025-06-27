/*
 *  rdma_mr lat -- simple memory registration latency measuring tool
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
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef VERSION
#define VERSION "0.1"
#endif

#define _GNU_SOURCE

#include <sys/param.h>
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
#include <hugetlbfs.h>
#include <sys/time.h>
#include <malloc.h>
#include <inttypes.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/capability.h>
#include <sys/types.h>
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <rdma/rdma_cma.h>

#include "options.h"
#include "ioctl.h"
#include "ib_user_ioctl_cmds.h"
#include "ts.h"

static pthread_barrier_t run_barrier;

struct rdma_connection {
	struct rdma_cm_id *cm_id;	/* client id or server child id */
};

struct cm_ctx {
	struct rdma_event_channel	*event_channel;
	pthread_t			event_thread;
	struct rdma_cm_event		*event;
};

struct thread_ctx {
	uint8_t *buf;
	union {
		struct ibv_mr **mr_list;
		struct ibv_pd **pd_list;
		struct ibv_context **uctx_list;
		struct ibv_mw **mw_list;
		struct ibv_xrcd **xrcd_list;
		struct ibv_ah **ah_list;
	} u;
	struct ibv_comp_channel *cq_channel;
	struct ibv_cq **cq_list;
	struct mlx5dv_devx_obj **devx_cq_list;
	uint32_t *devx_cq_handle;
	struct ibv_qp **qp_list;
	struct ibv_wq **wq_list;
	struct ibv_rwq_ind_table **rq_ind_tbl_list;
	struct ibv_flow **flow_list;
	struct rdma_connection *connections;
	struct time_stats alloc_stats;
	struct time_stats free_stats;
	long long issued;
};

struct run_ctx {
	struct ibv_device *device;
	struct ibv_context *context;
	struct ibv_pd *pd;
	struct cm_ctx cm_ctx;
	union ibv_gid local_gid;

	uint64_t size;
	uint64_t mr_size;	/* mr_size and size are same if all MR
				 * register the same pages.
				 * Otherwise size = num_mrs * mr_size.
				 */
	uint64_t min_mr_size; 	/*
				 * this is the min_mr size to test
				 * when testing with different sizes.
				 * starting from PAGE_SIZE to
				 * mr_size.
				 */
	uint64_t max_mr_size;	/* Do MR tests from min_mr_size to
				 * max_mr_size with doubling MR size
				 * on each iteration.
				 */
	uint64_t page_size;
	uint64_t align;
	uint64_t rlimit;
	uint64_t rlimit_set;
	int access_flags;

	int huge;
	int mmap;
	int odp;
	int lock_memory;
	int dedicated_pages;	/* Each MR gets dedicated pages */
	int read_fault;
	int count;		/* resource/operation count */
	int iter;		/* iteration - how many times to operate */
	int threads;
	int write_pattern;
	int drop_ipc_lock_cap;
	int ioctl_destroy;
	int segfault;
	int wait;
	int report_interval;
	int devx;
	char pattern;
	char *ibdev_name;
	char *resource_type;
	uint64_t step_size;	/* every new MR will be at this
				 * step_size from base address.
				 */
};

struct thread_start_info {
	pthread_t thread;
	struct run_ctx *ctx;
	struct thread_ctx t_ctx;
	int created;
};

#define HUGE_PAGE_KPATH "/proc/sys/vm/nr_hugepages"

static void usage(const char *argv0)
{
	printf("Usage:\n");
	printf("%s\n", argv0);
	printf("Options:\n");
	printf("  -d, --ibdev=<ibdev>      use IB device <dev> (default first device found)\n");
	printf("  -s --size=<size>         size of mr in bytes (default 4096)\n");
	printf("  -l --align=<align_size>  align memory allocation to this size\n");
	printf("  -c --count=<count>       number of resources (i.e. MR, PD etc) to alloc/register\n");
	printf("  -r --rlimit=<bytes>      memory resource hard limit in bytes\n");
	printf("  -i --iter=iteration      how many times to iterarate the operation\n");
	printf("  -u --huge                use huge pages\n");
	printf("  -o --odp                 use ODP registration\n");
	printf("  -t --threads=<threads>   Number of parallel threads\n");
	printf("  -a --assign              use dedicated pages for each MR\n");
	printf("  -A --all                 use all MR sizes starting from PAGE_SIZE to size\n");
	printf("  -m --mmap                use mmap for allocation for huge pages\n");
	printf("  -I --interval            seconds between periodic reports\n");
	printf("  -L --lock                lock memory before registration\n");
	printf("  -f --fault               read page fault memory before registration\n");
	printf("  -D --drop_ipc_lock       drop ipc lock capability before registration\n");
	printf("  -O --ioctl               destroy resource using ioctl method\n");
	printf("  -R --resource            resource type (pd, mr, uctx, mw, cq, qp, xrcd, wq, rqit, ah, flow, cmid)\n");
	printf("  -F --segfault            seg fault after registration\n");
	printf("  -x --devx                devx resource creation mode\n");
	printf("  -W --wait                Wait for user signal before resource creation and destroy\n");
	printf("  -h                       display this help message\n");
	printf("  -v                       display program version\n");
}

void version(const char *argv0)
{
	printf("%s %s\n", argv0, VERSION);
}

static void parse_options(struct run_ctx *ctx, int argc, char **argv)
{
	static struct option long_options[] = {
		{ .name = "ibdev",    .has_arg = 1, .val = 'd' },
		{ .name = "size",     .has_arg = 1, .val = 's' },
		{ .name = "align",    .has_arg = 1, .val = 'l' },
		{ .name = "iter",     .has_arg = 1, .val = 'i' },
		{ .name = "pattern",  .has_arg = 1, .val = 'p' },
		{ .name = "threads",  .has_arg = 1, .val = 't' },
		{ .name = "rlimit",   .has_arg = 1, .val = 'r' },
		{ .name = "count",    .has_arg = 1, .val = 'c' },
		{ .name = "interval", .has_arg = 1, .val = 'I' },
		{ .name = "resource", .has_arg = 1, .val = 'R' },
		{ .name = "huge",     .has_arg = 0, .val = 'u' },
		{ .name = "odp",      .has_arg = 0, .val = 'o' },
		{ .name = "assign",   .has_arg = 0, .val = 'a' },
		{ .name = "all",      .has_arg = 0, .val = 'A' },
		{ .name = "lock",     .has_arg = 0, .val = 'L' },
		{ .name = "fault",    .has_arg = 0, .val = 'f' },
		{ .name = "drop_ipc", .has_arg = 0, .val = 'D' },
		{ .name = "ioctl",    .has_arg = 0, .val = 'O' },
		{ .name = "segfault", .has_arg = 0, .val = 'F' },
		{ .name = "devx",     .has_arg = 0, .val = 'x' },
		{ .name = "wait",     .has_arg = 0, .val = 'W' },
		{ .name = "mmap",     .has_arg = 0, .val = 'm' },
		{ .name = NULL }
	};
	int opt;

	if (argc < 2) {
		usage(argv[0]);
		exit(1);
	}

	while ((opt = getopt_long(argc, argv, "hv:d:I:R:p:r:s:t:i:c:l:uoLfDOFWmaAx",
				  long_options, NULL)) != -1) {
		switch (opt) {
		case 'v':
			version(argv[0]);
			exit(0);
		case 'h':
			usage(argv[0]);
			exit(0);
		case 'a':
			ctx->dedicated_pages = 1;
			break;
		case 'd':
			ctx->ibdev_name = malloc(strlen(optarg));
			if (!ctx->ibdev_name) {
				fprintf(stderr, "Couldn't allocate mem.\n");
				exit(1);
			}
			strcpy(ctx->ibdev_name, optarg);
			break;
		case 'R':
			ctx->resource_type = malloc(strlen(optarg));
			if (!ctx->resource_type) {
				fprintf(stderr, "Couldn't allocate mem.\n");
				exit(1);
			}
			strcpy(ctx->resource_type, optarg);
			break;
		case 's':
			ctx->size = parse_size(optarg);
			break;
		case 'A':
			ctx->min_mr_size = sysconf(_SC_PAGESIZE);
			break;
		case 'I':
			ctx->report_interval = parse_int(optarg);
			break;
		case 'l':
			ctx->align = parse_size(optarg);
			break;
		case 'm':
			ctx->mmap = 1;
			break;
		case 'c':
			ctx->count = parse_int(optarg);
			break;
		case 'i':
			ctx->iter = parse_int(optarg);
			break;
		case 'r':
			ctx->rlimit = parse_size(optarg);
			ctx->rlimit_set = 1;
			break;
		case 'p':
			ctx->write_pattern = 1;
			ctx->pattern = *((char*)optarg);
			break;
		case 't':
			ctx->threads = parse_int(optarg);
			break;
		case 'u':
			ctx->huge = 1;
			break;
		case 'o':
			ctx->odp = 1;
			break;
		case 'L':
			ctx->lock_memory = 1;
			break;
		case 'f':
			ctx->read_fault = 1;
			break;
		case 'D':
			ctx->drop_ipc_lock_cap = 1;
			break;
		case 'O':
			ctx->ioctl_destroy = 1;
			break;
		case 'F':
			ctx->segfault = 1;
			break;
		case 'x':
			ctx->devx = 1;
			break;
		case 'W':
			ctx->wait = 1;
			break;
		}
	}
}

static void normalize_sizes(struct run_ctx *ctx)
{
	if ((strcmp(ctx->resource_type, "mr") == 0) ||
	    (strcmp(ctx->resource_type, "mw") == 0)) {
		ctx->mr_size = ctx->size;
		if (ctx->dedicated_pages) {
			ctx->step_size = ctx->size;
			ctx->size = ctx->size * ctx->count;
		} else {
			ctx->step_size = 0;
		}
	}
}

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

static int config_hugetlb_kernel(const struct run_ctx *ctx)
{
	long hpage_size = gethugepagesize();
	uint64_t num_hpages;

	if (hpage_size == 0)
		return -EINVAL;

	num_hpages = (ctx->size * ctx->threads) / hpage_size;
	if (num_hpages == 0)
		num_hpages = 1;

	return config_hugetlb_pages(num_hpages);
}

static int alloc_hugepage_mem(const struct run_ctx *ctx, struct thread_ctx *t_ctx)
{
	int mmap_flags = MAP_ANONYMOUS | MAP_PRIVATE;

	if (ctx->mmap) {
		mmap_flags |= MAP_HUGETLB;
		t_ctx->buf = mmap(0, ctx->size, PROT_WRITE | PROT_READ,
				mmap_flags, 0, 0);
		if (t_ctx->buf == MAP_FAILED) {
			perror("mmap");
			return -ENOMEM;
		}
	} else {
		t_ctx->buf = get_hugepage_region(ctx->size, GHR_STRICT | GHR_COLOR);
		if (!t_ctx->buf) {
			perror("mmap");
			return -ENOMEM;
		}
	}
	return 0;
}

static int alloc_mem(const struct run_ctx *ctx, struct thread_ctx *t_ctx)
{
	int err = 0;

	if (ctx->huge) {
		err = config_hugetlb_kernel(ctx);
		if (err) {
			printf("fail to configure hugetlb\n");
			err = -EINVAL;
			return err;
		}
		err = alloc_hugepage_mem(ctx, t_ctx);
		if (err)
			return err;
	} else {
		t_ctx->buf = memalign(ctx->align, ctx->size);
		if (!t_ctx->buf) {
			fprintf(stderr, "Couldn't allocate work buf.\n");
			err = -ENOMEM;
			return err;
		}
	}

	return err;
}

static void free_mem(const struct run_ctx *ctx, struct thread_ctx *t_ctx)
{
	if (ctx->huge) {
		if (t_ctx->buf)
			free_hugepage_region(t_ctx->buf);
		reset_huge_tlb_pages(0);
	} else {
		if (t_ctx->buf)
			free(t_ctx->buf);
	}
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

static int lock_mem(const struct run_ctx *ctx, struct thread_ctx *t_ctx)
{
	int ret;

	if (ctx->lock_memory)
		ret = mlock(t_ctx->buf, ctx->size);
	return ret;
}

static void read_fault(const struct run_ctx *ctx, struct thread_ctx *t_ctx)
{
	uint8_t dummy_buf[4096];
	uint8_t *read_ptr = t_ctx->buf;
	uint64_t read_size = ctx->size;

	if (!ctx->read_fault)
		return;

	while (read_size) {
		memcpy(&dummy_buf[0], read_ptr,
		       MIN(sizeof(dummy_buf), read_size));
		read_ptr += MIN(sizeof(dummy_buf), read_size);
		read_size -= MIN(sizeof(dummy_buf), read_size);
	}
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

enum resource_id {
	RTYPE_PD,
	RTYPE_MR,
	RTYPE_UCTX,
	RTYPE_MW,
	RTYPE_CQ,
	RTYPE_QP,
	RTYPE_XRCD,
	RTYPE_WQ,
	RTYPE_RQ_IND_TBL,
	RTYPE_AH,
	RTYPE_FLOW,
	RTYPE_CM_ID,
	RTYPE_MAX
};

struct resource_info {
	enum resource_id id;
	const char *name;
};

static const struct resource_info resource_types[] = {
	{ RTYPE_PD, "pd", },
	{ RTYPE_MR, "mr", },
	{ RTYPE_UCTX, "uctx", },
	{ RTYPE_MW, "mw", },
	{ RTYPE_QP, "qp", },
	{ RTYPE_CQ, "cq", },
	{ RTYPE_XRCD, "xrcd", },
	{ RTYPE_WQ, "wq", },
	{ RTYPE_RQ_IND_TBL, "rqit", },
	{ RTYPE_AH, "ah", },
	{ RTYPE_FLOW, "flow", },
	{ RTYPE_CM_ID, "cmid", },
	{ -1, NULL, },
};

static int check_resource_type(char *type)
{
	int err = -EINVAL;
	int i = 0;

	while (resource_types[i].name) {
		if (strcmp(type, resource_types[i].name)) {
			i++;
			continue;
		}
		err = resource_types[i].id;
		break;
	}
	return err;
}

static int alloc_resource_holder(const struct run_ctx *ctx,
				 struct thread_ctx *t)
{
	int type;

	type = check_resource_type(ctx->resource_type);
	if (type == RTYPE_CQ || type == RTYPE_QP ||
	    type == RTYPE_WQ || type == RTYPE_RQ_IND_TBL ||
	    type == RTYPE_FLOW || type == RTYPE_CM_ID) {
		if (ctx->devx) {
			t->devx_cq_list = calloc(ctx->count, sizeof(struct mlx5dv_devx_obj *));
			if (!t->devx_cq_list) {
				fprintf(stderr, "Couldn't allocate devx cq list memory\n");
				return -ENOMEM;
			}
			t->devx_cq_handle = calloc(ctx->count, sizeof(uint32_t));
		} else {
			t->cq_list = calloc(ctx->count, sizeof(struct ibv_cq*));
			if (!t->cq_list) {
				fprintf(stderr, "Couldn't allocate cq list memory\n");
				return -ENOMEM;
			}
		}
	}
	if (type == RTYPE_QP || type == RTYPE_FLOW) {
		t->qp_list = calloc(ctx->count, sizeof(struct ibv_qp*));
		if (!t->qp_list) {
			fprintf(stderr, "Couldn't allocate qp list memory\n");
			return -ENOMEM;
		}
		if (type == RTYPE_FLOW) {
			t->flow_list = calloc(ctx->count, sizeof(struct ibv_flow*));
			if (!t->flow_list) {
				fprintf(stderr, "Couldn't allocate flow list memory\n");
				return -ENOMEM;
			}
		}
	} else if (type == RTYPE_WQ || type == RTYPE_RQ_IND_TBL) {
		t->wq_list = calloc(ctx->count, sizeof(struct ibv_wq*));
		if (!t->wq_list) {
			fprintf(stderr, "Couldn't allocate wq list memory\n");
			return -ENOMEM;
		}
		if (type == RTYPE_RQ_IND_TBL) {
			t->rq_ind_tbl_list =
				calloc(ctx->count,
				       sizeof(struct ibv_rwq_ind_table*));
			if (!t->rq_ind_tbl_list) {
				fprintf(stderr, "Couldn't allocate rq indir tbl\n");
				return -ENOMEM;
			}
			return 0;
		}
		return 0;
	} else if (type == RTYPE_CM_ID) {
		t->connections = calloc(ctx->count, sizeof(struct rdma_connection));
		if (!t->connections) {
			fprintf(stderr, "Couldn't allocate connections\n");
			return -ENOMEM;
		}
	}

	t->u.mr_list = calloc(ctx->count, sizeof(struct ibv_mr*));
	if (!t->u.mr_list) {
		fprintf(stderr, "Couldn't allocate list memory\n");
		return -ENOMEM;
	}
	return 0;
}

static int alloc_uctx(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	int err = 0;

	t->u.uctx_list[i] = ibv_open_device(ctx->device);
	if (!t->u.uctx_list[i]) {
		fprintf(stderr, "alloc pd count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int alloc_pd(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	int err = 0;

	t->u.pd_list[i] = ibv_alloc_pd(ctx->context);
	if (!t->u.pd_list[i]) {
		fprintf(stderr, "alloc pd count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int alloc_mr(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	int err = 0;

	t->u.mr_list[i] =
			ibv_reg_mr(ctx->pd, t->buf + (i * ctx->step_size),
				   ctx->mr_size, ctx->access_flags);
	if (!t->u.mr_list[i]) {
		fprintf(stderr, "Registered MR count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int alloc_mw(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	int err = 0;

	t->u.mw_list[i] = ibv_alloc_mw(ctx->pd, IBV_MW_TYPE_2);
	if (!t->u.mw_list[i]) {
		fprintf(stderr, "Registered MW count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

struct mlx5_ifc_cqc_bits {
	uint8_t status[0x4];
	uint8_t as_notify[0x1];
	uint8_t initiator_src_dct[0x1];
	uint8_t dbr_umem_valid[0x1];
	uint8_t additional_element[0x1];
	uint8_t cqe_sz[0x3];
	uint8_t cc[0x1];
	uint8_t reserved_at_c[0x1];
	uint8_t scqe_break_moderation_en[0x1];
	uint8_t oi[0x1];
	uint8_t cq_period_mode[0x2];
	uint8_t cqe_comp_en[0x1];
	uint8_t mini_cqe_res_format[0x2];
	uint8_t st[0x4];
	uint8_t always_armed_cq[0x1];
	uint8_t element_type[0x3];
	uint8_t reserved_at_1c[0x2];
	uint8_t cqe_compression_layout[0x2];

	uint8_t dbr_umem_id[0x20];

	uint8_t reserved_at_40[0x14];
	uint8_t page_offset[0x6];
	uint8_t reserved_at_5a[0x2];
	uint8_t mini_cqe_res_format_3_2[0x2];
	uint8_t cq_time_stamp_format[0x2];

	uint8_t reserved_at_60[0x3];
	uint8_t log_cq_size[0x5];
	uint8_t uar_page[0x18];

	uint8_t reserved_at_80[0x4];
	uint8_t cq_period[0xc];
	uint8_t cq_max_count[0x10];

	uint8_t c_eqn_or_add_element[0x20];

	uint8_t reserved_at_c0[0x3];
	uint8_t log_page_size[0x5];
	uint8_t reserved_at_c8[0x18];

	uint8_t reserved_at_e0[0x20];

	uint8_t reserved_at_100[0x8];
	uint8_t last_notified_index[0x18];

	uint8_t reserved_at_120[0x8];
	uint8_t last_solicit_index[0x18];

	uint8_t reserved_at_140[0x8];
	uint8_t consumer_counter[0x18];

	uint8_t reserved_at_160[0x8];
	uint8_t producer_counter[0x18];

	uint8_t as_notify_params[0x40];

	uint8_t dbr_addr[0x40];
};

struct mlx5_ifc_create_cq_out_bits {
	uint8_t status[0x8];
	uint8_t reserved_at_8[0x18];

	uint8_t syndrome[0x20];

	uint8_t reserved_at_40[0x8];
	uint8_t cqn[0x18];

	uint8_t reserved_at_60[0x20];
};

struct mlx5_ifc_create_cq_in_bits {
	uint8_t opcode[0x10];
	uint8_t uid[0x10];

	uint8_t reserved_at_20[0x10];
	uint8_t op_mod[0x10];

	uint8_t reserved_at_40[0x8];
	uint8_t input_cqn[0x18];

	uint8_t reserved_at_60[0x20];

	struct mlx5_ifc_cqc_bits cq_context;

	uint8_t e_mtt_pointer_or_cq_umem_offset[0x40];

	uint8_t cq_umem_id[0x20];

	uint8_t cq_umem_valid[0x1];
	uint8_t reserved_at_2e1[0x1f];

	uint8_t reserved_at_300[0x580];

	uint8_t pas[];
};

enum {
	MLX5_CMD_OPCODE_CREATE_CQ = 0x400,

};

static int devx_alloc_cq(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	struct mlx5dv_devx_umem *cq_dbr_umem;
	struct mlx5dv_devx_umem *cq_umem;
	struct mlx5dv_devx_uar *uar;
	struct mlx5dv_devx devx = {0};
	struct mlx5dv_obj obj = {0};
	uint32_t *cq_dbr; /* array of 2 be32 values */
	void *cq_ring;
	void *cqc;

	uint32_t out[DEVX_ST_SZ_DW(create_cq_out)] = {0};
	uint32_t in[DEVX_ST_SZ_DW(create_cq_in)] = {0};
	uint32_t eq_num;
	int err;

	uar = mlx5dv_devx_alloc_uar(ctx->context, MLX5DV_UAR_ALLOC_TYPE_NC);

        /* Initialize CQ */
        cq_ring = memalign(getpagesize(), 4096);

        cq_umem = mlx5dv_devx_umem_reg(ctx->context, cq_ring,
				       4096, IBV_ACCESS_LOCAL_WRITE);
        if (cq_umem == NULL) {
		printf("Failed register CQ ring memory\n");
                err = errno;
                goto err_reg_ring;
        }

        cq_dbr = memalign(64, sizeof(uint64_t));
        memset(cq_dbr, 0, sizeof(uint64_t));

        cq_dbr_umem = mlx5dv_devx_umem_reg(ctx->context, cq_dbr, 64,
					   IBV_ACCESS_LOCAL_WRITE);
        if (cq_dbr_umem == NULL) {
                printf("Failed to register host CQ DBR memory\n");
                err = errno;
                goto err_reg_dbr;
        }

	/* Get an EQ number */
	err = mlx5dv_devx_query_eqn(ctx->context, 0, &eq_num);
	if (err)
		return err;

        DEVX_SET(create_cq_in, in, opcode, MLX5_CMD_OPCODE_CREATE_CQ);
        DEVX_SET64(create_cq_in, in, e_mtt_pointer_or_cq_umem_offset, 0);
        DEVX_SET(create_cq_in, in, cq_umem_id, cq_umem->umem_id);
	DEVX_SET(create_cq_in, in, cq_umem_valid, 1);

        cqc = DEVX_ADDR_OF(create_cq_in, in, cq_context);
        DEVX_SET(cqc, cqc, dbr_umem_id, cq_dbr_umem->umem_id);
        DEVX_SET(cqc, cqc, log_cq_size, 5);
        DEVX_SET(cqc, cqc, uar_page, uar->page_id);
        DEVX_SET(cqc, cqc, c_eqn_or_add_element, eq_num);
        DEVX_SET64(cqc, cqc, dbr_addr, (uintptr_t)cq_dbr);
        DEVX_SET64(cqc, cqc, dbr_addr, (uintptr_t)0);

	t->devx_cq_list[i] =
		mlx5dv_devx_obj_create(ctx->context,
				       in, sizeof(in), out, sizeof(out));
	if (!t->devx_cq_list[i]) {
		printf("fail to create devx cq errno = %d, %s\n",
		       errno, strerror(errno));
		return errno;
	}

	obj.devx.in = t->devx_cq_list[i];
	obj.devx.out = &devx;

	err = mlx5dv_init_obj(&obj, MLX5DV_OBJ_DEVX);
	if (err)
		return err;

	t->devx_cq_handle[i] = devx.handle;
	return 0;

err_reg_dbr:
err_reg_ring:
	free(cq_ring);
	return -ENOMEM;
}

static int ibv_alloc_cq(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	if (!t->cq_channel) {
		t->cq_channel = ibv_create_comp_channel(ctx->context);
		if (!t->cq_channel) {
			printf("%s fail to create cq channel\n", __func__);
			return -ENOMEM;
		}
	}
	t->cq_list[i] = ibv_create_cq(ctx->context, 4, NULL,
				      t->cq_channel, 0);
	if (!t->cq_list[i]) {
		printf("%s fail to create cq\n", __func__);
		return -ENOMEM;
	}
	return 0;
}

static int alloc_cq(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	if (ctx->devx)
		return devx_alloc_cq(ctx, t, i);
	else
		return ibv_alloc_cq(ctx, t, i);
}

static int alloc_qp(const struct run_ctx *ctx, struct thread_ctx *t,
		    int i, int type)
{
	struct ibv_qp_init_attr qp_attr = { 0 };
	int err = 0;

	qp_attr.send_cq = qp_attr.recv_cq = t->cq_list[i];
	qp_attr.cap.max_send_wr = 4;
	qp_attr.cap.max_recv_wr = 4;
	qp_attr.cap.max_send_sge = 1;
	qp_attr.cap.max_recv_sge = 1;
	qp_attr.sq_sig_all = 1;
	if (type == RTYPE_QP)
		qp_attr.qp_type = IBV_QPT_RC;
	else
		qp_attr.qp_type = IBV_QPT_RAW_PACKET;

	t->qp_list[i] = ibv_create_qp(ctx->pd, &qp_attr);
	if (!t->qp_list[i]) {
		fprintf(stderr, "Registered QP count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int alloc_xrcd(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	struct ibv_xrcd_init_attr xrcd_attr = { 0 };
	int err = 0;

	xrcd_attr.oflags = O_CREAT;
	xrcd_attr.fd = -1;	/* we don't care for fd in resource test */
	xrcd_attr.comp_mask = IBV_XRCD_INIT_ATTR_OFLAGS | IBV_XRCD_INIT_ATTR_FD;

	t->u.xrcd_list[i] = ibv_open_xrcd(ctx->context, &xrcd_attr);
	if (!t->u.xrcd_list[i]) {
		fprintf(stderr, "Registered xrcd count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int alloc_wq(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	struct ibv_wq_init_attr wq_attr = { 0 };
	int err = 0;

	wq_attr.wq_context = NULL;
	wq_attr.wq_type = IBV_WQT_RQ;
	wq_attr.max_wr = 1;
	wq_attr.max_sge = 1;
	wq_attr.pd = ctx->pd;
	wq_attr.cq = t->cq_list[i];
	wq_attr.comp_mask = 0;
	wq_attr.create_flags = IBV_WQ_INIT_ATTR_FLAGS;

	t->wq_list[i] = ibv_create_wq(ctx->context, &wq_attr);
	if (!t->wq_list[i]) {
		fprintf(stderr, "Registered WQ count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int alloc_rq_ind_tbl(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	struct ibv_rwq_ind_table_init_attr attr = { 0 };
	int err = 0;

	attr.log_ind_tbl_size = 0;
	attr.ind_tbl = &t->wq_list[i];
	attr.comp_mask = 0;

	t->rq_ind_tbl_list[i] = ibv_create_rwq_ind_table(ctx->context, &attr);
	if (!t->rq_ind_tbl_list[i]) {
		fprintf(stderr, "Registered table count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int alloc_ah(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	struct ibv_ah_attr ah_attr;
	int err = 0;

	memset(&ah_attr, 0, sizeof(ah_attr));
	ah_attr.port_num = 1;
	ah_attr.is_global = 1;
	ah_attr.grh.dgid = ctx->local_gid;

	t->u.ah_list[i] = ibv_create_ah(ctx->pd, &ah_attr);
	if (!t->u.ah_list[i]) {
		fprintf(stderr, "created ah count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int alloc_cm_id(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	int ret;

	ret = rdma_create_id(ctx->cm_ctx.event_channel,
			     &t->connections[i].cm_id, NULL,
			     RDMA_PS_TCP);
	return ret;
}

struct raw_eth_flow_attr {
	struct ibv_flow_attr attr;
	struct ibv_flow_spec_eth spec_eth;
};

static int alloc_flow(struct thread_ctx *t, int i)
{
	struct raw_eth_flow_attr attr;
	int err = 0;

	memset(&attr, 0, sizeof(attr));

	t->flow_list[i] = ibv_create_flow(t->qp_list[i], &attr.attr);
	if (!t->flow_list[i]) {
		fprintf(stderr, "created flow count = %d\n", i);
		err = -ENOMEM;
	}
	return err;
}

static int allocate_resources(const struct run_ctx *ctx, struct thread_ctx *t)
{
	struct ts_time stat = { 0 };
	int err = 0;
	int type;
	int i;

	type = check_resource_type(ctx->resource_type);
	if (type < 0)
		return err;

	for (i = 0; i < ctx->count; i++) {
		ts_log_start_time(&stat);
		switch (type) {
		case RTYPE_UCTX:
			err = alloc_uctx(ctx, t, i);
			break;
		case RTYPE_PD:
			err = alloc_pd(ctx, t, i);
			break;
		case RTYPE_MR:
			err = alloc_mr(ctx, t, i);
			break;
		case RTYPE_MW:
			err = alloc_mw(ctx, t, i);
			break;
		case RTYPE_QP:
			err = alloc_cq(ctx, t, i);
			if (err)
				break;
			err = alloc_qp(ctx, t, i, type);
			break;
		case RTYPE_CQ:
			err = alloc_cq(ctx, t, i);
			break;
		case RTYPE_XRCD:
			err = alloc_xrcd(ctx, t, i);
			break;
		case RTYPE_WQ:
			err = alloc_cq(ctx, t, i);
			if (err)
				break;
			err = alloc_wq(ctx, t, i);
			break;
		case RTYPE_RQ_IND_TBL:
			err = alloc_cq(ctx, t, i);
			if (err)
				break;
			err = alloc_wq(ctx, t, i);
			if (err)
				break;
			err = alloc_rq_ind_tbl(ctx, t, i);
			break;
		case RTYPE_AH:
			err = alloc_ah(ctx, t, i);
			break;
		case RTYPE_FLOW:
			err = alloc_cq(ctx, t, i);
			if (err)
				break;
			err = alloc_qp(ctx, t, i, type);
			if (err)
				break;
			err = alloc_flow(t, i);
			break;
		case RTYPE_CM_ID:
			err = alloc_cq(ctx, t, i);
			if (err)
				break;
			err = alloc_cm_id(ctx, t, i);
			if (err)
				break;
		}
		ts_log_end_time(&stat);
		if (err)
			break;
		ts_update_time_stats(&stat, &t->alloc_stats);
	}
	return err;
}

static void free_uctx(struct thread_ctx *t, int i)
{
	if (t->u.uctx_list[i])
		ibv_close_device(t->u.uctx_list[i]);
}

static void free_pd(struct thread_ctx *t, int i)
{
	if (t->u.pd_list[i])
		ibv_dealloc_pd(t->u.pd_list[i]);
}

static void free_mr(struct thread_ctx *t, int i)
{
	if (t->u.mr_list[i])
		ibv_dereg_mr(t->u.mr_list[i]);
}

static void free_mw(struct thread_ctx *t, int i)
{
	if (t->u.mw_list[i])
		ibv_dealloc_mw(t->u.mw_list[i]);
}

static void free_cq(const struct run_ctx *ctx, struct thread_ctx *t, int i)
{
	printf("sleep started\n");
	sleep(10);
	printf("sleep end\n");

	if (t->cq_list && t->cq_list[i])
		ibv_destroy_cq(t->cq_list[i]);
	else if (t->devx_cq_list && t->devx_cq_list[i]) {
		if (ctx->ioctl_destroy)
			mlx5_ioctl_devx_obj_destroy(ctx->context->cmd_fd, t->devx_cq_handle[i]);
		else
			mlx5dv_devx_obj_destroy(t->devx_cq_list[i]);
	}
}

static void free_qp(struct thread_ctx *t, int i)
{
	if (t->qp_list[i])
		ibv_destroy_qp(t->qp_list[i]);
}

static void free_xrcd(struct thread_ctx *t, int i)
{
	if (t->u.xrcd_list[i])
		ibv_close_xrcd(t->u.xrcd_list[i]);
}

static void free_wq(struct thread_ctx *t, int i)
{
	if (t->wq_list[i])
		ibv_destroy_wq(t->wq_list[i]);
}

static void free_rq_ind_tbl(struct thread_ctx *t, int i)
{
	if (t->rq_ind_tbl_list[i])
		ibv_destroy_rwq_ind_table(t->rq_ind_tbl_list[i]);
}

static void free_ah(struct thread_ctx *t, int i)
{
	if (t->u.ah_list[i])
		ibv_destroy_ah(t->u.ah_list[i]);
}

static void free_cm_id(struct thread_ctx *t, int i)
{
	if (t->connections[i].cm_id)
		rdma_destroy_id(t->connections[i].cm_id);
}

static void free_flow(struct thread_ctx *t, int i)
{
	if (t->flow_list[i])
		ibv_destroy_flow(t->flow_list[i]);
}

static void free_resources_ioctl(const struct run_ctx *ctx, struct thread_ctx *t, int type)
{
	struct ts_time stat = { 0 };
	uint32_t ret_count;
	uint32_t *handles;
	int uverbs_type;
	uint32_t i;
	int ret = 0;
	int fd;

	switch (type) {
	case RTYPE_MR:
		uverbs_type = UVERBS_OBJECT_MR;
		break;
	case RTYPE_PD:
		uverbs_type = UVERBS_OBJECT_PD;
		break;
	case RTYPE_MW:
		uverbs_type = UVERBS_OBJECT_MW;
		break;
	case RTYPE_XRCD:
		uverbs_type = UVERBS_OBJECT_XRCD;
		break;
	case RTYPE_RQ_IND_TBL:
		uverbs_type = UVERBS_OBJECT_RWQ_IND_TBL;
		break;
	case RTYPE_AH:
		uverbs_type = UVERBS_OBJECT_AH;
		break;
	case RTYPE_FLOW:
		uverbs_type = UVERBS_OBJECT_FLOW;
		break;
	default:
		ret = -EINVAL;
		break;
	}
	if (ret) {
		printf("%s unsupported type\n", __func__);
		return;
	}
	fd = ctx->context->cmd_fd;
	ret = rdma_core_get_obj_handles(fd, ctx->count, uverbs_type,
					&handles, &ret_count);
	if (ret) {
		printf("%s fail to get handles\n", __func__);
		return;
	}

	if (ret_count == 0) {
		printf("%s no handles\n", __func__);
		return;
	}

	for (i = 0; i < ret_count; i++) {
		ts_log_start_time(&stat);
		switch (type) {
		case RTYPE_MR:
			ret = rdma_core_destroy_mr_by_handle(fd, handles[i]);
			break;
		case RTYPE_PD:
			ret = rdma_core_destroy_pd_by_handle(fd, handles[i]);
			break;
		case RTYPE_MW:
			ret = rdma_core_destroy_mw_by_handle(fd, handles[i]);
			break;
		case RTYPE_XRCD:
			ret = rdma_core_destroy_xrcd_by_handle(fd, handles[i]);
			break;
		case RTYPE_RQ_IND_TBL:
			ret = rdma_core_destroy_rwq_ind_tbl_by_handle(fd, handles[i]);
			break;
		case RTYPE_AH:
			ret = rdma_core_destroy_ah_by_handle(fd, handles[i]);
			break;
		case RTYPE_FLOW:
			ret = rdma_core_destroy_flow_by_handle(fd, handles[i]);
			break;
		default:
			ret = -EINVAL;
		};
		ts_log_end_time(&stat);
		ts_update_time_stats(&stat, &t->free_stats);
		if (ret) {
			printf("%s i = %d handle = %d ret = %d\n", __func__, i, handles[i], ret);
			break;
		}
	}
}

static void _free_resources(const struct run_ctx *ctx, struct thread_ctx *t, int type)
{
	struct ts_time stat = { 0 };
	int i;

	for (i = 0; i < ctx->count; i++) {
		ts_log_start_time(&stat);
		switch (type) {
		case RTYPE_UCTX:
			free_uctx(t, i);
			break;
		case RTYPE_PD:
			free_pd(t, i);
			break;
		case RTYPE_MR:
			free_mr(t, i);
			break;
		case RTYPE_MW:
			free_mw(t, i);
			break;
		case RTYPE_CQ:
			free_cq(ctx, t, i);
			break;
		case RTYPE_QP:
			free_qp(t, i);
			free_cq(ctx, t, i);
			break;
		case RTYPE_XRCD:
			free_xrcd(t, i);
			break;
		case RTYPE_WQ:
			free_wq(t, i);
			free_cq(ctx, t, i);
			break;
		case RTYPE_RQ_IND_TBL:
			free_rq_ind_tbl(t, i);
			free_wq(t, i);
			free_cq(ctx, t, i);
			break;
		case RTYPE_AH:
			free_ah(t, i);
			break;
		case RTYPE_FLOW:
			free_flow(t, i);
			free_qp(t, i);
			free_cq(ctx, t, i);
			break;
		case RTYPE_CM_ID:
			free_cm_id(t, i);
			free_cq(ctx, t, i);
			break;
		}
		ts_log_end_time(&stat);
		update_min(&stat, &t->free_stats);
		update_max(&stat, &t->free_stats);
		update_avg(&stat, &t->free_stats);
		t->free_stats.total_latency += stat.latency;
		t->free_stats.count++;
	}
}

static void free_resources(const struct run_ctx *ctx, struct thread_ctx *t)
{
	int type;

	type = check_resource_type(ctx->resource_type);
	if (type < 0)
		return;

	if (ctx->ioctl_destroy && !ctx->devx)
		free_resources_ioctl(ctx, t, type);
	else
		_free_resources(ctx, t, type);
}

static int do_thread_init(const struct run_ctx *ctx, struct thread_ctx *t_ctx)
{
	int err;

	err = alloc_resource_holder(ctx, t_ctx);
	if (err) {
		fprintf(stderr, "Couldn't allocate resource holding memory\n");
		goto err;
	}
	ts_init(&t_ctx->alloc_stats);
	ts_init(&t_ctx->free_stats);

	err = alloc_mem(ctx, t_ctx);
	if (err) {
		fprintf(stderr, "Couldn't allocate memory of size %ld\n",
			ctx->size);
		goto err;
	}
	err = lock_mem(ctx, t_ctx);
	if (err) {
		fprintf(stderr, "Couldn't lock memory of size %ld\n",
			ctx->size);
		goto err;
	}

	read_fault(ctx, t_ctx);

	if (ctx->write_pattern)
		memset(t_ctx->buf, ctx->pattern, ctx->size);
err:
	return err;
}

static void do_thread_cleanup(const struct run_ctx *ctx, struct thread_ctx *t_ctx)
{
	free_mem(ctx, t_ctx);
}

static struct rdma_cm_event *cm_wait_for_event(struct run_ctx *ctx)
{
	int ret;

	ret = rdma_get_cm_event(ctx->cm_ctx.event_channel, &ctx->cm_ctx.event);
	if (ret) {
		printf("%s null event.\n", __func__);
		return NULL;
	}

	printf("rdmacm event: %s status = %d id = %p\n",
		rdma_event_str(ctx->cm_ctx.event->event),
		ctx->cm_ctx.event->status, ctx->cm_ctx.event->id);

	#if 0
	if (ctx->cm_ctx.event->event == RDMA_CM_EVENT_CONNECT_REQUEST ||
	    ctx->cm_ctx.event->event == RDMA_CM_EVENT_ESTABLISHED) {
		printf("listen_id = %p, id = %p\n",
			ctx->cm_ctx.event->listen_id, ctx->cm_ctx.event->id);
	}
	#endif
	return ctx->cm_ctx.event;
}

static void* cm_event_handler(void *arg)
{
	struct run_ctx *ctx = arg;
	struct rdma_cm_event *event;

	while (1) {
		event = cm_wait_for_event(ctx);
		rdma_ack_cm_event(event);
	}
	pthread_exit(NULL);
	return NULL;
}

static void cleanup_rdma_cm_event_channel(struct run_ctx *ctx)
{
	if (ctx->cm_ctx.event_thread) {
		pthread_cancel(ctx->cm_ctx.event_thread);
		pthread_join(ctx->cm_ctx.event_thread, NULL);
	}
	if (ctx->cm_ctx.event_channel)
		rdma_destroy_event_channel(ctx->cm_ctx.event_channel);
}

static void cleanup_test(struct run_ctx *ctx)
{
	cleanup_rdma_cm_event_channel(ctx);

	if (ctx->pd)
		ibv_dealloc_pd(ctx->pd);
	if (ctx->context)
		ibv_close_device(ctx->context);
}

static int setup_rdma_cm_event_channel(struct run_ctx *ctx)
{
	int err;

	ctx->cm_ctx.event_channel = rdma_create_event_channel();

	if (!ctx->cm_ctx.event_channel)
		return -ENOMEM;

	err = pthread_create(&ctx->cm_ctx.event_thread,
		             NULL, &cm_event_handler, ctx);
	return err;
}

static int setup_test(struct run_ctx *ctx, struct ibv_device *ib_dev)
{
	int err;

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

	ctx->context = ibv_open_device(ib_dev);
	if (!ctx->context) {
		fprintf(stderr, "Couldn't get context for %s\n",
			ibv_get_device_name(ib_dev));
		goto err;
	}

	ctx->pd = ibv_alloc_pd(ctx->context);
	if (!ctx->pd) {
		fprintf(stderr, "Couldn't allocate PD\n");
		err = -ENOMEM;
		goto err;
	}

	ctx->access_flags = IBV_ACCESS_LOCAL_WRITE;
	if (ctx->odp)
		ctx->access_flags |= IBV_ACCESS_ON_DEMAND;

	ibv_query_gid(ctx->context, 1, 0, &ctx->local_gid);

	ctx->device = ib_dev;

	err = setup_rdma_cm_event_channel(ctx);
err:
	return err;
}

static void print_lat_stats(uint64_t size, struct time_stats *s, char *str)
{
	if (size) {
		printf("size: "); print_size(size); printf(" ");
	}
	ts_print_lat_stats(s, str);
}

static void check_for_segfault(const struct run_ctx *ctx)
{
	if (ctx->segfault)
		*((char*)NULL) = 'a';
}

static void check_for_user_signal(const struct run_ctx *ctx)
{
	char temp;

	if (ctx->wait) {
		printf("Waiting for user input to proceed.\n");
		scanf("%c", &temp);
	}
}

static void dump_global_test_cfg(struct run_ctx *ctx)
{
	printf("Configuration\n");
	printf("pid     = %d\n", getpid());
	printf("align   = "); print_size(ctx->align); printf("\n");
	printf("count   = "); print_int(ctx->count); printf("\n");
	printf("hugetlb = %s\n", ctx->huge ? "enabled" : "disabled");
	printf("odp     = %s\n", ctx->odp ? "enabled" : "disabled");
	printf("mmap    = %s\n", ctx->mmap ? "enabled" : "disabled");
}

static int do_one_test(const struct run_ctx *ctx, struct thread_ctx *t)
{
	int err;

	check_for_user_signal(ctx);

	ts_log_start_time(&t->alloc_stats.total);
	err = allocate_resources(ctx, t);
	ts_log_end_time(&t->alloc_stats.total);
	if (err) {
		fprintf(stderr, "Couldn't register resources\n");
		goto cleanup;
	}

	check_for_user_signal(ctx);
	check_for_segfault(ctx);

	ts_log_start_time(&t->free_stats.total);
	free_resources(ctx, t);
	ts_log_end_time(&t->free_stats.total);
	return 0;

cleanup:
	free_resources(ctx, t);
	return err;
}

static void* do_run(void *arg)
{
	struct thread_start_info *info = arg;
	struct thread_ctx *t_ctx = &info->t_ctx;
	const struct run_ctx *ctx = info->ctx;
	int i = 0;
	int err;

	err = do_thread_init(ctx, t_ctx);
	if (err)
		goto done;

	/* Wait for all threads to finish their allocation */
	pthread_barrier_wait(&run_barrier);
	do {
		err = do_one_test(ctx, t_ctx);
		if (err)
			break;

		t_ctx->issued++;
		i++;
	} while (i < ctx->iter);

	do_thread_cleanup(ctx, t_ctx);
done:
	if (ctx->threads > 1)
		pthread_exit(NULL);
	return NULL;
}

struct print_thread_ctx {
	struct thread_start_info *threads;
	const struct run_ctx *ctx;
	pthread_t thread;
};

static void print_results(const struct run_ctx *ctx, struct thread_start_info *threads)
{
	int i;

	for (i = 0; i < ctx->threads; i++) {
		if (!threads[i].t_ctx.issued)
			continue;

		print_lat_stats(ctx->mr_size, &threads[i].t_ctx.alloc_stats, "alloc");
		print_lat_stats(ctx->mr_size, &threads[i].t_ctx.free_stats,  "free ");
		printf("issued:  "); print_int(threads[i].t_ctx.issued); printf("\n");
	}
}

static void* results_printer(void *arg)
{
	struct print_thread_ctx *print_ctx = arg;
	const struct run_ctx *ctx = print_ctx->ctx;
	struct thread_start_info *threads = print_ctx->threads;

	/* this barrier also ensures that all worker threads are created */
	pthread_barrier_wait(&run_barrier);

	while(1) {
		print_results(ctx, threads);
		sleep(ctx->report_interval);
	}
	pthread_exit(NULL);
	return NULL;
}

static int run_threads(struct run_ctx *ctx, struct thread_start_info *threads)
{
	int err;
	int i;

	for (i = 0; i < ctx->threads; i++) {
		threads[i].ctx = ctx;

		err = pthread_create(&threads[i].thread, NULL,
				     &do_run, &threads[i]);
		if (err)
			break;
		threads[i].created = 1;
	}

	for (i = 0; i < ctx->threads; i++) {
		if (!threads[i].created)
			continue;

		pthread_join(threads[i].thread, NULL);

	}
	return err;
}

static int run_one_instance(struct run_ctx *ctx,
			    struct thread_start_info *threads)
{
	threads[0].ctx = ctx;
	do_run(&threads[0]);
	return 0;
}

static int do_test(struct run_ctx *ctx, struct ibv_device *ib_dev)
{
	struct print_thread_ctx print_ctx = { 0 };
	struct thread_start_info *threads;
	int waiters = ctx->threads;
	int err;

	err = setup_test(ctx, ib_dev);
	if (err)
		goto err;

	threads = calloc(ctx->threads, sizeof(*threads));
	if (!threads) {
		err = -ENOMEM;
		goto err;
	}

	waiters += (ctx->report_interval ? 1 : 0);
	pthread_barrier_init(&run_barrier, NULL, waiters);
	print_ctx.ctx = ctx;
	print_ctx.threads = threads;

	if (ctx->report_interval) {
		err = pthread_create(&print_ctx.thread,
				     NULL, &results_printer, &print_ctx);
		if (err)
			goto thr_err;
	}

	if (ctx->threads > 1)
		err = run_threads(ctx, threads);
	else
		err = run_one_instance(ctx, threads);

	if (ctx->report_interval) {
		pthread_cancel(print_ctx.thread);
		pthread_join(print_ctx.thread, NULL);
	}

	print_results(ctx, threads);
thr_err:
	free(threads);
err:
	pthread_barrier_destroy(&run_barrier);
	cleanup_test(ctx);
	return err;
}

int main(int argc, char **argv)
{
	struct ibv_device **dev_list;
	struct ibv_device *ib_dev;
	struct run_ctx *ctx;
	int err;

	setvbuf(stdout, NULL, _IOLBF, 0);

	dev_list = ibv_get_device_list(NULL);
	if (!dev_list) {
		perror("Failed to get IB devices list");
		return 1;
	}

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx) {
		err = -ENOMEM;
		goto err;
	}
	ctx->size = sysconf(_SC_PAGESIZE);
	ctx->page_size = sysconf(_SC_PAGESIZE);
	ctx->align = sysconf(_SC_PAGESIZE);
	ctx->count = 1;
	ctx->iter = 1;
	ctx->threads = 1;

	parse_options(ctx, argc, argv);

	ctx->max_mr_size = ctx->size;
	if (ctx->min_mr_size == 0)
		ctx->min_mr_size = ctx->max_mr_size;

	if (!ctx->resource_type)
		ctx->resource_type = "mr";
	else {
		err = check_resource_type(ctx->resource_type);
		if (err < 0) {
			fprintf(stderr, "Invalid resource type = %s\n",
				ctx->resource_type);
			goto done;
		}
	}

	if (!ctx->ibdev_name) {
		ib_dev = *dev_list;
		if (!ib_dev) {
			fprintf(stderr, "No IB devices found\n");
			err = -ENODEV;
			goto err;
		}
	} else {
		int i;
		for (i = 0; dev_list[i]; i++) {
			if (!strcmp(ibv_get_device_name(dev_list[i]), ctx->ibdev_name))
				break;
		}
		ib_dev = dev_list[i];
		if (!ib_dev) {
			fprintf(stderr, "IB device %s not found\n", ctx->ibdev_name);
			err = -ENODEV;
			goto err;
		}
	}

	dump_global_test_cfg(ctx);

	while (ctx->min_mr_size <= ctx->max_mr_size) {
		ctx->size = ctx->min_mr_size;
		normalize_sizes(ctx);
		err = do_test(ctx, ib_dev);

		if (ctx->min_mr_size == ctx->max_mr_size)
			break;

		ctx->min_mr_size *= 2;
		/* when mr size is not aligned, still need to calculate for
		 * maximum size.
		 */
		if (ctx->min_mr_size > ctx->max_mr_size)
			ctx->min_mr_size = ctx->max_mr_size;
	}
err:
	ibv_free_device_list(dev_list);
done:
	return err;
}
