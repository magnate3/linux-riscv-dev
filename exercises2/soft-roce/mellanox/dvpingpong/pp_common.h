#ifndef _PP_COMMON_H
#define _PP_COMMON_H

#include <errno.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#define DBG(fmt, args...) \
	printf("=MZDBG:%s:%d: " fmt, __func__, __LINE__, ##args)

#define INFO(fmt, args...) \
	printf("=MZINFO:%s:%d: " fmt, __func__, __LINE__, ##args)

#define ERR(fmt, args...) \
	fprintf(stderr, "=MZERR:%s:%d(%d:%s) " fmt, __func__, __LINE__, errno, strerror(errno), ##args)

#define SERVER_PSN 0x1000
#define CLIENT_PSN 0x8000

#define PORT_NUM 1

#define PP_MAX_WR 64		/* Max outstanding send/recv wr */

#define PP_DATA_BUF_LEN ((1 << 20) + 63)

#define PP_MAX_LOG_CQ_SIZE 8	/* 256 cqe */

struct pp_context {
	int port_num;
	struct ibv_port_attr port_attr;
	/* Same qp_cap for all apps, so that they'll have same behavior */
	struct ibv_qp_cap cap;

	struct ibv_device **dev_list;
	struct ibv_context *ibctx;
	struct ibv_pd *pd;

	struct ibv_mr *mr[PP_MAX_WR];
	unsigned char *mrbuf[PP_MAX_WR];
	ssize_t mrbuflen;
};

struct pp_exchange_info {
	uint32_t qpn;
	uint32_t psn;

	union ibv_gid gid;
	uint32_t lid; /* For IB only */

	/* For RDMA read/write */
	void *addr[PP_MAX_WR];
	uint32_t mrkey[PP_MAX_WR];
};

int pp_ctx_init(struct pp_context *pp, const char *ibv_devname,
		int use_vfio, const char *vfio_pci_name);
void pp_ctx_cleanup(struct pp_context *pp);

int pp_exchange_info(struct pp_context *ppc, int my_sgid_idx,
		     int my_qp_num, uint32_t my_psn,
		     struct pp_exchange_info *remote, const char *sip);

static inline void mem_string(unsigned char *p, ssize_t len)
{
	ssize_t i;

	for (i = 0; i < len - 1; i++)
		p[i] = 'A' + i % ('z' - 'A' + 1);

	p[len - 1] = '\0';
}

/* To dump a long string like this: "0: 0BCDEFGHIJKLMNOP ... nopqrstuvwxyABC" */
void dump_msg_short(int index, struct pp_context *ppc);
#endif	/* _PP_COMMON_H */
