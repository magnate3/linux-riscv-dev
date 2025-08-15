/* Use of this source code is governed by the Apache 2.0 license
 *
 * Originally based upon the linux kernel samples/bpf/xdpsock_user.c code:
 * Copyright(c) 2017 - 2018 Intel Corporation.
 */
#if 0
#include <assert.h>
#include <errno.h>
#include <libgen.h>
#include <linux/bpf.h>
#include <linux/if_link.h>
#include <linux/if_xdp.h>
#include <linux/if_ether.h>
#include <net/if.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <net/ethernet.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>
#include <poll.h>

#include "bpf/libbpf.h"
#include <bpf/bpf.h>


#ifndef SOL_XDP
#define SOL_XDP 283
#endif

#ifndef AF_XDP
#define AF_XDP 44
#endif

#ifndef PF_XDP
#define PF_XDP AF_XDP
#endif

#define NUM_FRAMES 131072
#define FRAME_HEADROOM 0
#define FRAME_SHIFT 11
#define FRAME_SIZE 2048
#define NUM_DESCS 1024
#define BATCH_SIZE 1

#define FQ_NUM_DESCS 1024
#define CQ_NUM_DESCS 1024

#define DEBUG_HEXDUMP 1
typedef __u64 u64;
typedef __u32 u32;

#else
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <libgen.h>
#include <linux/bpf.h>
#include <linux/if_link.h>
#include <linux/if_xdp.h>
#include <linux/if_ether.h>
#include <net/if.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <net/ethernet.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <locale.h>
#include <sys/types.h>
#include <poll.h>
#include "bpf/libbpf.h"
#include <linux/icmp.h>
#include <bpf/bpf.h>
/* Power-of-2 number of sockets */
/* Round-robin receive */
#define RR_LB 0
#ifndef SOL_XDP
#define SOL_XDP 283
#endif
#ifndef AF_XDP
#define AF_XDP 44
#endif
#ifndef PF_XDP
#define PF_XDP AF_XDP
#endif
#define NUM_FRAMES 131072
#define FRAME_HEADROOM 0
#define FRAME_SHIFT 11
#define FRAME_SIZE 2048
#define NUM_DESCS 1024
#define BATCH_SIZE 16
#define FQ_NUM_DESCS 1024
#define CQ_NUM_DESCS 1024
#define DEBUG_HEXDUMP 0
typedef __u64 u64;
typedef __u32 u32;

#define MAX_SOCKS 2
static const char *opt_if = "";
int opt_ifindex = 0;
static int opt_shared_packet_buffer;
#endif
static u32 opt_xdp_flags = 0;
static int opt_interval = 1;
static int opt_queue = 0;
static u32 opt_xdp_bind_flags = 0;
static struct bpf_object *obj;
static int prog_id=-1;
struct xdp_umem_uqueue {
	u32 cached_prod;
	u32 cached_cons;
	u32 mask;
	u32 size;
	u32 *producer;
	u32 *consumer;
	u64 *ring;
	void *map;
};

struct xdp_umem {
	char *frames;
	struct xdp_umem_uqueue fq;
	struct xdp_umem_uqueue cq;
	int fd;
};

struct xdp_uqueue {
	u32 cached_prod;
	u32 cached_cons;
	u32 mask;
	u32 size;
	u32 *producer;
	u32 *consumer;
	struct xdp_desc *ring;
	void *map;
};

struct xdpsock {
	struct xdp_uqueue rx;
	struct xdp_uqueue tx;
	int sfd;
	struct xdp_umem *umem;
	u32 outstanding_tx;
	unsigned long rx_npkts;
	unsigned long tx_npkts;
	unsigned long prev_rx_npkts;
	unsigned long prev_tx_npkts;
};

struct data_val {
    char **data;
    int *sz;
    int numb_packs;
};

#define lassert(expr)							\
	do {								\
		if (!(expr)) {						\
			fprintf(stderr, "%s:%s:%i: Assertion failed: "	\
				#expr ": errno: %d/\"%s\"\n",		\
				__FILE__, __func__, __LINE__,		\
				errno, strerror(errno));		\
			exit(EXIT_FAILURE);				\
		}							\
	} while (0)

#define barrier() __asm__ __volatile__("": : :"memory")
#ifdef __aarch64__
#define u_smp_rmb() __asm__ __volatile__("dmb ishld": : :"memory")
#define u_smp_wmb() __asm__ __volatile__("dmb ishst": : :"memory")
#else
#define u_smp_rmb() barrier()
#define u_smp_wmb() barrier()
#endif

static int do_detach(int prog_id,u32 xdp_flags,int if_index);
static void hex_dump(void *pkt, size_t length, u64 addr)
{
	const unsigned char *address = (unsigned char *)pkt;
	const unsigned char *line = address;
	size_t line_size = 32;
	unsigned char c;
	char buf[32];
	int i = 0;

	sprintf(buf, "addr=%lu", addr);
	printf("length = %zu\n", length);
	printf("%s | ", buf);
	while (length-- > 0) {
		printf("%02X ", *address++);
		if (!(++i % line_size) || (length == 0 && i % line_size)) {
			if (length == 0) {
				while (i++ % line_size)
					printf("__ ");
			}
			printf(" | ");	/* right close */
			while (line < address) {
				c = *line++;
				printf("%c", (c < 33 || c == 255) ? 0x2E : c);
			}
			printf("\n");
			if (length > 0)
				printf("%s | ", buf);
		}
	}
	printf("\n");
}

static inline u32 umem_nb_free(struct xdp_umem_uqueue *q, u32 nb)
{
	u32 free_entries = q->cached_cons - q->cached_prod;

	if (free_entries >= nb)
		return free_entries;

	/* Refresh the local tail pointer */
	q->cached_cons = *q->consumer + q->size;

	return q->cached_cons - q->cached_prod;
}

static inline u32 xq_nb_free(struct xdp_uqueue *q, u32 ndescs)
{
	u32 free_entries = q->cached_cons - q->cached_prod;

	if (free_entries >= ndescs)
		return free_entries;

	/* Refresh the local tail pointer */
	q->cached_cons = *q->consumer + q->size;
	return q->cached_cons - q->cached_prod;
}

static inline u32 umem_nb_avail(struct xdp_umem_uqueue *q, u32 nb)
{
	u32 entries = q->cached_prod - q->cached_cons;

	if (entries == 0) {
		q->cached_prod = *q->producer;
		entries = q->cached_prod - q->cached_cons;
	}

	return (entries > nb) ? nb : entries;
}

static inline u32 xq_nb_avail(struct xdp_uqueue *q, u32 ndescs)
{
	u32 entries = q->cached_prod - q->cached_cons;

	if (entries == 0) {
		q->cached_prod = *q->producer;
		entries = q->cached_prod - q->cached_cons;
	}

	return (entries > ndescs) ? ndescs : entries;
}

static inline int umem_fill_to_kernel_ex(struct xdp_umem_uqueue *fq,
					 struct xdp_desc *d,
					 size_t nb)
{
	u32 i;

	if (umem_nb_free(fq, nb) < nb)
		return -ENOSPC;

	for (i = 0; i < nb; i++) {
		u32 idx = fq->cached_prod++ & fq->mask;

		fq->ring[idx] = d[i].addr;
	}

	u_smp_wmb();

	*fq->producer = fq->cached_prod;

	return 0;
}

static inline int umem_fill_to_kernel(struct xdp_umem_uqueue *fq, u64 *d,
				      size_t nb)
{
	u32 i;

	if (umem_nb_free(fq, nb) < nb)
		return -ENOSPC;

	for (i = 0; i < nb; i++) {
		u32 idx = fq->cached_prod++ & fq->mask;

		fq->ring[idx] = d[i];
	}

	u_smp_wmb();

	*fq->producer = fq->cached_prod;

	return 0;
}

static inline size_t umem_complete_from_kernel(struct xdp_umem_uqueue *cq,
					       u64 *d, size_t nb)
{
	u32 idx, i, entries = umem_nb_avail(cq, nb);

	u_smp_rmb();

	for (i = 0; i < entries; i++) {
		idx = cq->cached_cons++ & cq->mask;
		d[i] = cq->ring[idx];
	}

	if (entries > 0) {
		u_smp_wmb();

		*cq->consumer = cq->cached_cons;
	}

	return entries;
}

static inline void *xq_get_data(struct xdpsock *xsk, u64 addr)
{
	return &xsk->umem->frames[addr];
}

static inline int xq_deq(struct xdp_uqueue *uq,
			 struct xdp_desc *descs,
			 int ndescs)
{
	struct xdp_desc *r = uq->ring;
	unsigned int idx;
	int i, entries;

	entries = xq_nb_avail(uq, ndescs);

	u_smp_rmb();

	for (i = 0; i < entries; i++) {
		idx = uq->cached_cons++ & uq->mask;
		descs[i] = r[idx];
	}

	if (entries > 0) {
		u_smp_wmb();

		*uq->consumer = uq->cached_cons;
	}

	return entries;
}

static struct xdp_umem *xdp_umem_configure(int sfd)
{
	int fq_size = FQ_NUM_DESCS, cq_size = CQ_NUM_DESCS;
	struct xdp_mmap_offsets off;
	struct xdp_umem_reg mr;
	struct xdp_umem *umem;
	socklen_t optlen;
	void *bufs;

	umem = calloc(1, sizeof(*umem));
	lassert(umem);

	lassert(posix_memalign(&bufs, getpagesize(), /* PAGE_SIZE aligned */
			       NUM_FRAMES * FRAME_SIZE) == 0);
	mr.addr = (__u64)bufs;
	mr.len = NUM_FRAMES * FRAME_SIZE;
	mr.chunk_size = FRAME_SIZE;
	mr.headroom = FRAME_HEADROOM;
	printf("setsockopt XDP_UMEM_REG \n");
	lassert(setsockopt(sfd, SOL_XDP, XDP_UMEM_REG, &mr, sizeof(mr)) == 0);
	printf("setsockopt XDP_UMEM_REG over\n");
	lassert(setsockopt(sfd, SOL_XDP, XDP_UMEM_FILL_RING, &fq_size,
			   sizeof(int)) == 0);
	lassert(setsockopt(sfd, SOL_XDP, XDP_UMEM_COMPLETION_RING, &cq_size,
			   sizeof(int)) == 0);

	optlen = sizeof(off);
	lassert(getsockopt(sfd, SOL_XDP, XDP_MMAP_OFFSETS, &off,
			   &optlen) == 0);

	umem->fq.map = mmap(0, off.fr.desc +
			    FQ_NUM_DESCS * sizeof(u64),
			    PROT_READ | PROT_WRITE,
			    MAP_SHARED | MAP_POPULATE, sfd,
			    XDP_UMEM_PGOFF_FILL_RING);
	lassert(umem->fq.map != MAP_FAILED);

	umem->fq.mask = FQ_NUM_DESCS - 1;
	umem->fq.size = FQ_NUM_DESCS;
	umem->fq.producer = umem->fq.map + off.fr.producer;
	umem->fq.consumer = umem->fq.map + off.fr.consumer;
	umem->fq.ring = umem->fq.map + off.fr.desc;
	umem->fq.cached_cons = FQ_NUM_DESCS;

	umem->cq.map = mmap(0, off.cr.desc +
			     CQ_NUM_DESCS * sizeof(u64),
			     PROT_READ | PROT_WRITE,
			     MAP_SHARED | MAP_POPULATE, sfd,
			     XDP_UMEM_PGOFF_COMPLETION_RING);
	lassert(umem->cq.map != MAP_FAILED);

	umem->cq.mask = CQ_NUM_DESCS - 1;
	umem->cq.size = CQ_NUM_DESCS;
	umem->cq.producer = umem->cq.map + off.cr.producer;
	umem->cq.consumer = umem->cq.map + off.cr.consumer;
	umem->cq.ring = umem->cq.map + off.cr.desc;

	umem->frames = bufs;
	umem->fd = sfd;

	return umem;
}

static struct xdpsock *xsk_configure(struct xdp_umem *umem, int opt_ifindex)
{
	struct sockaddr_xdp sxdp = {};
	struct xdp_mmap_offsets off;
	int sfd, ndescs = NUM_DESCS;
	struct xdpsock *xsk;
	bool shared = true;
	socklen_t optlen;
	u64 i;

	sfd = socket(PF_XDP, SOCK_RAW, 0);
	lassert(sfd >= 0);

	xsk = calloc(1, sizeof(*xsk));
	lassert(xsk);

	xsk->sfd = sfd;
	xsk->outstanding_tx = 0;

	printf("%s socket fd %d \n ", __func__,sfd);
	if (!umem) {
		shared = false;
		xsk->umem = xdp_umem_configure(sfd);
	} else {
		xsk->umem = umem;
	}

	lassert(setsockopt(sfd, SOL_XDP, XDP_RX_RING,
			   &ndescs, sizeof(int)) == 0);
	lassert(setsockopt(sfd, SOL_XDP, XDP_TX_RING,
			   &ndescs, sizeof(int)) == 0);
	optlen = sizeof(off);
	lassert(getsockopt(sfd, SOL_XDP, XDP_MMAP_OFFSETS, &off,
			   &optlen) == 0);

	/* Rx */
	xsk->rx.map = mmap(NULL,
			   off.rx.desc +
			   NUM_DESCS * sizeof(struct xdp_desc),
			   PROT_READ | PROT_WRITE,
			   MAP_SHARED | MAP_POPULATE, sfd,
			   XDP_PGOFF_RX_RING);
	lassert(xsk->rx.map != MAP_FAILED);

	if (!shared) {
		for (i = 0; i < NUM_DESCS * FRAME_SIZE; i += FRAME_SIZE)
			lassert(umem_fill_to_kernel(&xsk->umem->fq, &i, 1)
				== 0);
	}

	/* Tx */
	xsk->tx.map = mmap(NULL,
			   off.tx.desc +
			   NUM_DESCS * sizeof(struct xdp_desc),
			   PROT_READ | PROT_WRITE,
			   MAP_SHARED | MAP_POPULATE, sfd,
			   XDP_PGOFF_TX_RING);
	lassert(xsk->tx.map != MAP_FAILED);

	xsk->rx.mask = NUM_DESCS - 1;
	xsk->rx.size = NUM_DESCS;
	xsk->rx.producer = xsk->rx.map + off.rx.producer;
	xsk->rx.consumer = xsk->rx.map + off.rx.consumer;
	xsk->rx.ring = xsk->rx.map + off.rx.desc;

	xsk->tx.mask = NUM_DESCS - 1;
	xsk->tx.size = NUM_DESCS;
	xsk->tx.producer = xsk->tx.map + off.tx.producer;
	xsk->tx.consumer = xsk->tx.map + off.tx.consumer;
	xsk->tx.ring = xsk->tx.map + off.tx.desc;
	xsk->tx.cached_cons = NUM_DESCS;

	sxdp.sxdp_family = PF_XDP;
	sxdp.sxdp_ifindex = opt_ifindex;
	sxdp.sxdp_queue_id = opt_queue;

	if (shared) {
		sxdp.sxdp_flags = XDP_SHARED_UMEM;
		sxdp.sxdp_shared_umem_fd = umem->fd;
	} else {
		sxdp.sxdp_flags = opt_xdp_bind_flags;
	}

	lassert(bind(sfd, (struct sockaddr *)&sxdp, sizeof(sxdp)) == 0);

	return xsk;
}


void close_sock(int opt_ifindex)
{
#if 0
	bpf_set_link_xdp_fd(opt_ifindex, -1, opt_xdp_flags);
#else
        do_detach(prog_id,opt_xdp_flags,opt_ifindex);
#endif
}

static void kick_tx(int fd)
{
	int ret;

	ret = sendto(fd, NULL, 0, MSG_DONTWAIT, NULL, 0);
	if (ret >= 0 || errno == ENOBUFS || errno == EAGAIN || errno == EBUSY)
		return;
	lassert(0);
}

static inline void complete_tx_only(struct xdpsock *xsk)
{
	u64 descs[BATCH_SIZE];
	unsigned int rcvd;

	if (!xsk->outstanding_tx)
		return;

	kick_tx(xsk->sfd);

	rcvd = umem_complete_from_kernel(&xsk->umem->cq, descs, BATCH_SIZE);
	if (rcvd > 0) {
		xsk->outstanding_tx -= rcvd;
		xsk->tx_npkts += rcvd;
	}
}

static inline int xq_enq_tx_only(struct xdpsock *xsk, struct xdp_uqueue *uq,
				 unsigned int id, unsigned int ndescs, char *data, int len)
{
	struct xdp_desc *r = uq->ring;
	unsigned int i;

	if (xq_nb_free(uq, ndescs) < ndescs)
		return -ENOSPC;

	for (i = 0; i < ndescs; i++) {
		u32 idx = uq->cached_prod++ & uq->mask;

		r[idx].addr	= (id + i) << FRAME_SHIFT;
		r[idx].len	= len;

	char *pkt = xq_get_data(xsk, r[idx].addr);
	memcpy(pkt, data, len);
	printf("Writing = %d\n", pkt);
	hex_dump(pkt, r[idx].len, r[idx].addr);
	}

	u_smp_wmb();

	*uq->producer = uq->cached_prod;
	return 0;
}

int write_sock(struct xdpsock *xsk, char *pkt, int l)
{
	static unsigned int idx = NUM_DESCS;

	if (xq_nb_free(&xsk->tx, BATCH_SIZE) >= BATCH_SIZE) {
		lassert(xq_enq_tx_only(xsk, &xsk->tx, idx, BATCH_SIZE, pkt, l) == 0);
		xsk->outstanding_tx += BATCH_SIZE;
		idx += BATCH_SIZE;
		idx %= NUM_FRAMES;
		if (idx == 0) {
			idx = NUM_DESCS;
		}
		printf("idx = %d\n", idx);
	}

	complete_tx_only(xsk); 

    return l;
}


struct data_val* read_sock(struct xdpsock *xsk)
{
	struct xdp_desc descs[BATCH_SIZE];
	struct data_val* dval = malloc(sizeof(struct data_val));
	unsigned int rcvd, i;


	dval->numb_packs = 0;
	rcvd = xq_deq(&xsk->rx, descs, BATCH_SIZE);
	if (!rcvd){ 
		return dval;
	}

	dval->data = malloc(rcvd * sizeof(char*));
	dval->sz = malloc(rcvd * sizeof(int));
	dval->numb_packs = rcvd;
	for (i = 0; i < rcvd; i++) {
		char *pkt = xq_get_data(xsk, descs[i].addr);
		printf("Reading = %d\n", pkt);
		hex_dump(pkt, descs[i].len, descs[i].addr);
	dval->data[i] = malloc(descs[i].len * sizeof(char));
	memcpy(dval->data[i], pkt, descs[i].len);
	dval->sz[i] = descs[i].len;
	
	}

	xsk->rx_npkts += rcvd;
	umem_fill_to_kernel_ex(&xsk->umem->fq, descs, rcvd);

    return dval;
}


static int do_attach(int fd,u32 xdp_flags,int if_index){
    struct bpf_prog_info info = {};
    uint32_t info_len = sizeof(info);
    int prog_id=-1,err = 0;
    err = bpf_xdp_attach(if_index, fd, xdp_flags, NULL);
    if (err < 0) {
        printf("ERROR: failed to attach program to nic\n");
        return prog_id;
    }
    err = bpf_obj_get_info_by_fd(fd, &info, &info_len);
    if (err) {
        printf("can't get prog info - %s\n", strerror(errno));
        return prog_id;
    }
    prog_id = info.id;
    return prog_id;
}
static int forece_detach(u32 xdp_flags,int if_index)
{
	u32 curr_prog_id = 0;
	int err = 0;
	err = bpf_xdp_query_id(if_index, xdp_flags, &curr_prog_id);
	if (err) {
		printf("bpf_xdp_query_id failed\n");
		return err;
	}
	if(curr_prog_id >= 0)
	{
	    err = bpf_xdp_detach(if_index, xdp_flags, NULL);
	    if (err < 0)
            	printf("ERROR: failed to detach prog from nic\n");
	}
	return err;
}
static int do_detach(int prog_id,u32 xdp_flags,int if_index)
{
	u32 curr_prog_id = 0;
	int err = 0;
	err = bpf_xdp_query_id(if_index, xdp_flags, &curr_prog_id);
	if (err) {
		printf("bpf_xdp_query_id failed\n");
		return err;
	}
	if (prog_id == curr_prog_id) {
		err = bpf_xdp_detach(if_index, xdp_flags, NULL);
		if (err < 0)
			printf("ERROR: failed to detach prog from nic\n");
	} else if (!curr_prog_id) {
		printf("couldn't find a prog id on nic\n");
	} else {
		printf("program on interface changed, not removing\n");
	}
	return err;
}
struct xdpsock** get_sock(int opt_ifindex){
	struct xdpsock **xsks = malloc(2*sizeof(struct xdpsock));
#if 0
	struct bpf_prog_load_attr prog_load_attr = {
		.prog_type	= BPF_PROG_TYPE_XDP,
	};
#endif
	int prog_fd, qidconf_map, xsks_map;
	char xdp_filename[256];
	struct bpf_map *map;
	int i, ret, key = 0;
	int num_socks = 0;

	if (!opt_ifindex) {
		fprintf(stderr, "ERROR: interface does not exist\n");
		exit(-1);
	}

	snprintf(xdp_filename, sizeof(xdp_filename), "xdpsock_kern.o");
#if 0
	prog_load_attr.file = xdp_filename;

	if (bpf_prog_load_xattr(&prog_load_attr, &obj, &prog_fd))
		exit(EXIT_FAILURE);
	if (!opt_ifindex) {
		fprintf(stderr, "ERROR: interface does not exist\n");
		exit(-1);
	}
#else

	struct bpf_program *prog;
        obj = bpf_object__open_file(xdp_filename, NULL);
        if (libbpf_get_error(obj)){
            return NULL;
        }
        prog = bpf_object__next_program(obj, NULL);
        bpf_program__set_type(prog, BPF_PROG_TYPE_XDP);
        if(prog){
            const char *sec_name = bpf_program__section_name(prog);
            printf("sec:%s\n",sec_name);
        }
        int err = bpf_object__load(obj);
        if (err){
            bpf_object__close(obj);
            return NULL;
        }
        prog_fd = bpf_program__fd(prog);
#endif
	if (prog_fd < 0) {
		fprintf(stderr, "ERROR: no program found: %s\n",
			strerror(prog_fd));
    		bpf_object__close(obj);
		exit(EXIT_FAILURE);
	}

	map = bpf_object__find_map_by_name(obj, "qidconf_map");
	qidconf_map = bpf_map__fd(map);
	if (qidconf_map < 0) {
		fprintf(stderr, "ERROR: no qidconf map found: %s\n",
			strerror(qidconf_map));
    		bpf_object__close(obj);
		exit(EXIT_FAILURE);
	}

	map = bpf_object__find_map_by_name(obj, "xsks_map");
	xsks_map = bpf_map__fd(map);
	if (xsks_map < 0) {
		fprintf(stderr, "ERROR: no xsks map found: %s\n",
			strerror(xsks_map));
		bpf_object__close(obj);
		exit(EXIT_FAILURE);
	}

#if 0
	if (bpf_set_link_xdp_fd(opt_ifindex, prog_fd, opt_xdp_flags) < 0) {
		fprintf(stderr, "ERROR: link set xdp fd failed\n");
		exit(EXIT_FAILURE);
	}
#else
	//opt_xdp_flags |= XDP_FLAGS_DRV_MODE;
	opt_xdp_flags |= XDP_FLAGS_SKB_MODE;
	opt_xdp_bind_flags |= XDP_COPY;
	prog_id=do_attach(prog_fd,opt_xdp_flags,opt_ifindex);
#endif
	ret = bpf_map_update_elem(qidconf_map, &key, &opt_queue, 0);
	if (ret) {
		fprintf(stderr, "ERROR: bpf_map_update_elem qidconf\n");
		bpf_object__close(obj);
		exit(EXIT_FAILURE);
	}

	printf("xsk_configure sock %d \n",num_socks);
	xsks[num_socks++] = xsk_configure(NULL, opt_ifindex);
	printf("xsk_configure sock %d \n",num_socks);
	for (i = 0; i < 1; i++){
		xsks[num_socks++] = xsk_configure(xsks[0]->umem, opt_ifindex);}

	for (i = 0; i < num_socks; i++) {
		key = i;
		ret = bpf_map_update_elem(xsks_map, &key, &xsks[i]->sfd, 0);
		if (ret) {
			fprintf(stderr, "ERROR: bpf_map_update_elem %d\n", i);
		         bpf_object__close(obj);
			exit(EXIT_FAILURE);
		}
	}

	return xsks;
}

static struct xdp_umem *xdp_umem_configure2(int sfd)
{
	int fq_size = FQ_NUM_DESCS, cq_size = CQ_NUM_DESCS;
	struct xdp_mmap_offsets off;
	struct xdp_umem_reg mr;
	struct xdp_umem *umem;
	socklen_t optlen;
	void *bufs;
	umem = calloc(1, sizeof(*umem));
	lassert(umem);
	lassert(posix_memalign(&bufs, getpagesize(), /* PAGE_SIZE aligned */
			       NUM_FRAMES * FRAME_SIZE) == 0);
	mr.addr = (__u64)bufs;
	mr.len = NUM_FRAMES * FRAME_SIZE;
	mr.chunk_size = FRAME_SIZE;
	mr.headroom = FRAME_HEADROOM;
	lassert(setsockopt(sfd, SOL_XDP, XDP_UMEM_REG, &mr, sizeof(mr)) == 0);
	lassert(setsockopt(sfd, SOL_XDP, XDP_UMEM_FILL_RING, &fq_size,
			   sizeof(int)) == 0);
	lassert(setsockopt(sfd, SOL_XDP, XDP_UMEM_COMPLETION_RING, &cq_size,
			   sizeof(int)) == 0);
	optlen = sizeof(off);
	lassert(getsockopt(sfd, SOL_XDP, XDP_MMAP_OFFSETS, &off,
			   &optlen) == 0);
	umem->fq.map = mmap(0, off.fr.desc +
			    FQ_NUM_DESCS * sizeof(u64),
			    PROT_READ | PROT_WRITE,
			    MAP_SHARED | MAP_POPULATE, sfd,
			    XDP_UMEM_PGOFF_FILL_RING);
	lassert(umem->fq.map != MAP_FAILED);
	umem->fq.mask = FQ_NUM_DESCS - 1;
	umem->fq.size = FQ_NUM_DESCS;
	umem->fq.producer = umem->fq.map + off.fr.producer;
	umem->fq.consumer = umem->fq.map + off.fr.consumer;
	umem->fq.ring = umem->fq.map + off.fr.desc;
	umem->fq.cached_cons = FQ_NUM_DESCS;
	umem->cq.map = mmap(0, off.cr.desc +
			     CQ_NUM_DESCS * sizeof(u64),
			     PROT_READ | PROT_WRITE,
			     MAP_SHARED | MAP_POPULATE, sfd,
			     XDP_UMEM_PGOFF_COMPLETION_RING);
	lassert(umem->cq.map != MAP_FAILED);
	umem->cq.mask = CQ_NUM_DESCS - 1;
	umem->cq.size = CQ_NUM_DESCS;
	umem->cq.producer = umem->cq.map + off.cr.producer;
	umem->cq.consumer = umem->cq.map + off.cr.consumer;
	umem->cq.ring = umem->cq.map + off.cr.desc;
	umem->frames = bufs;
	umem->fd = sfd;
	return umem;
}
static struct xdpsock *xsk_configure2(struct xdp_umem *umem, int ifindex)
{
	struct sockaddr_xdp sxdp = {};
	struct xdp_mmap_offsets off;
	int sfd, ndescs = NUM_DESCS;
	struct xdpsock *xsk;
	bool shared = true;
	socklen_t optlen;
	u64 i;
	sfd = socket(PF_XDP, SOCK_RAW, 0);
	lassert(sfd >= 0);
	xsk = calloc(1, sizeof(*xsk));
	lassert(xsk);
	xsk->sfd = sfd;
	xsk->outstanding_tx = 0;
	printf("%s socket fd %d \n ", __func__,sfd);
	if (!umem) {
		shared = false;
		xsk->umem = xdp_umem_configure2(sfd);
	} else {
		xsk->umem = umem;
	}
	lassert(setsockopt(sfd, SOL_XDP, XDP_RX_RING,
			   &ndescs, sizeof(int)) == 0);
	lassert(setsockopt(sfd, SOL_XDP, XDP_TX_RING,
			   &ndescs, sizeof(int)) == 0);
	optlen = sizeof(off);
	lassert(getsockopt(sfd, SOL_XDP, XDP_MMAP_OFFSETS, &off,
			   &optlen) == 0);
	/* Rx */
	xsk->rx.map = mmap(NULL,
			   off.rx.desc +
			   NUM_DESCS * sizeof(struct xdp_desc),
			   PROT_READ | PROT_WRITE,
			   MAP_SHARED | MAP_POPULATE, sfd,
			   XDP_PGOFF_RX_RING);
	lassert(xsk->rx.map != MAP_FAILED);
	if (!shared) {
		for (i = 0; i < NUM_DESCS * FRAME_SIZE; i += FRAME_SIZE)
			lassert(umem_fill_to_kernel(&xsk->umem->fq, &i, 1)
				== 0);
	}
	/* Tx */
	xsk->tx.map = mmap(NULL,
			   off.tx.desc +
			   NUM_DESCS * sizeof(struct xdp_desc),
			   PROT_READ | PROT_WRITE,
			   MAP_SHARED | MAP_POPULATE, sfd,
			   XDP_PGOFF_TX_RING);
	lassert(xsk->tx.map != MAP_FAILED);
	xsk->rx.mask = NUM_DESCS - 1;
	xsk->rx.size = NUM_DESCS;
	xsk->rx.producer = xsk->rx.map + off.rx.producer;
	xsk->rx.consumer = xsk->rx.map + off.rx.consumer;
	xsk->rx.ring = xsk->rx.map + off.rx.desc;
	xsk->tx.mask = NUM_DESCS - 1;
	xsk->tx.size = NUM_DESCS;
	xsk->tx.producer = xsk->tx.map + off.tx.producer;
	xsk->tx.consumer = xsk->tx.map + off.tx.consumer;
	xsk->tx.ring = xsk->tx.map + off.tx.desc;
	xsk->tx.cached_cons = NUM_DESCS;
	sxdp.sxdp_family = PF_XDP;
	sxdp.sxdp_ifindex = ifindex;
	sxdp.sxdp_queue_id = opt_queue;
	if (shared) {
		sxdp.sxdp_flags = XDP_SHARED_UMEM;
		sxdp.sxdp_shared_umem_fd = umem->fd;
	} else {
		sxdp.sxdp_flags = opt_xdp_bind_flags;
	}
	lassert(bind(sfd, (struct sockaddr *)&sxdp, sizeof(sxdp)) == 0);
	return xsk;
}
#if 0
int main(int argc, char **argv)
{
	struct xdpsock **xsks;
	struct xdpsock *sock1, *sock2;
	int ifindex = -1;
	char data[] = {1, 1, 1, 1, 1, 1, 4, 1, 3, 2, 18, 93, 8, 6, 0, 1, 8, 0,
			6, 4, 0, 1, 54, 21, -3, 42, -18, -93, -64, -88, 8, 100,
			0, 0, 0, 0, 0, 0, -40, 58, -44, 100};
	int len = 42;
	
	struct rlimit r = {RLIM_INFINITY, RLIM_INFINITY};
	if (argc < 2) {
		printf("Usage:\n\t%s net_iface\n", argv[0]);
		return 0;
	}
	
	if (setrlimit(RLIMIT_MEMLOCK, &r)) {
		fprintf(stderr, "ERROR: setrlimit(RLIMIT_MEMLOCK) \"%s\"\n",
			strerror(errno));
		exit(EXIT_FAILURE);
	}
	ifindex = if_nametoindex(argv[1]);
	
        printf("%s,%d\n",argv[1],ifindex);
	xsks = get_sock(ifindex);
	sock1 = xsks[0];
	sock2 = xsks[1];
	
	write_sock(sock1, data, len);
	read_sock(sock2);

	close_sock(ifindex);
        bpf_object__close(obj);
}
#elif 1
static struct option long_options[] = {
	{"rxdrop", no_argument, 0, 'r'},
	{"txonly", no_argument, 0, 't'},
	{"l2fwd", no_argument, 0, 'l'},
	{"interface", required_argument, 0, 'i'},
	{"queue", required_argument, 0, 'q'},
	{"poll", no_argument, 0, 'p'},
	{"shared-buffer", no_argument, 0, 's'},
	{"xdp-skb", no_argument, 0, 'S'},
	{"xdp-native", no_argument, 0, 'N'},
	{"interval", required_argument, 0, 'n'},
	{"zero-copy", no_argument, 0, 'z'},
	{"copy", no_argument, 0, 'c'},
	{0, 0, 0, 0}
};
static void usage(const char *prog)
{
	const char *str =
		"  Usage: %s [OPTIONS]\n"
		"  Options:\n"
		"  -r, --rxdrop		Discard all incoming packets (default)\n"
		"  -t, --txonly		Only send packets\n"
		"  -l, --l2fwd		MAC swap L2 forwarding\n"
		"  -i, --interface=n	Run on interface n\n"
		"  -q, --queue=n	Use queue n (default 0)\n"
		"  -p, --poll		Use poll syscall\n"
		"  -s, --shared-buffer	Use shared packet buffer\n"
		"  -S, --xdp-skb=n	Use XDP skb-mod\n"
		"  -N, --xdp-native=n	Enfore XDP native mode\n"
		"  -n, --interval=n	Specify statistics update interval (default 1 sec).\n"
		"  -z, --zero-copy      Force zero-copy mode.\n"
		"  -c, --copy           Force copy mode.\n"
		"\n";
	fprintf(stderr, str, prog);
	exit(EXIT_FAILURE);
}
static void parse_command_line(int argc, char **argv)
{
	int option_index, c;
	opterr = 0;
	for (;;) {
		c = getopt_long(argc, argv, "rtlfi:q:psSNn:cz", long_options,
				&option_index);
		if (c == -1)
			break;
		switch (c) {
			break;
		case 'i':
			opt_if = optarg;
			break;
		case 'q':
			opt_queue = atoi(optarg);
			break;
		case 's':
			opt_shared_packet_buffer = 1;
			break;
		case 'S':
			opt_xdp_flags |= XDP_FLAGS_SKB_MODE;
			opt_xdp_bind_flags |= XDP_COPY;
			break;
		case 'N':
			opt_xdp_flags |= XDP_FLAGS_DRV_MODE;
			break;
		case 'n':
			opt_interval = atoi(optarg);
			break;
		case 'z':
			opt_xdp_bind_flags |= XDP_ZEROCOPY;
			break;
		case 'c':
			opt_xdp_bind_flags |= XDP_COPY;
			break;
		default:
			usage(basename(argv[0]));
		}
	}
	opt_ifindex = if_nametoindex(opt_if);
	if (!opt_ifindex) {
		fprintf(stderr, "ERROR: interface \"%s\" does not exist\n",
			opt_if);
		usage(basename(argv[0]));
	}
}
int main(int argc, char **argv)
{
        struct xdpsock *xsks[MAX_SOCKS];
	struct rlimit r = {RLIM_INFINITY, RLIM_INFINITY};
	int prog_fd, qidconf_map, xsks_map;
	struct bpf_object *obj=NULL;
	char xdp_filename[256];
	struct bpf_map *map;
	int i, ret, key = 0;
	int num_socks = 0;
	struct xdpsock *sock1=NULL, *sock2=NULL;
	int prog_id=-1;
	char data[] = {1, 1, 1, 1, 1, 1, 4, 1, 3, 2, 18, 93, 8, 6, 0, 1, 8, 0,
			6, 4, 0, 1, 54, 21, -3, 42, -18, -93, -64, -88, 8, 100,
			0, 0, 0, 0, 0, 0, -40, 58, -44, 100};
	int len = 42;
#if 1
	parse_command_line(argc, argv);
#else
	opt_xdp_flags |= XDP_FLAGS_SKB_MODE;
	opt_xdp_bind_flags |= XDP_COPY;

	if (argc < 2) {
		printf("Usage:\n\t%s net_iface\n", argv[0]);
		return 0;
	}
	
	opt_ifindex = if_nametoindex(argv[1]);
#endif
	snprintf(xdp_filename, sizeof(xdp_filename), "xdpsock_kern.o");
	if (setrlimit(RLIMIT_MEMLOCK, &r)) {
		fprintf(stderr, "ERROR: setrlimit(RLIMIT_MEMLOCK) \"%s\"\n",
			strerror(errno));
		exit(EXIT_FAILURE);
	}
	if(opt_ifindex){
	     struct bpf_program *prog;
	     const char *qidconf_map_name="qidconf_map";
	     const char *xsks_map_name="xsks_map";
             obj = bpf_object__open_file(xdp_filename, NULL);
             if (libbpf_get_error(obj)){
                 return 1;
             }
             prog = bpf_object__next_program(obj, NULL);
             bpf_program__set_type(prog, BPF_PROG_TYPE_XDP);
             if(prog){
                 const char *sec_name = bpf_program__section_name(prog);
                 printf("sec:%s\n",sec_name);
             }
             int err = bpf_object__load(obj);
             if (err){
                 bpf_object__close(obj);
                 return 1;
             }
             prog_fd = bpf_program__fd(prog);
             map =bpf_object__find_map_by_name(obj,qidconf_map_name);
             if(map){
                 printf("finding %s \n",qidconf_map_name);
        }else{
            printf("finding %s failed\n",qidconf_map_name);
            bpf_object__close(obj);
            return 1;
        }
        qidconf_map = bpf_map__fd(map);
        if(qidconf_map < 0){
    		fprintf(stderr, "ERROR: no qidconf map found: %s\n",
    			strerror(qidconf_map));
    		bpf_object__close(obj);
    		return 1;
        }
    	map = bpf_object__find_map_by_name(obj,xsks_map_name);
    	xsks_map = bpf_map__fd(map);
    	if (xsks_map < 0) {
    		fprintf(stderr, "ERROR: no xsks map found: %s\n",
    			strerror(xsks_map));
    		bpf_object__close(obj);
    		return 1;
    	}
	}else{
		printf("nic does not exist\n");
        bpf_object__close(obj);
        return 1;
	}
        printf("%s,%d\n",opt_if,opt_ifindex);
	prog_id=do_attach(prog_fd,opt_xdp_flags,opt_ifindex);
	ret = bpf_map_update_elem(qidconf_map, &key, &opt_queue, 0);
	if (ret) {
		fprintf(stderr, "ERROR: bpf_map_update_elem qidconf\n");
		bpf_object__close(obj);
		exit(EXIT_FAILURE);
	}
	xsks[num_socks++] = xsk_configure2(NULL,opt_ifindex);
	printf("configure second socket \n");
#if 1
	for (i = 0; i < MAX_SOCKS - 1; i++)
		xsks[num_socks++] = xsk_configure2(xsks[0]->umem,opt_ifindex);
#endif
	// ...and insert them into the map. 
	for (i = 0; i < num_socks; i++) {
		key = i;
		ret = bpf_map_update_elem(xsks_map, &key, &xsks[i]->sfd, 0);
		if (ret) {
			fprintf(stderr, "ERROR: bpf_map_update_elem %d\n", i);
			bpf_object__close(obj);
			goto out1;
		}
	}

#if 1	
	sock1 = xsks[0];
	sock2 = xsks[1];
	
	write_sock(sock1, data, len);
	read_sock(sock2);
#endif
out1:
	if(sock1)
	    close(sock1->sfd);
	if(sock2)
	    close(sock2->sfd);
        do_detach(prog_id,opt_xdp_flags,opt_ifindex);
        bpf_object__close(obj);
	return 0;
}
#endif
