/* SPDX-License-Identifier: GPL-2.0 */
//#define _POSIX_C_SOURCE 199309L

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <locale.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
//#define _POSIX_C_SOURCE 200809L
#include <unistd.h>
#include<stdlib.h>

#include <sys/resource.h>

#include <bpf/bpf.h>
#include <xdp/xsk.h>
#include <xdp/libxdp.h>

#include <arpa/inet.h>
#include <net/if.h>
#include <linux/if_link.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#define __FAVOR_BSD
#include <netinet/tcp.h>
#undef __FAVOR_BSD
//#include <linux/tcp.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/tcp.h>  // for TCP header structures and constants


#include "../common/common_params.h"
#include "../common/common_user_bpf_xdp.h"
#include "../common/common_libbpf.h"

#define NUM_FRAMES         4096
#define FRAME_SIZE         XSK_UMEM__DEFAULT_FRAME_SIZE
#define RX_BATCH_SIZE      64
#define INVALID_UMEM_FRAME UINT64_MAX

static struct xdp_program *prog;
int xsk_map_fd;
bool custom_xsk = false;
struct config cfg = {
	.ifindex   = -1,
};

struct xsk_umem_info {
	struct xsk_ring_prod fq;
	struct xsk_ring_cons cq;
	struct xsk_umem *umem;
	void *buffer;
};
struct stats_record {
	uint64_t timestamp;
	uint64_t rx_packets;
	uint64_t rx_bytes;
	uint64_t tx_packets;
	uint64_t tx_bytes;
};
struct xsk_socket_info {
	struct xsk_ring_cons rx;
	struct xsk_ring_prod tx;
	struct xsk_umem_info *umem;
	struct xsk_socket *xsk;

	uint64_t umem_frame_addr[NUM_FRAMES];
	uint32_t umem_frame_free;

	uint32_t outstanding_tx;

	struct stats_record stats;
	struct stats_record prev_stats;
};

static inline __u32 xsk_ring_prod__free(struct xsk_ring_prod *r)
{
	r->cached_cons = *r->consumer + r->size;
	return r->cached_cons - r->cached_prod;
}

static const char *__doc__ = "AF_XDP kernel bypass example\n";

static const struct option_wrapper long_options[] = {

	{{"help",	 no_argument,		NULL, 'h' },
	 "Show help", false},

	{{"dev",	 required_argument,	NULL, 'd' },
	 "Operate on device <ifname>", "<ifname>", true},

	{{"skb-mode",	 no_argument,		NULL, 'S' },
	 "Install XDP program in SKB (AKA generic) mode"},

	{{"native-mode", no_argument,		NULL, 'N' },
	 "Install XDP program in native mode"},

	{{"auto-mode",	 no_argument,		NULL, 'A' },
	 "Auto-detect SKB or native mode"},

	{{"force",	 no_argument,		NULL, 'F' },
	 "Force install, replacing existing program on interface"},

	{{"copy",        no_argument,		NULL, 'c' },
	 "Force copy mode"},

	{{"zero-copy",	 no_argument,		NULL, 'z' },
	 "Force zero-copy mode"},

	{{"queue",	 required_argument,	NULL, 'Q' },
	 "Configure interface receive queue for AF_XDP, default=0"},

	{{"poll-mode",	 no_argument,		NULL, 'p' },
	 "Use the poll() API waiting for packets to arrive"},

	{{"quiet",	 no_argument,		NULL, 'q' },
	 "Quiet mode (no output)"},

	{{"filename",    required_argument,	NULL,  1  },
	 "Load program from <file>", "<file>"},

	{{"progname",	 required_argument,	NULL,  2  },
	 "Load program from function <name> in the ELF file", "<name>"},

	{{0, 0, NULL,  0 }, NULL, false}
};

static bool global_exit;


static struct xsk_umem_info *configure_xsk_umem(void *buffer, uint64_t size)
{
	struct xsk_umem_info *umem;
	int ret;

	umem = calloc(1, sizeof(*umem));
	if (!umem)
		return NULL;

	ret = xsk_umem__create(&umem->umem, buffer, size, &umem->fq, &umem->cq,
			       NULL);
	if (ret) {
		errno = -ret;
		return NULL;
	}

	umem->buffer = buffer;
	return umem;
}

static uint64_t xsk_alloc_umem_frame(struct xsk_socket_info *xsk)
{
	uint64_t frame;
	if (xsk->umem_frame_free == 0)
		return INVALID_UMEM_FRAME;

	frame = xsk->umem_frame_addr[--xsk->umem_frame_free];
	xsk->umem_frame_addr[xsk->umem_frame_free] = INVALID_UMEM_FRAME;
	return frame;
}

static void xsk_free_umem_frame(struct xsk_socket_info *xsk, uint64_t frame)
{
	assert(xsk->umem_frame_free < NUM_FRAMES);

	xsk->umem_frame_addr[xsk->umem_frame_free++] = frame;
}

static uint64_t xsk_umem_free_frames(struct xsk_socket_info *xsk)
{
	return xsk->umem_frame_free;
}

static struct xsk_socket_info *xsk_configure_socket(struct config *cfg,
						    struct xsk_umem_info *umem)
{
	struct xsk_socket_config xsk_cfg;
	struct xsk_socket_info *xsk_info;
	uint32_t idx;
	int i;
	int ret;
	uint32_t prog_id;

	xsk_info = calloc(1, sizeof(*xsk_info));
	if (!xsk_info)
		return NULL;

	xsk_info->umem = umem;
	xsk_cfg.rx_size = XSK_RING_CONS__DEFAULT_NUM_DESCS;
	xsk_cfg.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;
	xsk_cfg.xdp_flags = cfg->xdp_flags;
	xsk_cfg.bind_flags = cfg->xsk_bind_flags;
	xsk_cfg.libbpf_flags = (custom_xsk) ? XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD: 0;
	ret = xsk_socket__create(&xsk_info->xsk, cfg->ifname,
				 cfg->xsk_if_queue, umem->umem, &xsk_info->rx,
				 &xsk_info->tx, &xsk_cfg);
	if (ret)
		goto error_exit;

	if (custom_xsk) {
		ret = xsk_socket__update_xskmap(xsk_info->xsk, xsk_map_fd);
		if (ret)
			goto error_exit;
	} else {
		/* Getting the program ID must be after the xdp_socket__create() call */
		if (bpf_xdp_query_id(cfg->ifindex, cfg->xdp_flags, &prog_id))
			goto error_exit;
	}

	/* Initialize umem frame allocation */
	for (i = 0; i < NUM_FRAMES; i++)
		xsk_info->umem_frame_addr[i] = i * FRAME_SIZE;

	xsk_info->umem_frame_free = NUM_FRAMES;

	/* Stuff the receive path with buffers, we assume we have enough */
	ret = xsk_ring_prod__reserve(&xsk_info->umem->fq,
				     XSK_RING_PROD__DEFAULT_NUM_DESCS,
				     &idx);

	if (ret != XSK_RING_PROD__DEFAULT_NUM_DESCS)
		goto error_exit;

	for (i = 0; i < XSK_RING_PROD__DEFAULT_NUM_DESCS; i ++)
		*xsk_ring_prod__fill_addr(&xsk_info->umem->fq, idx++) =
			xsk_alloc_umem_frame(xsk_info);

	xsk_ring_prod__submit(&xsk_info->umem->fq,
			      XSK_RING_PROD__DEFAULT_NUM_DESCS);

	return xsk_info;

error_exit:
	errno = -ret;
	return NULL;
}

static void complete_tx(struct xsk_socket_info *xsk)
{
	unsigned int completed;
	uint32_t idx_cq;

	if (!xsk->outstanding_tx)
		return;

	sendto(xsk_socket__fd(xsk->xsk), NULL, 0, MSG_DONTWAIT, NULL, 0);

	/* Collect/free completed TX buffers */
	completed = xsk_ring_cons__peek(&xsk->umem->cq,
					XSK_RING_CONS__DEFAULT_NUM_DESCS,
					&idx_cq);

	if (completed > 0) {
		for (int i = 0; i < completed; i++)
			xsk_free_umem_frame(xsk,
					    *xsk_ring_cons__comp_addr(&xsk->umem->cq,
								      idx_cq++));

		xsk_ring_cons__release(&xsk->umem->cq, completed);
		xsk->outstanding_tx -= completed < xsk->outstanding_tx ?
			completed : xsk->outstanding_tx;
	}
}

static inline __sum16 csum16_add(__sum16 csum, __be16 addend)
{
	uint16_t res = (uint16_t)csum;

	res += (__u16)addend;
	return (__sum16)(res + (res < (__u16)addend));
}

static inline __sum16 csum16_sub(__sum16 csum, __be16 addend)
{
	return csum16_add(csum, ~addend);
}

static inline void csum_replace2(__sum16 *sum, __be16 old, __be16 new)
{
	*sum = ~csum16_add(csum16_sub(~(*sum), old), new);
}

// 用户态TCP协议栈的核心函数，仅考虑处理HTTP GET请求
static bool tcp_process(struct xsk_socket_info *xsk, uint8_t *pkt,
                        uint32_t len, struct ethhdr *eth,
                        struct iphdr *ip, struct tcphdr *tcp)
{
	/*
	简单假设
	1. 仅处理HTTP GET请求
	2. 握手三次后，建立连接
	3. 客户端发送HTTP GET请求，服务器回复HTTP响应
	4. 四次挥手后，终止连接
	*/ 
	uint32_t idx = 0;
	
	// 获取TCP数据包的有效载荷
	uint8_t *payload = pkt + sizeof(*eth) + sizeof(*ip) + sizeof(*tcp);
	uint32_t payload_len = ntohs(ip->tot_len) - sizeof(*ip) - sizeof(*tcp);

	// 判断是否为TCP连接建立阶段
	if (tcp->syn && !tcp->ack) {
		// 连接建立，发送 SYN+ACK 数据包
		// 构建 SYN+ACK 数据包头部
		struct tcphdr syn_ack;
		memset(&syn_ack, 0, sizeof(syn_ack));
		syn_ack.source = tcp->dest;
		syn_ack.dest = tcp->source;
		syn_ack.seq = tcp->ack_seq;
		syn_ack.ack_seq = htonl(ntohl(tcp->seq) + 1);
		syn_ack.doff = sizeof(syn_ack) / 4;
		syn_ack.syn = 1;
		syn_ack.ack = 1;
		syn_ack.window = htons(65535);
		syn_ack.check = 0;
		syn_ack.urg_ptr = 0;

		// 发送 SYN+ACK 数据包
		uint32_t ret = xsk_ring_prod__reserve(&xsk->tx, 1, &idx);
		if (ret != 1) {
			return false;
		}
		uint8_t *ack_pkt = pkt + sizeof(*eth) + sizeof(*ip);
		memcpy(ack_pkt, &syn_ack, sizeof(syn_ack));
		*xsk_ring_prod__fill_addr(&xsk->tx, idx) = (__u64)ack_pkt;
		xsk_ring_prod__submit(&xsk->tx, 1);
		xsk->outstanding_tx++;
		// 更新统计信息
		xsk->stats.tx_bytes += len;
		xsk->stats.tx_packets++;

		// 连接建立成功
		return true;
	}

	// 判断是否为TCP连接终止阶段
	if (tcp->fin && tcp->ack) {
		// 连接终止，发送 FIN+ACK 数据包
		// 构建 FIN+ACK 数据包头部
		struct tcphdr fin_ack;
		memset(&fin_ack, 0, sizeof(fin_ack));
		fin_ack.source = tcp->dest;
		fin_ack.dest = tcp->source;
		fin_ack.seq = tcp->ack_seq;
		fin_ack.ack_seq = htonl(ntohl(tcp->seq) + 1);
		fin_ack.doff = sizeof(fin_ack) / 4;
		fin_ack.fin = 1;
		fin_ack.ack = 1;
		fin_ack.window = htons(65535);
		fin_ack.check = 0;
		fin_ack.urg_ptr = 0;

		// 发送 FIN+ACK 数据包
		uint32_t ret = xsk_ring_prod__reserve(&xsk->tx, 1, &idx);
		if (ret != 1) {
			return false;
		}
		uint8_t *fin_ack_pkt = pkt + sizeof(*eth) + sizeof(*ip);
		memcpy(fin_ack_pkt, &fin_ack, sizeof(fin_ack));
		*xsk_ring_prod__fill_addr(&xsk->tx, idx) = (__u64)fin_ack_pkt;
		xsk_ring_prod__submit(&xsk->tx, 1);
		xsk->outstanding_tx++;
		// 更新统计信息
		xsk->stats.tx_bytes += len;
		xsk->stats.tx_packets++;
		// 连接终止成功
		return true;
	}

	// 判断是否为HTTP GET请求
	if (payload_len >= 4 && !memcmp(payload, "GET", 3)) {
		// HTTP GET 请求，发送 HTTP 响应
		// 生成 HTTP 响应
		char *http_response = "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, world!";
		uint32_t http_response_len = strlen(http_response);

		// 构建 HTTP 响应数据包头部
		struct tcphdr http_ack;
		memset(&http_ack, 0, sizeof(http_ack));
		http_ack.source = tcp->dest;
		http_ack.dest = tcp->source;
		http_ack.seq = tcp->ack_seq;
		http_ack.ack_seq = htonl(ntohl(tcp->seq) + payload_len);
		http_ack.doff = sizeof(http_ack) / 4;
		http_ack.ack = 1;
		http_ack.window = htons(65535);
		http_ack.check = 0;
		http_ack.urg_ptr = 0;

		// 发送 HTTP 响应数据包
		uint32_t ret = xsk_ring_prod__reserve(&xsk->tx, 1, &idx);
		if (ret != 1) {
			return false;
		}
		uint8_t *http_ack_pkt = pkt + sizeof(*eth) + sizeof(*ip);
		memcpy(http_ack_pkt, &http_ack, sizeof(http_ack));
		memcpy(http_ack_pkt + sizeof(http_ack), http_response, http_response_len);
		*xsk_ring_prod__fill_addr(&xsk->tx, idx) = (__u64)http_ack_pkt;
		xsk_ring_prod__submit(&xsk->tx, 1);
		xsk->outstanding_tx++;
		// 更新统计信息
		xsk->stats.tx_bytes += len;
		xsk->stats.tx_packets++;
	}

	// 无需生成响应
	return false;
}


// 用于处理接收到的数据包并生成响应
static bool process_packet(struct xsk_socket_info *xsk,
			   uint64_t addr, uint32_t len)
{
	uint8_t *pkt = xsk_umem__get_data(xsk->umem->buffer, addr);

	 // 根据一些简化的假设处理数据包并生成响应
	int ret;
	uint32_t tx_idx = 0;
	struct ethhdr *eth = (struct ethhdr *) pkt;
	struct iphdr *ip = (struct iphdr *) (eth + 1);
    struct tcphdr *tcp = (struct tcphdr *) (ip + 1);

	// 判断数据包是否满足生成响应的条件
	if (ntohs(eth->h_proto) != ETH_P_IP ||
        len < (sizeof(*eth) + sizeof(*ip) + sizeof(*tcp)) ||
        ip->protocol != IPPROTO_TCP)
        return false;

	// 处理TCP连接
    bool handled = tcp_process(xsk, pkt, len, eth, ip, tcp);

    // 如果TCP连接未被处理，则直接将数据包发送出去
    if (!handled) {
        ret = xsk_ring_prod__reserve(&xsk->tx, 1, &tx_idx);
        if (ret != 1) {
            /* No more transmit slots, drop the packet */
            return false;
        }

        xsk_ring_prod__tx_desc(&xsk->tx, tx_idx)->addr = addr;
        xsk_ring_prod__tx_desc(&xsk->tx, tx_idx)->len = len;
        xsk_ring_prod__submit(&xsk->tx, 1);
        xsk->outstanding_tx++;
    }

	// 更新统计信息
	xsk->stats.tx_bytes += len;
	xsk->stats.tx_packets++;
	return true;
}

static void handle_receive_packets(struct xsk_socket_info *xsk)
{
	unsigned int rcvd, stock_frames, i;
	uint32_t idx_rx = 0, idx_fq = 0;
	int ret;

	rcvd = xsk_ring_cons__peek(&xsk->rx, RX_BATCH_SIZE, &idx_rx);
	if (!rcvd)
		return;

	/* Stuff the ring with as much frames as possible */
	stock_frames = xsk_prod_nb_free(&xsk->umem->fq,
					xsk_umem_free_frames(xsk));

	if (stock_frames > 0) {

		ret = xsk_ring_prod__reserve(&xsk->umem->fq, stock_frames,
					     &idx_fq);

		/* This should not happen, but just in case */
		while (ret != stock_frames)
			ret = xsk_ring_prod__reserve(&xsk->umem->fq, rcvd,
						     &idx_fq);

		for (i = 0; i < stock_frames; i++)
			*xsk_ring_prod__fill_addr(&xsk->umem->fq, idx_fq++) =
				xsk_alloc_umem_frame(xsk);

		xsk_ring_prod__submit(&xsk->umem->fq, stock_frames);
	}

	/* Process received packets */
	for (i = 0; i < rcvd; i++) {
		uint64_t addr = xsk_ring_cons__rx_desc(&xsk->rx, idx_rx)->addr;
		uint32_t len = xsk_ring_cons__rx_desc(&xsk->rx, idx_rx++)->len;

		if (!process_packet(xsk, addr, len))
			xsk_free_umem_frame(xsk, addr);

		xsk->stats.rx_bytes += len;
	}

	xsk_ring_cons__release(&xsk->rx, rcvd);
	xsk->stats.rx_packets += rcvd;

	/* Do we need to wake up the kernel for transmission */
	complete_tx(xsk);
  }

static void rx_and_process(struct config *cfg,
			   struct xsk_socket_info *xsk_socket)
{
	struct pollfd fds[2];
	int ret, nfds = 1;

	memset(fds, 0, sizeof(fds));
	fds[0].fd = xsk_socket__fd(xsk_socket->xsk);
	fds[0].events = POLLIN;

	while(!global_exit) {
		if (cfg->xsk_poll_mode) {
			ret = poll(fds, nfds, -1);
			if (ret <= 0 || ret > 1)
				continue;
		}
		handle_receive_packets(xsk_socket);
	}
}

#define NANOSEC_PER_SEC 1000000000 /* 10^9 */
static uint64_t gettime(void)
{
	struct timespec t;
	int res;

	res = clock_gettime(CLOCK_MONOTONIC, &t);
	if (res < 0) {
		fprintf(stderr, "Error with gettimeofday! (%i)\n", res);
		exit(EXIT_FAIL);
	}
	return (uint64_t) t.tv_sec * NANOSEC_PER_SEC + t.tv_nsec;
}

static double calc_period(struct stats_record *r, struct stats_record *p)
{
	double period_ = 0;
	__u64 period = 0;

	period = r->timestamp - p->timestamp;
	if (period > 0)
		period_ = ((double) period / NANOSEC_PER_SEC);

	return period_;
}

static void stats_print(struct stats_record *stats_rec,
			struct stats_record *stats_prev)
{
	uint64_t packets, bytes;
	double period;
	double pps; /* packets per sec */
	double bps; /* bits per sec */

	char *fmt = "%-12s %'11lld pkts (%'10.0f pps)"
		" %'11lld Kbytes (%'6.0f Mbits/s)"
		" period:%f\n";

	period = calc_period(stats_rec, stats_prev);
	if (period == 0)
		period = 1;

	packets = stats_rec->rx_packets - stats_prev->rx_packets;
	pps     = packets / period;

	bytes   = stats_rec->rx_bytes   - stats_prev->rx_bytes;
	bps     = (bytes * 8) / period / 1000000;

	printf(fmt, "AF_XDP RX:", stats_rec->rx_packets, pps,
	       stats_rec->rx_bytes / 1000 , bps,
	       period);

	packets = stats_rec->tx_packets - stats_prev->tx_packets;
	pps     = packets / period;

	bytes   = stats_rec->tx_bytes   - stats_prev->tx_bytes;
	bps     = (bytes * 8) / period / 1000000;

	printf(fmt, "       TX:", stats_rec->tx_packets, pps,
	       stats_rec->tx_bytes / 1000 , bps,
	       period);

	printf("\n");
}

static void *stats_poll(void *arg)
{
	unsigned int interval = 2;
	struct xsk_socket_info *xsk = arg;
	static struct stats_record previous_stats = { 0 };

	previous_stats.timestamp = gettime();

	/* Trick to pretty printf with thousands separators use %' */
	setlocale(LC_NUMERIC, "en_US");

	while (!global_exit) {
		sleep(interval);
		xsk->stats.timestamp = gettime();
		stats_print(&xsk->stats, &previous_stats);
		previous_stats = xsk->stats;
	}
	return NULL;
}

static void exit_application(int signal)
{
	int err;

	cfg.unload_all = true;
	err = do_unload(&cfg);
	if (err) {
		fprintf(stderr, "Couldn't detach XDP program on iface '%s' : (%d)\n",
			cfg.ifname, err);
	}

	signal = signal;
	global_exit = true;
}

int main(int argc, char **argv)
{
	//变量声明
	int ret;
	void *packet_buffer;
	uint64_t packet_buffer_size;
	DECLARE_LIBBPF_OPTS(bpf_object_open_opts, opts);
	DECLARE_LIBXDP_OPTS(xdp_program_opts, xdp_opts, 0);
	struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};
	struct xsk_umem_info *umem;
	struct xsk_socket_info *xsk_socket;
	pthread_t stats_poll_thread;
	int err;
	char errmsg[1024];

	// 注册全局退出处理函数
	signal(SIGINT, exit_application);

	// 解析命令行参数
	parse_cmdline_args(argc, argv, long_options, &cfg, __doc__);

	/* 检查必需的选项 */
	if (cfg.ifindex == -1) {
		fprintf(stderr, "ERROR: Required option --dev missing\n\n");
		usage(argv[0], __doc__, long_options, (argc == 1));
		return EXIT_FAIL_OPTION;
	}

	/* 加载自定义程序（如果配置了）*/
	if (cfg.filename[0] != 0) {
		struct bpf_map *map;

		custom_xsk = true;
		xdp_opts.open_filename = cfg.filename;
		xdp_opts.prog_name = cfg.progname;
		xdp_opts.opts = &opts;

		if (cfg.progname[0] != 0) {
			xdp_opts.open_filename = cfg.filename;
			xdp_opts.prog_name = cfg.progname;
			xdp_opts.opts = &opts;

			prog = xdp_program__create(&xdp_opts);
		} else {
			prog = xdp_program__open_file(cfg.filename,
						  NULL, &opts);
		}
		// 检查程序加载错误
		err = libxdp_get_error(prog);
		if (err) {
			libxdp_strerror(err, errmsg, sizeof(errmsg));
			fprintf(stderr, "ERR: loading program: %s\n", errmsg);
			return err;
		}
		// 将程序附加到网络接口
		err = xdp_program__attach(prog, cfg.ifindex, cfg.attach_mode, 0);
		if (err) {
			libxdp_strerror(err, errmsg, sizeof(errmsg));
			fprintf(stderr, "Couldn't attach XDP program on iface '%s' : %s (%d)\n",
				cfg.ifname, errmsg, err);
			return err;
		}

		/* 获取 xsks_map */
		map = bpf_object__find_map_by_name(xdp_program__bpf_obj(prog), "xsks_map");
		xsk_map_fd = bpf_map__fd(map);
		if (xsk_map_fd < 0) {
			fprintf(stderr, "ERROR: no xsks map found: %s\n",
				strerror(xsk_map_fd));
			exit(EXIT_FAILURE);
		}
	}

	/* 允许无限制地锁定内存，
	以便可以锁定所有用于数据包缓冲区的内存
	 */
	if (setrlimit(RLIMIT_MEMLOCK, &rlim)) {
		fprintf(stderr, "ERROR: setrlimit(RLIMIT_MEMLOCK) \"%s\"\n",
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* 分配用于数据包缓冲区的内存
	Allocate memory for NUM_FRAMES of the default XDP frame size */
	packet_buffer_size = NUM_FRAMES * FRAME_SIZE;
	if (posix_memalign(&packet_buffer,
			   getpagesize(), /* PAGE_SIZE aligned */
			   packet_buffer_size)) {
		fprintf(stderr, "ERROR: Can't allocate buffer memory \"%s\"\n",
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* 初始化 umem */
	umem = configure_xsk_umem(packet_buffer, packet_buffer_size);
	if (umem == NULL) {
		fprintf(stderr, "ERROR: Can't create umem \"%s\"\n",
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* 配置和打开 AF_XDP（xsk）套接字*/
	xsk_socket = xsk_configure_socket(&cfg, umem);
	if (xsk_socket == NULL) {
		fprintf(stderr, "ERROR: Can't setup AF_XDP socket \"%s\"\n",
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	/*启动统计信息显示线程*/
	if (verbose) {
		ret = pthread_create(&stats_poll_thread, NULL, stats_poll,
				     xsk_socket);
		if (ret) {
			fprintf(stderr, "ERROR: Failed creating statistics thread "
				"\"%s\"\n", strerror(errno));
			exit(EXIT_FAILURE);
		}
	}

	/* 接收和处理数据包 */
	rx_and_process(&cfg, xsk_socket);

	/* 清理资源 */
	xsk_socket__delete(xsk_socket->xsk);
	xsk_umem__delete(umem->umem);

	return EXIT_OK;
}
/*
这段代码的主要功能是创建和配置 AF_XDP（xsk）套接字，并使用该套接字接收和处理数据包。它还包括加载和附加自定义的 XDP 程序，设置内存锁定限制，分配缓冲区内存等。以下是每个部分的简要解释：

- 解析命令行参数并检查必需选项。
- 如果配置了自定义程序，加载并附加该程序到网络接口，并获取 `xsks_map` 的文件描述符。
- 设置内存锁定的限制，允许锁定足够的内存来存储数据包缓冲区。
- 分配数据包缓冲区内存并初始化 `umem`（用户态内存）。
- 配置和打开 AF_XDP（xsk）套接字。
- 如果启用了详细输出，启动统计信息显示线程。
- 接收和处理数据包，执行 `rx_and_process` 函数。
- 清理资源，包括删除套接字和释放内存。
*/
