/*
 * Copyright (c) 2016-2017  Intel Corporation.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _TCP_MISC_H_
#define _TCP_MISC_H_

#include <rte_tcp.h>
#include <rte_cycles.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * TCP protocols related structures/functions definitions.
 * Main purpose to simplify (and optimise) processing and representation
 * of protocol related data.
 */

#define	TCP_WSCALE_DEFAULT	7
#define	TCP_WSCALE_NONE		0

#define	TCP_TX_HDR_MAX	(sizeof(struct rte_tcp_hdr) + TCP_TX_OPT_LEN_MAX)

/* max header size for normal data+ack packet */
#define	TCP_TX_HDR_DACK	(sizeof(struct rte_tcp_hdr) + TCP_TX_OPT_LEN_TMS)

#define	TCP4_MIN_MSS	536

#define	TCP6_MIN_MSS	1220

/* default MTU, no TCP options. */
#define TCP4_NOP_MSS	\
	(RTE_ETHER_MTU - sizeof(struct rte_ipv4_hdr) - \
	 sizeof(struct rte_tcp_hdr))

#define TCP6_NOP_MSS	\
	(RTE_ETHER_MTU - sizeof(struct rte_ipv6_hdr) - \
	 sizeof(struct rte_tcp_hdr))

/* default MTU, TCP options present */
#define TCP4_OP_MSS	(TCP4_NOP_MSS - TCP_TX_OPT_LEN_MAX)

#define TCP6_OP_MSS	(TCP6_NOP_MSS - TCP_TX_OPT_LEN_MAX)

/* Initial Window Configuration parameter, probably will be configured during
 * the startup in future */
#define TCP_INITIAL_CWND_MAX 14600

/*
 * TCP flags
 */
#define	TCP_FLAG_FIN	0x01
#define	TCP_FLAG_SYN	0x02
#define	TCP_FLAG_RST	0x04
#define	TCP_FLAG_PSH	0x08
#define	TCP_FLAG_ACK	0x10
#define	TCP_FLAG_URG	0x20

/* TCP flags mask. */
#define	TCP_FLAG_MASK	UINT8_MAX

union typflg {
	uint16_t raw;
	struct {
		uint8_t type;  /* TLE_V4/TLE_V6 */
		uint8_t flags; /* TCP header flags */
	};
};


union seqlen {
	uint64_t raw;
	struct {
		uint32_t seq;
		uint32_t len;
	};
};

#define	TCP_DATA_ALIGN	4

#define	TCP_DATA_OFFSET	4

/*
 * recognizable options.
 */
#define	TCP_OPT_KIND_EOL	0x00
#define	TCP_OPT_KIND_NOP	0x01
#define	TCP_OPT_KIND_MSS	0x02
#define	TCP_OPT_KIND_WSC	0x03
#define	TCP_OPT_KIND_TMS	0x08

#define	TCP_OPT_LEN_EOL		0x01
#define	TCP_OPT_LEN_NOP		0x01
#define	TCP_OPT_LEN_MSS		0x04
#define	TCP_OPT_LEN_WSC		0x03
#define	TCP_OPT_LEN_TMS		0x0a

#define	TCP_TX_OPT_LEN_MAX	\
	RTE_ALIGN_CEIL(TCP_OPT_LEN_MSS + TCP_OPT_LEN_WSC + TCP_OPT_LEN_TMS + \
		TCP_OPT_LEN_EOL, TCP_DATA_ALIGN)

/*
 * recomended format for TSOPT from RFC 1323, appendix A:
 *  +--------+--------+--------+--------+
 *  |   NOP  |  NOP   |  TSopt |   10   |
 *  +--------+--------+--------+--------+
 *  |          TSval   timestamp        |
 *  +--------+--------+--------+--------+
 *  |          TSecr   timestamp        |
 *  +--------+--------+--------+--------+
 */
#define	TCP_TX_OPT_LEN_TMS	(TCP_OPT_LEN_TMS + 2 * TCP_OPT_LEN_NOP)

#define TCP_OPT_TMS_HDR		(rte_be_to_cpu_32( \
	TCP_OPT_KIND_NOP << 3 * CHAR_BIT | \
	TCP_OPT_KIND_NOP << 2 * CHAR_BIT | \
	TCP_OPT_KIND_TMS << CHAR_BIT | \
	TCP_OPT_LEN_TMS))

#define	TCP_OPT_KL(k, l)	(rte_be_to_cpu_16((k) << CHAR_BIT | (l)))

#define	TCP_OPT_KL_MSS		TCP_OPT_KL(TCP_OPT_KIND_MSS, TCP_OPT_LEN_MSS)
#define	TCP_OPT_KL_WSC		TCP_OPT_KL(TCP_OPT_KIND_WSC, TCP_OPT_LEN_WSC)
#define	TCP_OPT_KL_TMS		TCP_OPT_KL(TCP_OPT_KIND_TMS, TCP_OPT_LEN_TMS)

/*
 * Timestamp option.
 */
union tsopt {
	uint64_t raw;
	struct {
		uint32_t val;
		uint32_t ecr;
	};
};

struct tcpopt {
	union {
		uint16_t raw;
		struct {
			uint8_t kind;
			uint8_t len;
		};
	} kl;
	union {
		uint16_t mss;
		uint8_t  wscale;
		union tsopt ts;
	};
} __attribute__((__packed__));

struct syn_opts {
	uint16_t mss;
	uint8_t  wscale;
	union tsopt ts;
};

struct resp_info {
	uint32_t flags;
};


/* window update information (RFC 793 WL1, WL2) */
union wui {
	uint64_t raw;
	struct {
		uint32_t wl1;
		uint32_t wl2;
	};
};

/*
 * helper structure: holds aggregated information about group
 * of processed data+ack packets.
 */
struct dack_info {
	struct {                    /* # of received segments with: */
		uint32_t data;      /* incoming data */
		uint32_t ack;       /* newly acked data */
		uint32_t dup;       /* duplicate acks */
		uint32_t badseq;    /* bad seq/ack */
		uint32_t ofo;       /* OFO incoming data */
	} segs;
	uint32_t ack;       /* highest received ACK */
	union tsopt ts;     /* TS of highest ACK */
	union wui wu;       /* window update information */
	uint32_t wnd;
	struct {               /* 3 duplicate ACKs were observed after */
		uint32_t seg;  /* # of meaningful ACK segments */
		uint32_t ack;  /* ACK sequence */
	} dup3;
};

/* get current timestamp in ms */
static inline uint32_t
tcp_get_tms(uint32_t mshift)
{
	uint64_t ts;
	ts = rte_get_tsc_cycles() >> mshift;
	return ts;
}

static inline int
tcp_seq_lt(uint32_t l, uint32_t r)
{
	return (int32_t)(l - r) < 0;
}

static inline int
tcp_seq_leq(uint32_t l, uint32_t r)
{
	return (int32_t)(l - r) <= 0;
}

static inline uint32_t
tcp_seq_min(uint32_t l, uint32_t r)
{
	if (tcp_seq_lt(l, r))
		return l;
	else
		return r;
}


static inline void
get_syn_opts(struct syn_opts *so, uintptr_t p, uint32_t len)
{
	uint32_t i, kind;
	const struct tcpopt *opt;

	memset(so, 0, sizeof(*so));

	i = 0;
	while (i < len) {
		opt = (const struct tcpopt *)(p + i);
		kind = opt->kl.kind;
		if (kind == TCP_OPT_KIND_EOL)
			return;
		else if (kind == TCP_OPT_KIND_NOP)
			i += sizeof(opt->kl.kind);
		else {
			i += opt->kl.len;
			if (i <= len) {
				if (opt->kl.raw == TCP_OPT_KL_MSS)
					so->mss = rte_be_to_cpu_16(opt->mss);
				else if (opt->kl.raw == TCP_OPT_KL_WSC)
					so->wscale = opt->wscale;
				else if (opt->kl.raw == TCP_OPT_KL_TMS) {
					so->ts.val =
						rte_be_to_cpu_32(opt->ts.val);
					so->ts.ecr =
						rte_be_to_cpu_32(opt->ts.ecr);
				}
			}
		}
	}
}

/*
 * generates SYN options, assumes that there are
 * at least TCP_TX_OPT_LEN_MAX bytes available.
 */
static inline void
fill_syn_opts(void *p, const struct syn_opts *so)
{
	uint8_t *to;
	struct tcpopt *opt;

	to = (uint8_t *)p;

	/* setup MSS*/
	opt = (struct tcpopt *)to;
	opt->kl.raw = TCP_OPT_KL_MSS;
	opt->mss = rte_cpu_to_be_16(so->mss);

	to += TCP_OPT_LEN_MSS;
	opt = (struct tcpopt *)to;

	/* setup TMS*/
	if (so->ts.val != 0) {

		opt->kl.raw = TCP_OPT_KL_TMS;
		opt->ts.val = rte_cpu_to_be_32(so->ts.val);
		opt->ts.ecr = rte_cpu_to_be_32(so->ts.ecr);

		to += TCP_OPT_LEN_TMS;
		opt = (struct tcpopt *)to;
	}

	/* setup TMS*/
	if (so->wscale != 0) {

		opt->kl.raw = TCP_OPT_KL_WSC;
		opt->wscale = so->wscale;

		to += TCP_OPT_LEN_WSC;
		opt = (struct tcpopt *)to;
	}

	to[0] = TCP_OPT_KIND_EOL;
}

/*
 * generate TMS option, for non SYN packet, make sure
 * there at least TCP_TX_OPT_LEN_TMS available.
 */
static inline void
fill_tms_opts(void *p, uint32_t val, uint32_t ecr)
{
	uint32_t *opt;

	opt = (uint32_t *)p;
	opt[0] = TCP_OPT_TMS_HDR;
	opt[1] = rte_cpu_to_be_32(val);
	opt[2] = rte_cpu_to_be_32(ecr);
}

static inline union tsopt
get_tms_opts(uintptr_t p, uint32_t len)
{
	union tsopt ts;
	uint32_t i, kind;
	const uint32_t *opt;
	const struct tcpopt *to;

	opt = (const uint32_t *)p;

	/* TS option is presented in recommended way */
	if (len >= TCP_TX_OPT_LEN_TMS && opt[0] == TCP_OPT_TMS_HDR) {
		ts.val = rte_be_to_cpu_32(opt[1]);
		ts.ecr = rte_be_to_cpu_32(opt[2]);
		return ts;
	}

	/* parse through whole list of options. */
	ts.raw = 0;
	i = 0;
	while (i < len) {
		to = (const struct tcpopt *)(p + i);
		kind = to->kl.kind;
		if (kind == TCP_OPT_KIND_EOL)
			break;
		else if (kind == TCP_OPT_KIND_NOP)
			i += sizeof(to->kl.kind);
		else {
			i += to->kl.len;
			if (i <= len && to->kl.raw == TCP_OPT_KL_TMS) {
				ts.val = rte_be_to_cpu_32(to->ts.val);
				ts.ecr = rte_be_to_cpu_32(to->ts.ecr);
				break;
			}
		}
	}

	return ts;
}


static inline uint32_t
tcp_mbuf_seq_free(struct rte_mbuf *mb[], uint32_t num)
{
	uint32_t i, len;

	len = 0;
	for (i = 0; i != num; i++) {
		len += mb[i]->pkt_len;
		rte_pktmbuf_free(mb[i]);
	}

	return len;
}

#ifdef __cplusplus
}
#endif

#endif /* _TCP_MISC_H_ */
