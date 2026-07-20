#ifndef __DPVS_IPV6_H__
#define __DPVS_IPV6_H__

#include <netinet/ip6.h>
#include <rte_ip.h>
#include "rte_mbuf.h"
#include "linux_ipv6.h"
#ifndef DPDK_IPV6_DEBUG
#include "flow.h"
#endif
#define IPV6
#define RTE_LOGTYPE_IPV6    RTE_LOGTYPE_USER1
//#define     IPV6_MTU_DEFAULT        (1500)
#define     IPV6_MTU_DEFAULT        RTE_ETHER_MTU
#define IPV6_VERSION		0x60
#define IPV6_VERSION_MASK	0xf0
int ipv6_reassemble_test(struct rte_mbuf **pkts, int num);
struct rte_mbuf * ipv6_reassemble(struct rte_mbuf *mbuf);
/*
 * helper functions
 */
static inline struct ip6_hdr *ip6_hdr(const struct rte_mbuf *mbuf)
{
    /* can only invoked at L3 */
    return rte_pktmbuf_mtod(mbuf, struct ip6_hdr *);
}

static inline bool ip6_is_frag(struct ip6_hdr *ip6h)
{
    return (ip6h->ip6_nxt == IPPROTO_FRAGMENT);
}

enum {
    INET6_PROTO_F_NONE      = 0x01,
    INET6_PROTO_F_FINAL     = 0x02,
};

/*
 * inet6_protocol:
 * to process IPv6 upper-layer protocol or ext-header.
 *
 * @handler
 * handler protocol, it consume pkt or return next-header.
 *
 * 1. if return > 0, it's always "nexthdr",
 *    no matter if proto is final or not.
 * 2. if return == 0, the pkt is consumed.
 * 3. should not return < 0, or it'll be ignored.
 * 4. mbuf->l3_len must be upadted by handler
 *    to the value as ext-header length.
 *
 * @flags: INET6_PROTO_F_XXX
 */
struct inet6_protocol {
    int             (*handler)(struct rte_mbuf *mbuf);
    unsigned int    flags;
};

int ipv6_init(void);
int ipv6_term(void);
#ifdef DPDK_IPV6_DEBUG
int ipv6_xmit(struct rte_mbuf *mbuf);
#else
int ipv6_xmit(struct rte_mbuf *mbuf, struct flow6 *fl6);
#endif
int ip6_output(struct rte_mbuf *mbuf);

int ip6_local_out(struct rte_mbuf *mbuf);
#ifndef DPDK_IPV6_DEBUG
int ipv6_register_hooks(struct inet_hook_ops *ops, size_t n);
int ipv6_unregister_hooks(struct inet_hook_ops *ops, size_t n);
int ipv6_register_protocol(struct inet6_protocol *prot,
                           unsigned char protocol);
int ipv6_unregister_protocol(struct inet6_protocol *prot,
                             unsigned char protocol);

int ipv6_stats_cpu(struct inet_stats *stats);

void install_ipv6_keywords(void);
void ipv6_keyword_value_init(void);

/* control plane */
int ipv6_ctrl_init(void);
int ipv6_ctrl_term(void);

/* extension header and options. */
int ipv6_exthdrs_init(void);
void ipv6_exthdrs_term(void);
int ipv6_parse_hopopts(struct rte_mbuf *mbuf);
#endif
int ip6_skip_exthdr(const struct rte_mbuf *imbuf, int start,
                    __u8 *nexthdrp);
/* get ipv6 header length, including extension header length. */
int ip6_hdrlen(const struct rte_mbuf *mbuf);

/*
 * Exthdr supported checksum function for upper layer protocol.
 * @param ol_flags
 *    The ol_flags of the associated mbuf.
 * @param exthdrlen
 *    The IPv6 fixed header length plus the extension header length.
 * @param l4_proto
 *    The L4 protocol type, i.e. IPPROTO_TCP, IPPROTO_UDP, IPPROTO_ICMP
 * @return
 *    The non-complemented checksum to set in the L4 header.
 */
uint16_t ip6_phdr_cksum(struct ip6_hdr*, uint64_t ol_flags,
        uint32_t exthdrlen, uint8_t l4_proto);
uint16_t ip6_udptcp_cksum(struct ip6_hdr*, const void *l4_hdr,
        uint32_t exthdrlen, uint8_t l4_proto);

/**
 * Compute the raw (non complemented) checksum of a packet.
 *
 * @param m
 *   The pointer to the mbuf.
 * @param off
 *   The offset in bytes to start the checksum.
 * @param len
 *   The length in bytes of the data to checksum.
 * @param cksum
 *   A pointer to the checksum, filled on success.
 * @return
 *   0 on success, -1 on error (bad length or offset).
 */
static inline int
ipv6_raw_cksum_mbuf(const struct rte_mbuf *m, uint32_t off, uint32_t len,
	uint16_t *cksum)
{
	const struct rte_mbuf *seg;
	const char *buf;
	uint32_t sum, tmp;
	uint32_t seglen, done;

#if 0
	/* easy case: all data in the first segment */
	if (off + len <= rte_pktmbuf_data_len(m)) {
		*cksum = rte_raw_cksum(rte_pktmbuf_mtod_offset(m,
				const char *, off), len);
		return 0;
	}
#endif
	if (unlikely(off + len > rte_pktmbuf_pkt_len(m)))
		return -1; /* invalid params, return a dummy value */

	/* else browse the segment to find offset */
	seglen = 0;
	for (seg = m; seg != NULL; seg = seg->next) {
		seglen = rte_pktmbuf_data_len(seg);
		if (off < seglen)
			break;
		off -= seglen;
	}
	seglen -= off;
	buf = rte_pktmbuf_mtod_offset(seg, const char *, off);
#if 0
	if (seglen >= len) {
		/* all in one segment */
		*cksum = rte_raw_cksum(buf, len);
		return 0;
	}
#endif
	/* hard case: process checksum of several segments */
	sum = 0;
	done = 0;
	for (;;) {
		tmp = __rte_raw_cksum(buf, seglen, 0);
		if (done & 1)
			tmp = rte_bswap16((uint16_t)tmp);
		sum += tmp;
		done += seglen;
		if (done == len)
			break;
		seg = seg->next;
		buf = rte_pktmbuf_mtod(seg, const char *);
		seglen = rte_pktmbuf_data_len(seg);
		if (seglen > len - done)
			seglen = len - done;
	}

	*cksum = __rte_raw_cksum_reduce(sum);
	return 0;
}

#endif /* __DPVS_IPV6_H__ */
