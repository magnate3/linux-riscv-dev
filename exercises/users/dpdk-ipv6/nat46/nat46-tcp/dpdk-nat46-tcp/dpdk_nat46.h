#ifndef __DPVS_NAT64_H__
#define __DPVS_NAT64_H__
#include <netinet/ip6.h>
#include <rte_mbuf.h>
#include "./conf/common.h"
#include "dpdk_ipv4.h"
#include "dpdk_ipv6.h"
static inline int mbuf_nat6to4_len(struct rte_mbuf *mbuf)
{
    int offset = sizeof(struct ip6_hdr);
    uint8_t nexthdr = ip6_hdr(mbuf)->ip6_nxt;
    int len;
    /* ip6_skip_exthdr may return -1 */
    offset = ip6_skip_exthdr(mbuf, offset, &nexthdr);
    
    len = mbuf->pkt_len - offset + sizeof(struct rte_ipv4_hdr);

    return len;
}

static inline int mbuf_nat4to6_len(struct rte_mbuf *mbuf)
{
    // not include  sizeof(struct ip6_hdr)
    return (mbuf->pkt_len - ip4_hdrlen(mbuf));
    //return (mbuf->pkt_len - ip4_hdrlen(mbuf) + sizeof(struct ip6_hdr));
}

//int mbuf_6to4(struct rte_mbuf *mbuf,
//              const struct in_addr *saddr,
//              const struct in_addr *daddr);

int mbuf_6to4(struct rte_mbuf *mbuf, const  uint32_t srcIP, const uint32_t dstIP);
int mbuf_4to6(struct rte_mbuf *mbuf,
              const struct in6_addr *saddr,
              const struct in6_addr *daddr);

int prepare_xmit46_out(struct rte_mbuf *mbuf);
int eth_hdr_push_for_ipv4(struct rte_mbuf *mbuf);
struct ip6_hdr * ipv6_hdr_push_common(const struct rte_mbuf *mbuf,  struct rte_mbuf *dst_mbuf, uint16_t plen);
struct  rte_ipv4_hdr* ipv4_hdr_push_common(const struct rte_mbuf *mbuf,  struct rte_mbuf *dst_mbuf, uint16_t plen, uint16_t proto);
int eth_hdr_push(struct rte_mbuf *mbuf);
int process_udp64(struct rte_mbuf *mbuf);
int process_tcp64(struct rte_mbuf *mbuf);
#endif /* __DPVS_NAT64_H__ */
