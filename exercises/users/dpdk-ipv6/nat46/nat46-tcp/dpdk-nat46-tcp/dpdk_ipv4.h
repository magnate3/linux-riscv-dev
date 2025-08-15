#ifndef __DPVS_IPV4_H__
#define __DPVS_IPV4_H__
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_common.h>
#include <rte_ip.h>
#include <stdbool.h>
#define IP_BUF_LEN 16
/* helper functions */
static inline struct rte_ipv4_hdr *ip4_hdr(const struct rte_mbuf *mbuf)
{
    /* can only invoked at L3 */
    return rte_pktmbuf_mtod(mbuf, struct rte_ipv4_hdr *);
}

static inline int ip4_hdrlen(const struct rte_mbuf *mbuf)
{
    return (ip4_hdr(mbuf)->version_ihl & 0xf) << 2;
}

static inline void ip4_send_csum(struct rte_ipv4_hdr *iph)
{
    iph->hdr_checksum = 0;
    iph->hdr_checksum = rte_ipv4_cksum(iph);
}

static inline bool ip4_is_frag(struct rte_ipv4_hdr *iph)
{
    return (iph->fragment_offset
            & htons(RTE_IPV4_HDR_MF_FLAG | RTE_IPV4_HDR_OFFSET_MASK)) != 0;
}

/*
 *  * Process the pseudo-header checksum of an IPv4 header.
 *   *
 *    * Different from "rte_ipv4_phdr_cksum", "ip4_phdr_cksum" allows for ipv4 options.
 *     * The checksum field must be set to 0 by the caller.
 *      *
 *       * @param iph
 *        *   The pointer to the contiguous IPv4 header.
 *         * @param ol_flags
 *          *   The ol_flags of the associated mbuf.
 *           * @return
 *            *   The non-complemented pseudo checksum to set in the L4 header.
 *             */
static inline uint16_t ip4_phdr_cksum(struct rte_ipv4_hdr *iph, uint64_t ol_flags)
{
    uint16_t csum;
    uint16_t total_length = iph->total_length;

    iph->total_length = htons(ntohs(total_length) -
            ((iph->version_ihl & 0xf) << 2) + sizeof(struct rte_ipv4_hdr));
    csum = rte_ipv4_phdr_cksum(iph, ol_flags);

    iph->total_length = total_length;
    return csum;
}

/*
 *  * Process the IPv4 UDP or TCP checksum.
 *   *
 *    * Different from "rte_ipv4_udptcp_cksum", "ip4_udptcp_cksum" allows for ipv4 options.
 *     * The IP and layer 4 checksum must be set to 0 in the packet by the caller.
 *      *
 *       * @param iph
 *        *   The pointer to the contiguous IPv4 header.
 *         * @param l4_hdr
 *          *   The pointer to the beginning of the L4 header.
 *           * @return
 *            *   The complemented checksum to set in the L4 header.
 *             */
static inline uint16_t ip4_udptcp_cksum(struct rte_ipv4_hdr *iph, const void *l4_hdr)
{
    uint16_t csum;
    uint16_t total_length = iph->total_length;

    iph->total_length = htons(ntohs(total_length) -
            ((iph->version_ihl & 0xf) << 2) + sizeof(struct rte_ipv4_hdr));
    csum = rte_ipv4_udptcp_cksum(iph, l4_hdr);

    iph->total_length = total_length;
    return csum;
}

void dpdk_dump_iph(const struct rte_ipv4_hdr *ip_hdr);
#endif /* __DPVS_IPV4_H__ */
