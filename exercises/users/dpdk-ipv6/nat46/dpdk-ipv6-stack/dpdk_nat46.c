#include <netinet/in.h>
#include <netinet/icmp6.h>
#include <netinet/ip6.h>
#include<rte_ip.h>
#include<rte_ether.h>
#include "dpdk_nat46.h"
#include "dpdk_opt.h"
#include "dpdk_ipv6.h"
#include "dpdk_ipv4.h"
#include "dpdk_mbuf.h"
struct rte_ether_addr cli_mac=
    {{0x48, 0x57, 0x02, 0x64, 0xea, 0x1e}};
extern struct in6_addr net_ip6 ;
extern struct in6_addr gw_ip6 ;
extern struct rte_ether_addr gw_mac;
extern struct rte_ether_addr net_ethaddr;
static uint32_t client_ip_addr = RTE_IPV4(10,10,103,81);
extern uint32_t server_ip_addr;
int mbuf_6to4(struct rte_mbuf *mbuf,
              const struct in_addr *saddr,
              const struct in_addr *daddr)
{
    struct ip6_hdr *ip6h = ip6_hdr(mbuf);
    struct rte_ipv4_hdr *ip4h;
    uint8_t next_prot;
    uint8_t ttl;

    /*
     * ext_hdr not support yet
     */
    if (ip6h->ip6_nxt != IPPROTO_TCP &&
        ip6h->ip6_nxt != IPPROTO_UDP &&
        ip6h->ip6_nxt != IPPROTO_ICMPV6 &&
        ip6h->ip6_nxt != IPPROTO_OPT) {
        return EDPVS_NOTSUPP;
    }
    if (rte_pktmbuf_adj(mbuf, mbuf->l3_len) == NULL)
        return EDPVS_DROP;

    next_prot = ip6h->ip6_nxt;
    ttl = ip6h->ip6_hlim;
    ip4h = (struct rte_ipv4_hdr *)rte_pktmbuf_prepend(mbuf, sizeof(struct rte_ipv4_hdr));
    if (!ip4h)
        return EDPVS_NOROOM;

    ip4h->version_ihl     = ((4 << 4) | 5);
    ip4h->type_of_service = 0;
    ip4h->total_length    = htons(mbuf_nat6to4_len(mbuf));
    ip4h->fragment_offset = htons(RTE_IPV4_HDR_DF_FLAG);
    ip4h->time_to_live    = ttl;
    ip4h->next_proto_id   = next_prot;
    ip4h->hdr_checksum    = 0;
    ip4h->src_addr        = saddr->s_addr;
    ip4h->dst_addr        = daddr->s_addr;
    ip4h->packet_id       = 0; // NO FRAG, so 0 is OK?

    mbuf->l3_len = sizeof(struct rte_ipv4_hdr);

    return EDPVS_OK;
}
/* 
 * mbuf->l3_len =  sizeof(struct rte_ipv4_hdr)
 * data_off 指向ipv4hdr开始位置
 * rte_pktmbuf_adj从二层到三层转发,去二层头就可以用这个。首部向后缩小空间, 改变data_off的值。
 */
int mbuf_4to6(struct rte_mbuf *mbuf,
              const struct in6_addr *saddr,
              const struct in6_addr *daddr)
{
    struct rte_ipv4_hdr *ip4h = ip4_hdr(mbuf);
    struct ip6_hdr *ip6h;
    uint16_t plen;
    uint8_t hops;
    uint8_t next_prot;
#if 1
    if (mbuf->l3_len != sizeof(struct rte_ipv4_hdr)) {
        return EDPVS_NOTSUPP;
    }
#else
    if (mbuf->l3_len != ip4_hdrlen(mbuf))  {
        return EDPVS_NOTSUPP;
    }
#endif
    if (rte_pktmbuf_adj(mbuf, mbuf->l3_len) == NULL)
        return EDPVS_DROP;

    plen = mbuf_nat4to6_len(mbuf);
    next_prot = ip4h->next_proto_id;
    hops = ip4h->time_to_live;
    ip6h = (struct ip6_hdr *)rte_pktmbuf_prepend(mbuf, sizeof(struct ip6_hdr));
    if (!ip6h)
        return EDPVS_NOROOM;

    ip6h->ip6_flow  = 0;
    ip6h->ip6_vfc   = 0x60;
    ip6h->ip6_plen  = htons(plen);
    ip6h->ip6_nxt   = next_prot;
    ip6h->ip6_hlim  = hops;
    ip6h->ip6_src   = *saddr;
    ip6h->ip6_dst   = *daddr;

    mbuf->l3_len = sizeof(struct ip6_hdr);

    return EDPVS_OK;
}
int xmit46_out(struct rte_mbuf *mbuf)
{
    struct rte_ipv6_hdr *ip6h;
    struct rte_ether_hdr *eth_hdr; 
    int err;
    err = mbuf_4to6(mbuf, &net_ip6, &gw_ip6);
    if (err) {
        goto error;
    }
    ip6h = rte_pktmbuf_mtod(mbuf, struct rte_ipv6_hdr *);
#if 0
    rte_memcpy(iph->src_addr, &net_ip6, sizeof(struct in6_addr));
    rte_memcpy(iph->dst_addr, &gw_ip6, sizeof(struct in6_addr));
#endif
    mbuf->ol_flags |= PKT_TX_IPV6;
    eth_hdr = (struct rte_ether_hdr *)rte_pktmbuf_prepend(mbuf, (uint16_t)sizeof(struct rte_ether_hdr));
    if (!eth_hdr) {
        goto error;
    }
    rte_ether_addr_copy(&gw_mac, &eth_hdr->d_addr);
    rte_ether_addr_copy(&net_ethaddr, &eth_hdr->s_addr);
    eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV6);
    mbuf->packet_type = RTE_ETHER_TYPE_IPV6;
    return EDPVS_OK;
error:
    rte_pktmbuf_free(mbuf);
    return EDPVS_NOROOM;
}
struct ip6_hdr * ipv6_hdr_push_common(const struct rte_mbuf *mbuf,  struct rte_mbuf *dst_mbuf, uint16_t plen)
{
    struct rte_ipv4_hdr *ip4h = ip4_hdr(mbuf);
    struct ip6_hdr *ip6h = mbuf_push_ip6_hdr(dst_mbuf);
    uint8_t hops;
    hops = ip4h->time_to_live;
    ip6h->ip6_flow  = 0;
    ip6h->ip6_vfc   = 0x60;
    ip6h->ip6_plen  = htons(plen);
    ip6h->ip6_hlim  = hops;
    ip6h->ip6_src   = net_ip6;
    ip6h->ip6_dst   = gw_ip6;
    dst_mbuf->l3_len = sizeof(struct ip6_hdr);
    return ip6h;
}
struct  rte_ipv4_hdr* ipv4_hdr_push_common(const struct rte_mbuf *mbuf,  struct rte_mbuf *dst_mbuf, uint16_t plen, uint16_t proto)
{
    struct rte_ipv4_hdr *ip4h = mbuf_push_ip4_hdr(dst_mbuf);
    ip4h->hdr_checksum = 0;
    ip4h->packet_id = 0;
    ip4h->src_addr =  rte_cpu_to_be_32(server_ip_addr);
    ip4h->dst_addr =  rte_cpu_to_be_32(client_ip_addr);
    ip4h->version_ihl = 0x45;  // version is 4 and IHL is 5
    //ip4h->fragment_offset= 0;  // version is 4 and IHL is 5
    ip4h->type_of_service = 0;
    ip4h->time_to_live = 64;
    //i4hdr->total_length = htons(payload_len + sizeof(struct rte_icmp_hdr) + sizeof(struct rte_ipv4_hdr));
    ip4h->total_length = htons(plen + sizeof(struct rte_ipv4_hdr));
    ip4h->fragment_offset = htons(RTE_IPV4_HDR_DF_FLAG);
    ip4h->next_proto_id = proto;
    ip4h->hdr_checksum = rte_ipv4_cksum(ip4h);
    dst_mbuf->l3_len = sizeof(struct rte_ipv4_hdr);
    return ip4h;
}
int eth_hdr_push(struct rte_mbuf *mbuf)
{
    struct rte_ether_hdr *eth_hdr = NULL;
    eth_hdr = mbuf_push_rte_eth_hdr(mbuf);
    rte_ether_addr_copy(&gw_mac, &eth_hdr->d_addr);
    rte_ether_addr_copy(&net_ethaddr, &eth_hdr->s_addr);
    eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV6);
    mbuf->packet_type = RTE_ETHER_TYPE_IPV6;
    return EDPVS_OK;
}
int eth_hdr_push_for_ipv4(struct rte_mbuf *mbuf)
{
    struct rte_ether_hdr *eth_hdr = NULL;
    eth_hdr = mbuf_push_rte_eth_hdr(mbuf);
    rte_ether_addr_copy(&cli_mac, &eth_hdr->d_addr);
    rte_ether_addr_copy(&net_ethaddr, &eth_hdr->s_addr);
    eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
    mbuf->packet_type = RTE_ETHER_TYPE_IPV4;
    return EDPVS_OK;
}
