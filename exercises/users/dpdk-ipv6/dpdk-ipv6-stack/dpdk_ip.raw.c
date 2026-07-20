#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <netinet/icmp6.h>

#include <rte_config.h>
#include <rte_ether.h>
#include <rte_bus_pci.h>
#include <rte_ethdev.h>
#include "dpdk_ip.h"
static enum DPDK_IP_Return
protocol_process_tcp(const void *data, uint16_t len)
{
    if (len < sizeof(struct rte_tcp_hdr))
        return DPDK_IP_UNKNOWN;

    const struct rte_tcp_hdr *hdr;
    hdr = (const struct rte_tcp_hdr *)data;

    return DPDK_IP_SUCC;
}
static enum DPDK_IP_Return
protocol_process_udp(const void* data,uint16_t len)
{
    if (len < sizeof(struct rte_udp_hdr))
        return DPDK_IP_UNKNOWN;

    const struct rte_udp_hdr *hdr;
    hdr = (const struct rte_udp_hdr *)data;

    return DPDK_IP_SUCC;
}

#ifdef INET6
/*
 * https://www.iana.org/assignments/ipv6-parameters/ipv6-parameters.xhtml
 */
#ifndef IPPROTO_HIP
#define IPPROTO_HIP 139
#endif

#ifndef IPPROTO_SHIM6
#define IPPROTO_SHIM6   140
#endif

#ifndef IPPROTO_MH
#define IPPROTO_MH   135
#endif
static int
get_ipv6_hdr_len(uint8_t *proto, void *data, uint16_t len)
{
    int ext_hdr_len = 0;

    switch (*proto) {
        case IPPROTO_HOPOPTS:   case IPPROTO_ROUTING:   case IPPROTO_DSTOPTS:
        case IPPROTO_MH:        case IPPROTO_HIP:       case IPPROTO_SHIM6:
            ext_hdr_len = *((uint8_t *)data + 1) + 1;
            break;
        case IPPROTO_FRAGMENT:
            ext_hdr_len = 8;
            break;
        case IPPROTO_AH:
            ext_hdr_len = (*((uint8_t *)data + 1) + 2) * 4;
            break;
        case IPPROTO_NONE:
#ifdef FF_IPSEC
        case IPPROTO_ESP:
            //proto = *((uint8_t *)data + len - 1 - 4);
            //ext_hdr_len = len;
#endif
        default:
            return ext_hdr_len;
    }

    if (ext_hdr_len >= len) {
        return len;
    }

    *proto = *((uint8_t *)data);
    ext_hdr_len += get_ipv6_hdr_len(proto, data + ext_hdr_len, len - ext_hdr_len);

    return ext_hdr_len;
}

static enum DPDK_IP_Return
protocol_process_icmp6(void *data, uint16_t len)
{
    if (len < sizeof(struct rte_icmp6_hdr))
        return DPDK_IP_UNKNOWN;

    const struct rte_icmp6_hdr *hdr;
    hdr = (const struct rte_icmp6_hdr *)data;

    if (hdr->icmp6_type >= ND_ROUTER_SOLICIT && hdr->icmp6_type <= ND_REDIRECT)
        return DPDK_IP_NDP;

    return DPDK_IP_UNKNOWN;
}
#endif

static enum DPDK_IP_Return
protocol_process_ip(const void *data, uint16_t len, uint16_t eth_frame_type)
{
    uint8_t proto;
    int hdr_len;
    const void *next;
    uint16_t next_len;

    if (eth_frame_type == RTE_ETHER_TYPE_IPV4) {
        if(len < sizeof(struct rte_ipv4_hdr))
            return DPDK_IP_UNKNOWN;

        const struct rte_ipv4_hdr *hdr = (const struct rte_ipv4_hdr *)data;
        hdr_len = (hdr->version_ihl & 0x0f) << 2;
        if (len < hdr_len)
            return DPDK_IP_UNKNOWN;

        proto = hdr->next_proto_id;
#ifdef INET6
    } else if(eth_frame_type == RTE_ETHER_TYPE_IPV6) {
        if(len < sizeof(struct rte_ipv6_hdr))
            return DPDK_IP_UNKNOWN;

        hdr_len = sizeof(struct rte_ipv6_hdr);
        proto = ((struct rte_ipv6_hdr *)data)->proto;
        hdr_len += get_ipv6_hdr_len(&proto, (void *)data + hdr_len, len - hdr_len);

        if (len < hdr_len)
            return DPDK_IP_UNKNOWN;
#endif
    } else {
        return DPDK_IP_UNKNOWN;
    }

    next = (const void *)data + hdr_len;
    next_len = len - hdr_len;

    switch (proto) {
        case IPPROTO_TCP:
            return protocol_process_tcp(next, next_len);
        case IPPROTO_UDP:
            return protocol_process_udp(next, next_len);
        case IPPROTO_IPIP:
            return protocol_process_ip(next, next_len, RTE_ETHER_TYPE_IPV4);
#ifdef INET6
        case IPPROTO_IPV6:
            return protocol_process_ip(next, next_len, RTE_ETHER_TYPE_IPV6);
        case IPPROTO_ICMPV6:
            return protocol_process_icmp6(next, next_len);
#endif
    }

    return DPDK_IP_UNKNOWN;
}

enum DPDK_IP_Return
dpdk_ip_proto_process(const void *data, uint16_t len, uint16_t eth_frame_type)
{
    return protocol_process_ip(data, len, eth_frame_type);
}
