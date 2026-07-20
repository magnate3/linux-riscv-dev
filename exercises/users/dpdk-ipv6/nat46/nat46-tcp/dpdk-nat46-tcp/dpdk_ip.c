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
#include "dpdk_icmp6.h"
#include "dpdk_nat46.h"
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
protocol_process_icmp6(const void *data, uint16_t len)
{
    if (len < sizeof(struct icmp6_hdr))
        return DPDK_IP_UNKNOWN;

    const struct icmp6_hdr *hdr;
    hdr = (const struct icmp6_hdr *)data;

    if (hdr->icmp6_type >= ND_ROUTER_SOLICIT && hdr->icmp6_type <= ND_REDIRECT)
        return DPDK_IP_NDP;

    return DPDK_IP_UNKNOWN;
}
static int
protocol_process_icmp6_raw(struct rte_mbuf *mbuf, uint16_t len, uint16_t ip_offfset, uint16_t icmp6_offfset)
{
    // dpdk not have 'struct rte_icmp6_hdr' 
    if (len < sizeof(struct icmp6_hdr))
        return DPDK_IP_UNKNOWN;
    return icmp6_rcv(mbuf,ip_offfset, icmp6_offfset);
}
#endif

static enum DPDK_IP_Return
protocol_process_ip(const void *data, uint16_t len, uint16_t eth_frame_type)
{
    return DPDK_IP_UNKNOWN;
}
static  int
protocol_process_ip_raw(struct rte_mbuf *mbuf)
{
    uint8_t proto;
    int hdr_len;
    const void *next;
    uint16_t next_len;
    const struct rte_ether_hdr *hdr;
    void *data = rte_pktmbuf_mtod(mbuf, void*);
    uint16_t len = rte_pktmbuf_data_len(mbuf);
    uint16_t ether_type;
    uint16_t l2_len = 0;
    const struct rte_vlan_hdr *vlanhdr;
    if(len < RTE_ETHER_ADDR_LEN)
        return DPDK_IP_UNKNOWN; 
    hdr = (const struct rte_ether_hdr *)data;
    ether_type = rte_be_to_cpu_16(hdr->ether_type);
    data += RTE_ETHER_HDR_LEN;
    len -= RTE_ETHER_HDR_LEN;
    l2_len += RTE_ETHER_HDR_LEN;
    if (ether_type == RTE_ETHER_TYPE_VLAN) {
        vlanhdr = (struct rte_vlan_hdr *)data;
        ether_type = rte_be_to_cpu_16(vlanhdr->eth_proto);
        data += sizeof(struct rte_vlan_hdr);
        len -= sizeof(struct rte_vlan_hdr);
        l2_len += sizeof(struct rte_vlan_hdr);
    }
    if(ether_type == RTE_ETHER_TYPE_ARP)
        return DPDK_IP_ARP;
    if (ether_type == RTE_ETHER_TYPE_IPV4) {
        if(len < sizeof(struct rte_ipv4_hdr))
            return DPDK_IP_UNKNOWN;

        const struct rte_ipv4_hdr *hdr = (const struct rte_ipv4_hdr *)data;
        hdr_len = (hdr->version_ihl & 0x0f) << 2;
        if (len < hdr_len)
            return DPDK_IP_UNKNOWN;

        proto = hdr->next_proto_id;
#ifdef INET6
    } else if(ether_type == RTE_ETHER_TYPE_IPV6) {
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
    mbuf->l2_len = l2_len;
    mbuf->l3_len = hdr_len;
    if(ether_type == RTE_ETHER_TYPE_IPV6) {
         switch (proto) {
           case IPPROTO_TCP:
               printf("tcp over ipv6 ************************                 \n");    
               return process_tcp64(mbuf);
           case IPPROTO_UDP:
               return process_udp64(mbuf);
#ifdef INET6
           case IPPROTO_IPV6:
               return protocol_process_ip(next, next_len, RTE_ETHER_TYPE_IPV6);
           case IPPROTO_ICMPV6:
               //printf("icmp6 ************************                 \n");    
               return protocol_process_icmp6_raw(mbuf,next_len,l2_len,l2_len + hdr_len);
#endif
           default:
               rte_pktmbuf_free(mbuf);
         }
         return DPDK_IP_UNKNOWN;
     }
       switch (proto) {
           case IPPROTO_TCP:
               return protocol_process_tcp(next, next_len);
           case IPPROTO_UDP:
               return protocol_process_udp(next, next_len);
           case IPPROTO_IPIP:
               return protocol_process_ip(next, next_len, RTE_ETHER_TYPE_IPV4);
           default:
               rte_pktmbuf_free(mbuf);
       }
   
    return DPDK_IP_UNKNOWN;
}

//enum DPDK_IP_Return
//dpdk_ip_proto_process(const void *data, uint16_t len, uint16_t eth_frame_type)
int dpdk_ip_proto_process_raw(struct rte_mbuf *mbuf)
{
    //return protocol_process_ip(data, len, eth_frame_type);
    return protocol_process_ip_raw(mbuf);
}
