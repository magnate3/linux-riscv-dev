#ifndef __DPDK_H
#define __DPDK_H

#include <rte_version.h>
#if RTE_VERSION < RTE_VERSION_NUM(19, 0, 0, 0)
#define RTE_IPV4_CKSUM(iph) rte_ipv4_cksum((struct ipv4_hdr*)iph)
#define RTE_IPV4_UDPTCP_CKSUM(iph, th) rte_ipv4_udptcp_cksum((const struct ipv4_hdr *)iph, th)
#define RTE_IPV6_UDPTCP_CKSUM(iph, th) rte_ipv6_udptcp_cksum((const struct ipv6_hdr *)iph, (const void *)th)
#else
#define RTE_IPV4_CKSUM(iph) rte_ipv4_cksum((const struct rte_ipv4_hdr *)iph)
#define RTE_IPV4_UDPTCP_CKSUM(iph, th) rte_ipv4_udptcp_cksum((const struct rte_ipv4_hdr *)iph, th)
#define RTE_IPV6_UDPTCP_CKSUM(iph, th) rte_ipv6_udptcp_cksum((const struct rte_ipv6_hdr *)iph, (const void *)th)
#endif

#if RTE_VERSION < RTE_VERSION_NUM(21, 11, 0, 0)
#define RTE_MBUF_F_TX_VLAN  PKT_TX_VLAN
#endif


#endif
