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

#if RTE_VERSION < RTE_VERSION_NUM(21, 0, 0, 0)
#define RTE_MBUF_F_RX_L4_CKSUM_BAD  PKT_RX_L4_CKSUM_BAD
#define RTE_MBUF_F_RX_IP_CKSUM_BAD  PKT_RX_IP_CKSUM_BAD
#define RTE_MBUF_F_TX_IPV6          PKT_TX_IPV6
#define RTE_MBUF_F_TX_IP_CKSUM      PKT_TX_IP_CKSUM
#define RTE_MBUF_F_TX_IPV4          PKT_TX_IPV4
#define RTE_MBUF_F_TX_TCP_CKSUM     PKT_TX_TCP_CKSUM
#define RTE_MBUF_F_TX_UDP_CKSUM     PKT_TX_UDP_CKSUM
#endif

#if RTE_VERSION < RTE_VERSION_NUM(21, 0, 0, 0)
#define RTE_ETH_MQ_RX_NONE              ETH_MQ_RX_NONE
#define RTE_ETH_MQ_TX_NONE              ETH_MQ_TX_NONE

#define RTE_ETH_TX_OFFLOAD_IPV4_CKSUM   DEV_TX_OFFLOAD_IPV4_CKSUM
#define RTE_ETH_TX_OFFLOAD_TCP_CKSUM    DEV_TX_OFFLOAD_TCP_CKSUM
#define RTE_ETH_TX_OFFLOAD_UDP_CKSUM    DEV_TX_OFFLOAD_UDP_CKSUM
#define RTE_ETH_TX_OFFLOAD_VLAN_INSERT  DEV_TX_OFFLOAD_VLAN_INSERT
#define RTE_ETH_RX_OFFLOAD_VLAN_STRIP   DEV_RX_OFFLOAD_VLAN_STRIP

#define RTE_ETH_RSS_IPV4                ETH_RSS_IPV4
#define RTE_ETH_RSS_FRAG_IPV4           ETH_RSS_FRAG_IPV4
#define RTE_ETH_RSS_IPV6                ETH_RSS_IPV6
#define RTE_ETH_RSS_FRAG_IPV6           ETH_RSS_FRAG_IPV6

#define RTE_ETH_RSS_NONFRAG_IPV4_UDP    ETH_RSS_NONFRAG_IPV4_UDP
#define RTE_ETH_RSS_NONFRAG_IPV6_UDP    ETH_RSS_NONFRAG_IPV6_UDP
#define RTE_ETH_RSS_NONFRAG_IPV4_TCP    ETH_RSS_NONFRAG_IPV4_TCP
#define RTE_ETH_RSS_NONFRAG_IPV6_TCP    ETH_RSS_NONFRAG_IPV6_TCP

#define RTE_ETH_MQ_RX_RSS               ETH_MQ_RX_RSS
#endif
#endif
