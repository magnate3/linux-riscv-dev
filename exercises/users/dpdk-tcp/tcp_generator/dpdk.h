/*
 * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Jianzhang Peng (pengjianzhang@baidu.com)
 */

#ifndef __DPDK_H
#define __DPDK_H

#include <rte_version.h>

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

#if RTE_VERSION < RTE_VERSION_NUM(21, 0, 0, 0)
#define RTE_MBUF_F_RX_L4_CKSUM_BAD  PKT_RX_L4_CKSUM_BAD
#define RTE_MBUF_F_RX_IP_CKSUM_BAD  PKT_RX_IP_CKSUM_BAD
#define RTE_MBUF_F_TX_IPV6          PKT_TX_IPV6
#define RTE_MBUF_F_TX_IP_CKSUM      PKT_TX_IP_CKSUM
#define RTE_MBUF_F_TX_IPV4          PKT_TX_IPV4
#define RTE_MBUF_F_TX_TCP_CKSUM     PKT_TX_TCP_CKSUM
#define RTE_MBUF_F_TX_UDP_CKSUM     PKT_TX_UDP_CKSUM
#endif

#if RTE_VERSION >= RTE_VERSION_NUM(19, 0, 0, 0)
#include <net/ethernet.h>

#define ETHER_TYPE_IPv4 ETHERTYPE_IP
#define ETHER_TYPE_IPv6 ETHERTYPE_IPV6
#define ETHER_TYPE_ARP  ETHERTYPE_ARP

#define RTE_ETH_MACADDR_GET(port_id, mac_addr) rte_eth_macaddr_get(port_id, (struct rte_ether_addr *)mac_addr)
#else
#define RTE_ETH_MACADDR_GET(port_id, mac_addr) rte_eth_macaddr_get(port_id, (struct ether_addr *)mac_addr)
#endif

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
#if RTE_VERSION < RTE_VERSION_NUM(21, 11, 0, 0)
/**
 *  * Macro to browse all running lcores except the main lcore.
 *   */
#define RTE_LCORE_FOREACH_WORKER(i)                                     \
        for (i = rte_get_next_lcore(-1, 1, 0);                          \
             i < RTE_MAX_LCORE;                                         \
             i = rte_get_next_lcore(i, 1, 0))

#define RTE_BIT64(nr) (UINT64_C(1) << (nr))

/**
 *  * Get the uint32_t value for a specified bit set.
 *   *
 *    * @param nr
 *     *   The bit number in range of 0 to 31.
 *      */
#define RTE_BIT32(nr) (UINT32_C(1) << (nr))
#define RTE_ETH_RSS_IPV6_TCP_EX        RTE_BIT64(16)
#define RTE_ETH_RX_OFFLOAD_UDP_CKSUM        RTE_BIT64(2)
#define RTE_ETH_RX_OFFLOAD_TCP_CKSUM        RTE_BIT64(3)
#define RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE   RTE_BIT64(16)
#define RTE_ETH_TX_OFFLOAD_SECURITY         RTE_BIT64(17)
#define RTE_ETH_RSS_TCP ( \
        RTE_ETH_RSS_NONFRAG_IPV4_TCP | \
        RTE_ETH_RSS_NONFRAG_IPV6_TCP | \
        RTE_ETH_RSS_IPV6_TCP_EX)
#define RTE_ETH_RX_OFFLOAD_IPV4_CKSUM       RTE_BIT64(1)
#define RTE_ETH_RX_OFFLOAD_CHECKSUM (RTE_ETH_RX_OFFLOAD_IPV4_CKSUM | \
                                 RTE_ETH_RX_OFFLOAD_UDP_CKSUM | \
                                 RTE_ETH_RX_OFFLOAD_TCP_CKSUM)
//#define RTE_MBUF_F_TX_TCP_CKSUM     (1ULL << 52)
/**
 *  * Packet is IPv4. This flag must be set when using any offload feature
 *   * (TSO, L3 or L4 checksum) to tell the NIC that the packet is an IPv4
 *    * packet. If the packet is a tunneled packet, this flag is related to
 *     * the inner headers.
 *      */
//#define RTE_MBUF_F_TX_IPV4          (1ULL << 55)

/**
 *  * Offload the IP checksum in the hardware. The flag RTE_MBUF_F_TX_IPV4 should
 *   * also be set by the application, although a PMD will only check
 *    * RTE_MBUF_F_TX_IP_CKSUM.
 *     *  - fill the mbuf offload information: l2_len, l3_len
 *      */
//#define RTE_MBUF_F_TX_IP_CKSUM      (1ULL << 54)
#endif
#endif
