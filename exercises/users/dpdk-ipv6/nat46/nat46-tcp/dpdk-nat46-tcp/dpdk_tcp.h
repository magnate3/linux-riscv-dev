#ifndef __TCP_H
#define __TCP_H

#include <stdint.h>
#include <netinet/ip.h>
#define __FAVOR_BSD
#include <netinet/tcp.h>
#undef __FAVOR_BSD
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_tcp.h>
#include <rte_common.h>
#include "dpdk_ipv4.h"
#include "dpdk_ipv6.h"
#define USE_TCP_DPDK 1
void dpdk_dump_tcph(struct rte_tcp_hdr *tcp_hdr, unsigned int l4_len);
#if USE_TCP_DPDK
inline struct rte_tcp_hdr*tcp_hdr(const struct rte_mbuf *mbuf);
#else
inline struct tcphdr *tcp_hdr(const struct rte_mbuf *mbuf);
#endif
#endif
