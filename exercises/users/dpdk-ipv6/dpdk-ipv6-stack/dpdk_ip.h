#ifndef _STACK_DPDK_IP_H
#define _STACK_DPDK_IP_H
#include <rte_mbuf.h>
#include <rte_memory.h>
enum DPDK_IP_Return {
    DPDK_IP_UNKNOWN = -1,
    DPDK_IP_ARP = 1,
    DPDK_IP_SUCC = 2,
#ifdef INET6
    DPDK_IP_NDP = 3,  // Neighbor Solicitation/Advertisement, Router Solicitation/Advertisement/Redirect
#endif
};
enum DPDK_IP_Return dpdk_ip_proto_process(const void *data, uint16_t len, uint16_t eth_frame_type);
int dpdk_ip_proto_process_raw(struct rte_mbuf *mbuf);
#endif
