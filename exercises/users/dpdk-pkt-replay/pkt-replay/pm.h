#ifndef PM_H
#define PM_H

#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_tcp.h>
#include <rte_vxlan.h>
struct packet_model
{
    struct
    {
        struct rte_ether_hdr eth;
        struct rte_vlan_hdr vlan;
        struct rte_ipv4_hdr ip;
        struct rte_udp_hdr udp;
        struct rte_vxlan_hdr vx;
    }__attribute__((__packed__)) vxlan;
    struct
    {
        struct rte_ether_hdr eth;
        struct rte_ipv4_hdr ip;
        struct rte_tcp_hdr tcp;
    }__attribute__((__packed__)) tcp;
    struct
    {
        struct rte_ether_hdr eth;
        struct rte_ipv4_hdr ip;
        struct rte_udp_hdr udp;
    }__attribute__((__packed__)) udp;
    int is_udp;
    int is_vxlan;
};

#endif
