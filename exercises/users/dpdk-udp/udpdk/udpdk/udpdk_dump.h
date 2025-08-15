//
// Created by leoll2 on 11/19/20.
// Copyright (c) 2020 Leonardo Lai. All rights reserved.
//
// The following code derives in part from netmap pkt-gen.c
//

#ifndef UDPDK_DUMP_H
#define UDPDK_DUMP_H

void udpdk_dump_payload(const char *payload, int len);

void udpdk_dump_mbuf(struct rte_mbuf *m);
void udpdk_dump_eth(struct rte_mbuf *m);
/* helper functions */
static inline struct rte_ipv4_hdr *ip4_hdr(const struct rte_mbuf *mbuf)
{
    /* can only invoked at L3 */
    return rte_pktmbuf_mtod_offset(mbuf, struct rte_ipv4_hdr *,sizeof(struct rte_ether_hdr));
}

static inline uint16_t ip4_hdrlen(const struct rte_mbuf *mbuf)
{
    return (ip4_hdr(mbuf)->version_ihl & 0xf) << 2;
}
#endif  // UDPDK_DUMP_H
