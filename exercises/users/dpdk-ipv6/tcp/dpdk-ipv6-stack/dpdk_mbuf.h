/*
 * DPVS is a software load balancer (Virtual Server) based on DPDK.
 *
 * Copyright (C) 2021 iQIYI (www.iqiyi.com).
 * All Rights Reserved.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */
/*
 * mbuf.c of dpvs.
 *
 * it includes some mbuf related function beyond dpdk librte_mbuf.
 */
#ifndef __DP_VS_MBUF_H__
#define __DP_VS_MBUF_H__
#include <stdlib.h>
#include <string.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_version.h>
#include "rte_mbuf.h"
#define RTE_PKTMBUF_PUSH(m, type) (type *)rte_pktmbuf_append(m, sizeof(type))
#define mbuf_eth_hdr(m) rte_pktmbuf_mtod(m, struct rte_ether_hdr*)
#define mbuf_ip_hdr(m) rte_pktmbuf_mtod_offset(m, struct iphdr*, sizeof(struct rte_ether_hdr))
#define mbuf_ip6_hdr(m) rte_pktmbuf_mtod_offset(m, struct ip6_hdr*, sizeof(struct rte_ether_hdr))
#define mbuf_push_ip6_hdr(m) RTE_PKTMBUF_PUSH(m, struct ip6_hdr)
#define mbuf_push_ip4_hdr(m) RTE_PKTMBUF_PUSH(m, struct rte_ipv4_hdr)
#define mbuf_push_eth_hdr(m) RTE_PKTMBUF_PUSH(m, struct rte_ether_hdr)
#define mbuf_push_rte_eth_hdr(m) RTE_PKTMBUF_PUSH(m, struct rte_ether_hdr)
#define mbuf_icmp_hdr(m) rte_pktmbuf_mtod_offset(m, struct icmphdr*, sizeof(struct struct rte_ether_hdr) + sizeof(struct iphdr));
#define mbuf_tcp_hdr(m) ({                      \
    struct tcphdr *th = NULL;                   \
    struct rte_ether_hdr *eth = mbuf_eth_hdr(m);      \
    if (eth->ether_type == htons(RTE_ETHER_TYPE_IPV4)) {  \
        th = rte_pktmbuf_mtod_offset(m, struct tcphdr*, sizeof(struct rte_ether_hdr) + sizeof(struct iphdr));     \
    } else {    \
        th = rte_pktmbuf_mtod_offset(m, struct tcphdr*, sizeof(struct rte_ether_hdr) + sizeof(struct ip6_hdr));   \
    }\
    th;})
void mbuf_log(struct rte_mbuf *m, const char *tag);

#ifdef DPERF_DEBUG
#define MBUF_LOG(m, tag) mbuf_log(m, tag)
#else
#define MBUF_LOG(m, tag)
#endif
/* for each mbuf including heading mbuf and segments */
#define mbuf_foreach(m, pos)    \
    for (pos = m; pos != NULL; pos = pos->next)

/* for each segments of mbuf */
#define mbuf_foreach_seg(m, s)    \
    for (s = m->next; s != NULL; s = s->next)

#define mbuf_foreach_seg_safe(m, n, s)    \
    for (s = m->next, n = s ? s->next : NULL; \
        s != NULL; \
        s = n, n = s ? s->next : NULL)

#define MBUF_USERDATA(m, type, field) \
    (*((type *)(mbuf_userdata((m), (field)))))

#define MBUF_USERDATA_CONST(m, type, field) \
    (*((type *)(mbuf_userdata_const((m), (field)))))

typedef union {
    void *hdr;
    struct {
        uint64_t l2_len:RTE_MBUF_L2_LEN_BITS;           /* L2 Header Length */
        uint64_t l3_len:RTE_MBUF_L3_LEN_BITS;           /* L3 Header Length */
        uint64_t l4_len:RTE_MBUF_L4_LEN_BITS;           /* L4 Header Length */
        uint64_t outer_l2_len:RTE_MBUF_OUTL2_LEN_BITS;  /* Outer L2 Header Length */
        uint64_t outer_l3_len:RTE_MBUF_OUTL3_LEN_BITS;  /* Outer L3 Header Length */
    };
} mbuf_userdata_field_proto_t;

typedef void * mbuf_userdata_field_route_t;

typedef enum {
    MBUF_FIELD_PROTO = 0,
    MBUF_FIELD_ROUTE,
} mbuf_usedata_field_t;

/**
 * mbuf_copy_bits - copy bits from mbuf to buffer.
 * see skb_copy_bits().
 */
static inline int mbuf_copy_bits(const struct rte_mbuf *mbuf,
                 int offset, void *to, int len)
{
    const struct rte_mbuf *seg;
    int start, copy, end;

    if (offset + len > (int)mbuf->pkt_len)
        return -1;

    start = 0;
    mbuf_foreach(mbuf, seg) {
        end = start + seg->data_len;

        if ((copy = end - offset) > 0) {
            if (copy > len)
                copy = len;

            memcpy(to, rte_pktmbuf_mtod_offset(
                        seg, void *, offset - start),
                   copy);

            if ((len -= copy) == 0)
                return 0;
            offset += copy;
            to += copy;
        }

        start = end;
    }

    if (!len)
        return 0;

    return -1;
}

static inline void *mbuf_tail_point(const struct rte_mbuf *mbuf)
{
    return rte_pktmbuf_mtod_offset(mbuf, void *, mbuf->data_len);
}

static inline void *mbuf_header_pointer(const struct rte_mbuf *mbuf,
                    unsigned int offset, int len, void *buffer)
{
    if (unlikely(mbuf->data_len < offset + len)) {
        if (unlikely(mbuf->pkt_len < offset + len))
            return NULL;

        if (mbuf_copy_bits(mbuf, offset, buffer, len) != 0)
            return NULL;

        return buffer;
    }

    return rte_pktmbuf_mtod_offset(mbuf, void *, offset);
}

/**
 * mbuf_may_pull - pull bits from segments to heading mbuf if needed.
 * see pskb_may_pull() && __pskb_pull_tail().
 *
 * it expands heading mbuf, moving it's tail forward and copying necessary
 * data from segments part.
 *
 * return 0 if success and -1 on error.
 */
int mbuf_may_pull(struct rte_mbuf *mbuf, unsigned int len);

/**
* Copy a rte_mbuf including the data area.
*
* return a new rte_mbuf if success and NULL on error.
*/
struct rte_mbuf *mbuf_copy(struct rte_mbuf *md, struct rte_mempool *mp);
void mbuf_copy_metadata(struct rte_mbuf *mi, struct rte_mbuf *m);

#ifdef CONFIG_DPVS_MBUF_DEBUG
inline void dp_vs_mbuf_dump(const char *msg, int af, const struct rte_mbuf *mbuf);
#endif

void *mbuf_userdata(struct rte_mbuf *, mbuf_usedata_field_t);
const void *mbuf_userdata_const(const struct rte_mbuf *, mbuf_usedata_field_t);

static inline void mbuf_userdata_reset(struct rte_mbuf *m)
{
    memset((void *)m->dynfield1, 0, sizeof(m->dynfield1));
}
static void mbuf_dump(struct rte_mbuf* m)
{
    printf("RTE_PKTMBUF_HEADROOM: %u\n", RTE_PKTMBUF_HEADROOM);
    printf("sizeof(mbuf): %lu\n", sizeof(struct rte_mbuf));
    printf("m addr: %p\n", m);
    printf("m->buf_addr: %p\n", m->buf_addr);
    printf("m->refcnt: %u\n", m->refcnt);
    printf("m->data_off: %u\n", m->data_off);
    printf("m->buf_len: %u\n", m->buf_len);
    printf("m->pkt_len: %u\n", m->pkt_len);
    printf("m->data_len: %u\n", m->data_len);
    printf("m->nb_segs: %u\n", m->nb_segs);
    printf("m->next: %p\n", m->next);
    printf("m->buf_addr+m->data_off: %p\n", (char*)m->buf_addr+m->data_off);
    printf("rte_pktmbuf_mtod(m): %p\n", rte_pktmbuf_mtod(m, char*));
    printf("rte_pktmbuf_data_len(m): %u\n", rte_pktmbuf_data_len(m));
    printf("rte_pktmbuf_pkt_len(m): %u\n", rte_pktmbuf_pkt_len(m)); 
    printf("rte_pktmbuf_headroom(m): %u\n", rte_pktmbuf_headroom(m));
    printf("rte_pktmbuf_tailroom(m): %u\n", rte_pktmbuf_tailroom(m));
    printf("rte_pktmbuf_data_room_size(mpool): %u\n", 
                rte_pktmbuf_data_room_size(m->pool));
    printf("rte_pktmbuf_priv_size(mpool): %u\n\n",
                rte_pktmbuf_priv_size(m->pool));
}
int mbuf_init(void);

/*
 * Return a pointer to L2 header, and set mbuf->l2_len.
 * The start of data in the mbuf should be L2 data.
 * It assumes that L2 header is in the first seg if the mbuf is not continuous.
 * Only support outer headers for tunnelling packets.
 * */
void *mbuf_header_l2(struct rte_mbuf *mbuf);

/*
 * Return a pointer to L3 header, and set mbuf->l3_len.
 * The start of data in the mbuf should be L2 data.
 * It assumes that L3 header is in the first seg if the mbuf is not continuous.
 * Only support outer headers for tunnelling packets.
 * */
void *mbuf_header_l3(struct rte_mbuf *mbuf);

/*
 * Return a pointer to L4 header, and set mbuf->l4_len.
 * The start of data in the mbuf should be L2 data.
 * It assumes that L4 header is in the first seg if the mbuf is not continuous.
 * Only support outer headers for tunnelling packets.
 * */
void *mbuf_header_l4(struct rte_mbuf *mbuf);

/*
 * Return ether type (ETHER_TYPE_XXX) in the mbuf.
 * The start of data in the mbuf should be L2 data,
 * and vlan is ignored.
 * Only support outer headers for tunnelling packets.
 * */
uint16_t mbuf_ether_type(struct rte_mbuf *mbuf);

/*
 * Return socket address family (AF_INET | AF_INET6) derived from ether type
 * in the mbuf. The function is based on "mbuf_ether_type".
 * */
int mbuf_address_family(struct rte_mbuf *mbuf);

/*
 * Return protocol type (IPPROTO_XX) in the mbuf.
 * The start of data in the mbuf should be L2 data.
 * Only support outer headers for tunnelling packets.
 * */
uint8_t mbuf_protocol(struct rte_mbuf *mbuf);

#endif /* __DP_VS_MBUF_H__ */
