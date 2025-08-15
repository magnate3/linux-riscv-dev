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
/**
 * IPv6 protocol for "lite stack".
 * Linux Kernel net/ipv6/ is referred.
 *
 * Lei Chen <raychen@qiyi.com>, initial, Jul 2018.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <netinet/ip6.h>
#include <rte_prefetch.h>
#include <rte_ip.h>
#include <rte_ip_frag.h>
#include <rte_cycles.h>
#include "conf/common.h"
#include "dpdk_mbuf.h"
#include "dpdk_eth.h"
#include "dpdk_ipv6.h"
#include "dpdk_icmp6.h"
#include "dpdk_common.h"
#include "dpdk_reassembly.h"
#include "util.h"
#ifdef CONFIG_DPVS_IP_HEADER_DEBUG
static inline void ip6_show_hdr(const char *func, struct rte_mbuf *mbuf)
{
    struct ip6_hdr *hdr;
    char sbuf[64], dbuf[64];

    hdr = ip6_hdr(mbuf);

    inet_ntop(AF_INET6, &hdr->ip6_src, sbuf, sizeof(sbuf));
    inet_ntop(AF_INET6, &hdr->ip6_dst, dbuf, sizeof(dbuf));

    RTE_LOG(DEBUG, IPV6, "%s: [%d] proto %d, %s -> %s\n",
            func, rte_lcore_id(), hdr->ip6_nxt, sbuf, dbuf);
}
#endif

int ip6_local_out(struct rte_mbuf *mbuf)
{
    return 0;
}


static int ip6_forward(struct rte_mbuf *mbuf)
{
    return EDPVS_INVAL;
}


struct rte_mbuf * ipv6_reassemble(struct rte_mbuf *mbuf)
{
    //struct rte_ipv6_fragment_ext *frag_hdr;
    struct ipv6_extension_fragment *frag_hdr;
    struct rte_ipv6_hdr *ip_hdr;
    struct rte_ether_hdr * eth_hdr;
    uint64_t cur_tsc = rte_rdtsc();
    struct netif_port*dev = NULL ;
        dev= netif_port_get(mbuf->port);
        if(!dev)
        {
            return NULL;
        }
        ip_hdr = rte_pktmbuf_mtod_offset(mbuf,struct rte_ipv6_hdr *, sizeof(struct rte_ether_hdr));
        frag_hdr = rte_ipv6_frag_get_ipv6_fragment_header(ip_hdr);
        if (frag_hdr != NULL) {
                        struct rte_mbuf *mo;
                        /* prepare mbuf: setup l2_len/l3_len. */
                        mbuf->l2_len = sizeof(struct rte_ether_hdr);
                        mbuf->l3_len = sizeof(*ip_hdr) + sizeof(*frag_hdr);

                        mo = rte_ipv6_frag_reassemble_packet(dev->frag_tbl, &dev->death_row, mbuf, cur_tsc, ip_hdr, frag_hdr);
                        if (mo == NULL)
                        {
                             return NULL;
                        }
                        if (mo != mbuf) {
                                mbuf = mo;
                                eth_hdr = rte_pktmbuf_mtod(mbuf,
                                                        struct rte_ether_hdr *);
                                ip_hdr = (struct rte_ipv6_hdr *)(eth_hdr + 1);
                                RTE_LOG(INFO, IP_RSMBL, "reassemble mbuf(%p) succ output ip6 header info : \n",mbuf);
                                //ip6_dump_hdr((struct ip6_hdr *)ip_hdr);
                                //mbuf_dump(mbuf); 
#if 0 
                                if(unlikely(0 != rte_pktmbuf_linearize(mbuf)))
                                {
                                     RTE_LOG(INFO, IP_RSMBL, "mbuf(%p) linearize failed : \n",mbuf);
                                     return NULL;
                                }
#endif
                                return mbuf;
                        }
                }
 
       else
       {
            return mbuf;
       }
#if 0
    RTE_LOG(ERR, IP_RSMBL, "death count %d \n",death_row.cnt);
    rte_ip_frag_free_death_row(&dev->death_row,
                                PREFETCH_OFFSET);
#endif
    return NULL;
 }
int ipv6_reassemble_test(struct rte_mbuf **pkts, int num)
{
    //struct rte_ipv6_fragment_ext *frag_hdr;
    struct ipv6_extension_fragment *frag_hdr;
    struct rte_ipv6_hdr *ip_hdr;
    struct rte_ether_hdr * eth_hdr;
    struct rte_ip_frag_death_row death_row;
    struct rte_ip_frag_tbl *frag_tbl;
    uint64_t frag_cycles;
    int i ;
    int socket = 0;
    struct rte_mbuf *mbuf;
    uint64_t cur_tsc = rte_rdtsc();
      /* Each table entry holds information about packet fragmentation. 8< */
    frag_cycles = (rte_get_tsc_hz() + MS_PER_S - 1) / MS_PER_S *
                max_flow_ttl;
    if ((frag_tbl = rte_ip_frag_table_create(max_flow_num,
                        IP_FRAG_TBL_BUCKET_ENTRIES, max_flow_num, frag_cycles,
                        socket)) == NULL) {
                RTE_LOG(ERR, IP_RSMBL, "ip_frag_tbl_creat failed \n");
                return -1;
    }
    for(i=0; i < num; ++ i)
    {
        mbuf = pkts[i]; 
        ip_hdr = rte_pktmbuf_mtod_offset(mbuf,struct rte_ipv6_hdr *, sizeof(struct rte_ether_hdr));
        frag_hdr = rte_ipv6_frag_get_ipv6_fragment_header(ip_hdr);
        if (frag_hdr != NULL) {
                        struct rte_mbuf *mo;
                        /* prepare mbuf: setup l2_len/l3_len. */
                        mbuf->l2_len = sizeof(struct rte_ether_hdr);
                        mbuf->l3_len = sizeof(*ip_hdr) + sizeof(*frag_hdr);

                        mo = rte_ipv6_frag_reassemble_packet(frag_tbl, &death_row, mbuf, cur_tsc, ip_hdr, frag_hdr);
                        if (mo == NULL)
                        {
                             continue;
                        }
                        if (mo != mbuf) {
                                mbuf = mo;
                                eth_hdr = rte_pktmbuf_mtod(mbuf,
                                                        struct rte_ether_hdr *);
                                ip_hdr = (struct rte_ipv6_hdr *)(eth_hdr + 1);
                                RTE_LOG(INFO, IP_RSMBL, "reassemble succ output ip6 header info : \n");
                                //ip6_dump_hdr(ip_hdr);
                        }
                }
    }
 
    RTE_LOG(ERR, IP_RSMBL, "death count %d \n",death_row.cnt);
    rte_ip_frag_free_death_row(&death_row,
                                PREFETCH_OFFSET);
    return 0;
 }
int ipv6_xmit(struct rte_mbuf *mbuf)
{
    //struct ip6_hdr *hdr=NULL;
    //struct netif_port *dev=NULL;

    if (unlikely(NULL == mbuf)) {
        if (mbuf)
            rte_pktmbuf_free(mbuf);
        return EDPVS_INVAL;
    }
    eth_send_packets(&mbuf,DEFAULT_PORTID,0, 1);
    return 0;
}

/*
 * ip6_hdrlen: get ip6 header length, including extension header length
 */
int ip6_hdrlen(const struct rte_mbuf *mbuf) {
    struct ip6_hdr *ip6h = ip6_hdr(mbuf);
    uint8_t ip6nxt = ip6h->ip6_nxt;
    int ip6_hdrlen = ip6_skip_exthdr(mbuf, sizeof(struct ip6_hdr), &ip6nxt);

    /* ip6_skip_exthdr may return -1 */
    return (ip6_hdrlen >= 0) ? ip6_hdrlen : sizeof(struct ip6_hdr);
}

/*
 * "ip6_phdr_cksum" is a upgraded version of DPDK routine "rte_ipv6_phdr_cksum"
 * to support IPv6 extension headers (RFC 2460).
 * */
uint16_t ip6_phdr_cksum(struct ip6_hdr *ip6h, uint64_t ol_flags,
        uint32_t exthdrlen, uint8_t l4_proto)
{
    uint16_t csum;
    uint8_t ip6nxt = ip6h->ip6_nxt;
    uint32_t ip6plen = ip6h->ip6_plen;
    struct in6_addr ip6dst = ip6h->ip6_dst;

    ip6h->ip6_nxt = l4_proto;

    /* length of L4 header plus L4 data */
    ip6h->ip6_plen = htons(ntohs(ip6h->ip6_plen) +
            sizeof(struct ip6_hdr) - exthdrlen);

    /* ip6_dst translation for NEXTHDR_ROUTING exthdrs */
    if (unlikely(ip6nxt == NEXTHDR_ROUTING)) {
        struct ip6_rthdr0 *rh = (struct ip6_rthdr0 *)(ip6h + 1);
        if (likely(rh->ip6r0_segleft > 0))
            ip6h->ip6_dst = rh->ip6r0_addr[rh->ip6r0_segleft - 1];
    }
    /*FIXME: what if NEXTHDR_ROUTING is not the first exthdr? */

    csum = rte_ipv6_phdr_cksum((struct rte_ipv6_hdr *)ip6h, ol_flags);

    /* restore original ip6h header */
    ip6h->ip6_nxt = ip6nxt;
    ip6h->ip6_plen = ip6plen;
    if (unlikely(ip6nxt == NEXTHDR_ROUTING))
        ip6h->ip6_dst = ip6dst;

    return csum;
}

/*
 * "ip6_udptcp_cksum" is a upgraded version of DPDK routine "rte_ipv6_udptcp_cksum"
 * to support IPv6 extension headers (RFC 2460).
 * */
uint16_t ip6_udptcp_cksum(struct ip6_hdr *ip6h, const void *l4_hdr,
        uint32_t exthdrlen, uint8_t l4_proto)
{
    uint16_t csum;
    uint8_t ip6nxt = ip6h->ip6_nxt;
    uint32_t ip6plen = ip6h->ip6_plen;
    struct in6_addr ip6dst = ip6h->ip6_dst;

    ip6h->ip6_nxt = l4_proto;

    /* length of L4 header plus L4 data */
    ip6h->ip6_plen = htons(ntohs(ip6h->ip6_plen) +
            sizeof(struct ip6_hdr) - exthdrlen);

    /* ip6_dst translation for NEXTHDR_ROUTING exthdrs */
    if (unlikely(ip6nxt == NEXTHDR_ROUTING)) {
        struct ip6_rthdr0 *rh = (struct ip6_rthdr0 *)(ip6h + 1);
        if (likely(rh->ip6r0_segleft > 0))
            ip6h->ip6_dst = rh->ip6r0_addr[rh->ip6r0_segleft - 1];
    }
    /*FIXME: what if NEXTHDR_ROUTING is not the first exthdr? */

    csum = rte_ipv6_udptcp_cksum((struct rte_ipv6_hdr *)ip6h, l4_hdr);

    /* restore original ip6h header */
    ip6h->ip6_nxt = ip6nxt;
    ip6h->ip6_plen = ip6plen;
    if (unlikely(ip6nxt == NEXTHDR_ROUTING))
        ip6h->ip6_dst = ip6dst;

    return csum;
}
