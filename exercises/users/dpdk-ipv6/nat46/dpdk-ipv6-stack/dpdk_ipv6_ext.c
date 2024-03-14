#include <netinet/ip6.h>
#include "dpdk_ipv6.h"
#include "dpdk_mbuf.h"
static int ip6_ext_hdr(__u8 nexthdr);
int ip6_skip_exthdr(const struct rte_mbuf *imbuf, int start, __u8 *nexthdrp)
{
    __u8 nexthdr = *nexthdrp;

    while (ip6_ext_hdr(nexthdr)) {
        struct ip6_ext _hdr, *hp;
        int hdrlen;

        if (nexthdr == NEXTHDR_NONE)
            return -1;
        hp = mbuf_header_pointer(imbuf, start, sizeof(_hdr), &_hdr);
        if (hp == NULL)
            return -1;
        if (nexthdr == NEXTHDR_FRAGMENT) {
            __be16 _frag_off, *fp;
            fp = mbuf_header_pointer(imbuf,
                        start + offsetof(struct ip6_frag, ip6f_offlg),
                        sizeof(_frag_off),
                        &_frag_off);
            if (fp == NULL)
                return -1;

            if (ntohs(*fp) & ~0x7)
                break;
            hdrlen = 8;
        } else if (nexthdr == NEXTHDR_AUTH)
            hdrlen = (hp->ip6e_len + 2) << 2;
        else
            hdrlen = ((hp)->ip6e_len + 1) << 3;

        nexthdr = hp->ip6e_nxt;
        start += hdrlen;
    }

    *nexthdrp = nexthdr;
    return start;
}
static int ip6_ext_hdr(__u8 nexthdr)
{
    /*
 *      * find out if nexthdr is an extension header or a protocol
 *           */
    return ( (nexthdr == NEXTHDR_HOP)   ||
         (nexthdr == NEXTHDR_ROUTING)   ||
         (nexthdr == NEXTHDR_FRAGMENT)  ||
         (nexthdr == NEXTHDR_AUTH)  ||
         (nexthdr == NEXTHDR_NONE)  ||
         (nexthdr == NEXTHDR_DEST) );
}
