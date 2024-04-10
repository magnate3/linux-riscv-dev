#include "dpdk_tcp.h"

void dpdk_dump_tcph(struct rte_tcp_hdr *tcp_hdr, unsigned int l4_len)
{
     unsigned int          tcp_hdr_len;
     tcp_hdr_len = (tcp_hdr->data_off >> 4) << 2;
     printf("sport=%u, dport=%u, hdrlen=%u, flags=%c%c%c%c%c%c, data_len=%u",
              rte_be_to_cpu_16(tcp_hdr->src_port),
              rte_be_to_cpu_16(tcp_hdr->dst_port),
              tcp_hdr_len,
              (tcp_hdr->tcp_flags & RTE_TCP_URG_FLAG) == 0 ? '-' : 'u',
              (tcp_hdr->tcp_flags & RTE_TCP_ACK_FLAG) == 0 ? '-' : 'a',
              (tcp_hdr->tcp_flags & RTE_TCP_PSH_FLAG) == 0 ? '-' : 'p',
              (tcp_hdr->tcp_flags & RTE_TCP_RST_FLAG) == 0 ? '-' : 'r',
              (tcp_hdr->tcp_flags & RTE_TCP_SYN_FLAG) == 0 ? '-' : 's',
              (tcp_hdr->tcp_flags & RTE_TCP_FIN_FLAG) == 0 ? '-' : 'f',
              l4_len - tcp_hdr_len);

    printf("  seq=%u, ack=%u, window=%u, urgent=%u \n",
              rte_be_to_cpu_32(tcp_hdr->sent_seq),
              rte_be_to_cpu_32(tcp_hdr->recv_ack),
              rte_be_to_cpu_16(tcp_hdr->rx_win),
              rte_be_to_cpu_16(tcp_hdr->tcp_urp));

}
#if USE_TCP_DPDK
inline struct rte_tcp_hdr*tcp_hdr(const struct rte_mbuf *mbuf)
{
    int iphdrlen;
    unsigned char version, *verp;

    verp = rte_pktmbuf_mtod(mbuf, unsigned char*);
    version = (*verp >> 4) & 0xf;

    if (4 == version) {
        iphdrlen = ip4_hdrlen(mbuf);
    } else if (6 == version) {
        struct ip6_hdr *ip6h = ip6_hdr(mbuf);
        uint8_t ip6nxt = ip6h->ip6_nxt;
        iphdrlen = ip6_skip_exthdr(mbuf, sizeof(struct ip6_hdr), &ip6nxt);
        if (iphdrlen < 0)
            return NULL;
    } else {
        return NULL;
    }

    /* do not support frags */
    if (unlikely(mbuf->data_len < iphdrlen + sizeof(struct tcphdr)))
        return NULL;

    return rte_pktmbuf_mtod_offset(mbuf, struct rte_tcp_hdr*, iphdrlen);
}
#else
inline struct tcphdr *tcp_hdr(const struct rte_mbuf *mbuf)
{
    int iphdrlen;
    unsigned char version, *verp;

    verp = rte_pktmbuf_mtod(mbuf, unsigned char*);
    version = (*verp >> 4) & 0xf;

    if (4 == version) {
        iphdrlen = ip4_hdrlen(mbuf);
    } else if (6 == version) {
        struct ip6_hdr *ip6h = ip6_hdr(mbuf);
        uint8_t ip6nxt = ip6h->ip6_nxt;
        iphdrlen = ip6_skip_exthdr(mbuf, sizeof(struct ip6_hdr), &ip6nxt);
        if (iphdrlen < 0)
            return NULL;
    } else {
        return NULL;
    }

    /* do not support frags */
    if (unlikely(mbuf->data_len < iphdrlen + sizeof(struct tcphdr)))
        return NULL;

    return rte_pktmbuf_mtod_offset(mbuf, struct tcphdr *, iphdrlen);
}
#endif
