#include "dpdk_ipv4.h"
static int ipv4_local_in_fin(struct rte_mbuf *mbuf);
static int ipv4_local_in(struct rte_mbuf *mbuf)
{
    //int err;

    if (ip4_is_frag(ip4_hdr(mbuf))) {
#if 0
        if ((err = ip4_defrag(mbuf, IP_DEFRAG_LOCAL_IN)) != EDPVS_OK) {
            return err;
        }
#endif
    }

   ipv4_local_in_fin(mbuf);
}
static int ipv4_local_in_fin(struct rte_mbuf *mbuf)
{
     struct rte_ipv4_hdr *iph = ip4_hdr(mbuf);
     int err, hlen;    
     /* remove network header */
    hlen = ip4_hdrlen(mbuf);
    rte_pktmbuf_adj(mbuf, hlen);
    return err;
}

static void ip_format_addr(char *buf, uint16_t size,const uint32_t ip_addr)
{
    snprintf(buf, size, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8 ,
             (uint8_t)((ip_addr >> 24) & 0xff),
             (uint8_t)((ip_addr >> 16) & 0xff),
             (uint8_t)((ip_addr >> 8) & 0xff),
             (uint8_t)((ip_addr)&0xff));
}
void dpdk_dump_iph(const struct rte_ipv4_hdr *ip_hdr)
{
    char buf[IP_BUF_LEN] = {0};
    ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(ip_hdr->src_addr)); 
    printf("src ip : %s, ",buf);
    memset(buf,IP_BUF_LEN,0);
    ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(ip_hdr->dst_addr)); 
    printf("dst ip : %s ",buf);
    printf("\tip total len %u \n", rte_be_to_cpu_16(ip_hdr->total_length));
}
