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
