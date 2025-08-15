#include "globals.h"

uint16_t add_icmp(struct rte_mbuf *pkt)
{
  struct rte_icmp_hdr *icmp_hdr;

  icmp_hdr = (struct rte_icmp_hdr *)((unsigned char *)pkt + (l2_len + l3_len));

  // Consruct ICMP Request
  icmp_hdr->icmp_type = RTE_IP_ICMP_ECHO_REQUEST;
  icmp_hdr->icmp_code = 0;
  icmp_hdr->icmp_cksum = ck_sum(icmp_hdr, sizeof(struct rte_icmp_hdr))  ;          // rte_cpu_to_be
      icmp_hdr->icmp_ident = src_id; // generate randomly
  icmp_hdr->icmp_seq_nb = seq_nb;

  return (uint16_t)(sizeof(struct rte_icmp_hdr) + data_size);
}
