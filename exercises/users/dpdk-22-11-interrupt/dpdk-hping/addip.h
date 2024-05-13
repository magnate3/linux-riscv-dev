#include <stdlib.h>

#include "globals.h"

#define IP_DEFTTL 64

void add_ip(struct rte_mbuf *pkt, uint32_t client_ip, uint32_t server_ip)
{
  struct rte_ipv4_hdr *ip_hdr;

  ip_hdr = rte_pktmbuf_mtod_offset(pkt, (struct rte_ipv4_hdr *), sizeof(struct rte_ether_hdr));
  ip_hdr->version_ihl = RTE_IPV4_VHL_DEF;
  ip_hdr->type_of_service = 0;
  ip_hdr->fragment_offset = 0;
  ip_hdr->time_to_live = IP_DEFTTL;     // need to check for macro definition
  ip_hdr->next_proto_id = IPPROTO_ICMP; /// based on next proto
  ip_hdr->src_addr = rte_cpu_to_be_32(client_ip);
  ip_hdr->dst_addr = rte_cpu_to_be_32(server_ip);

  ip_hdr->packet_id = (src_id == -1) ? rte_cpu_to_be_32((unsigned short)rand()) : htons((unsigned short)src_id);

  uint16_t ip_packet_size = (uint16_t)data_size + sizeof(struct rte_ipv4_hdr);

  l3_len = sizeof(struct rte_ipv4_hdr);

  if (opt_icmp_mode)
  {
    ip_packet_size += add_icmp(pkt);
  }
  // else if (opt_tcp_mode)
  // {
  //   ip_packet_size += sizeof(struct rte_tcp_hdr);
  //   ip_hdr->total_length = rte_cpu_to_be_16(ip_packet_size);
  //   add_tcp();
  // }
  // else if (opt_udp_mode)
  // {
  //   ip_packet_size += sizeof(struct rte_udp_hdr);
  //   ip_hdr->total_length = rte_cpu_to_be_16(ip_packet_size);
  //   add_udp();
  // }
  // ip_hdr->hdr_checksum = ip_sum((unaligned_uint16_t *)ip_hdr,sizeof(*ipv4_hdr)); possible that this function might be correct

  ip_hdr->total_length = rte_cpu_to_be_16(ip_packet_size);
  ip_hdr->hdr_checksum = rte_ipv4_cksum(ip_hdr);
}