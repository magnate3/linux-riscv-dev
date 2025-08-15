#include "globals.h"

#include "rtt.h"

void parse_client(struct rte_mbuf *pkt)
{
  struct rte_ether_hdr *eth_hdr;
  struct rte_vlan_hdr *vlan_hdr;
  struct rte_ipv4_hdr *ip_hdr;
  struct rte_udp_hdr *udp_hdr;
  struct rte_tcp_hdr *tcp_hdr;
  struct rte_icmp_hdr *icmp_hdr;
  uint16_t eth_type, offset = 0;
  uint16_t next_proto;

  float ms_delay;

  // int l2_len; //  in case of VLAN we need to add it to l2_len

  eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
  eth_type = rte_cpu_to_be_16(eth_hdr->ether_type);

  if (eth_type == RTE_ETHER_TYPE_VLAN)
  {
    vlan_hdr = (struct rte_vlan_hdr *)((unsigned char *)(eth_hdr + 1));
    eth_type = rte_cpu_to_be_16(vlan_hdr->eth_proto);
    offset += sizeof(struct rte_vlan_hdr);
  }
  if (eth_type == RTE_ETHER_TYPE_IPV4)
  {
    ip_hdr = (struct rte_ipv4_hdr *)((unsigned char *)(eth_hdr + 1) + offset);
    // extract ip features
    // IP Check

    next_proto = ip_hdr->next_proto_id;

    switch (next_proto)
    {
    // case IPPROTO_UDP:
    // {
    //   udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
    //   // extract the info required port info, etc
    // }
    // case IPPROTO_TCP:
    // {
    //   tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);
    //   // extract the info required port info, etc
    // }
    case IPPROTO_ICMP:
    {
      icmp_hdr = (struct rte_icmp_hdr *)(ip_hdr + 1);
      int seq_num = rte_be_to_cpu_16(icmp_hdr->icmp_seq_nb);

      if (icmp_hdr->icmp_type == RTE_IP_ICMP_ECHO_REPLY)
      {
        icmp_hdr->icmp_type = RTE_IP_ICMP_ECHO_REQUEST;
        icmp_hdr->icmp_code = 0;
        icmp_hdr->icmp_cksum = ck_sum((unaligned_uint16_t *)(icmp_hdr), sizeof(*icmp_hdr)); //

        int status = rtt(&seq_num, 0, &ms_delay);
      }

      printf("icmp_seq=%d rtt=%.1f ms\n", seq_num, ms_delay);

      seq_nb++;
      icmp_hdr->icmp_seq_nb = seq_nb;

      break;
    }

      rte_be32_t temp_ip;

      temp_ip = ip_hdr->dst_addr;
      ip_hdr->dst_addr = ip_hdr->src_addr;
      ip_hdr->src_addr = temp_ip;
    }
  }
}

void parser_server(struct rte_mbuf *pkt)
{
  struct rte_ether_hdr *eth_hdr;
  struct rte_vlan_hdr *vlan_hdr;
  struct rte_ipv4_hdr *ip_hdr;
  struct rte_udp_hdr *udp_hdr;
  struct rte_tcp_hdr *tcp_hdr;
  struct rte_icmp_hdr *icmp_hdr;
  uint16_t eth_type, offset = 0;
  uint16_t next_proto;

  eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
  eth_type = rte_cpu_to_be_16(eth_hdr->ether_type);

  if (eth_type == RTE_ETHER_TYPE_VLAN)
  {
    vlan_hdr = (struct rte_vlan_hdr *)((unsigned char *)(eth_hdr)); // eth + 1?
    eth_type = rte_cpu_to_be_16(vlan_hdr->eth_proto);
    offset += sizeof(struct rte_vlan_hdr);
  }
  if (eth_type == RTE_ETHER_TYPE_IPV4)
  {
    ip_hdr = (struct rte_ipv4_hdr *)((unsigned char *)(eth_hdr) + offset); // eth + 1?
    // extract ip features
    // ip

    rte_be32_t temp_ip;

    temp_ip = ip_hdr->dst_addr;
    ip_hdr->dst_addr = ip_hdr->src_addr;
    ip_hdr->src_addr = temp_ip;

    next_proto = ip_hdr->next_proto_id;

    switch (next_proto)
    {
    // case IPPROTO_UDP:
    // {
    //   udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
    //   // extract the info required port info, etc
    // }
    // case IPPROTO_TCP:
    // {
    //   tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);
    //   // extract the info required port info, etc
    // }
    case IPPROTO_ICMP:
    {
      icmp_hdr = (struct rte_icmp_hdr *)(ip_hdr + 1);

      if (icmp_hdr->icmp_type == RTE_IP_ICMP_ECHO_REQUEST)
      {
        icmp_hdr->icmp_type = RTE_IP_ICMP_ECHO_REPLY;
        icmp_hdr->icmp_code = 0;
        icmp_hdr->icmp_cksum = ck_sum((unaligned_uint16_t *)(icmp_hdr), sizeof(*icmp_hdr));
      }

      break;
    }
    }
  }
}
