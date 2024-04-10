#include "util.h"
#include<stdio.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_mempool.h>

#include <sys/socket.h>   
#include <netinet/in.h>   
#include <arpa/inet.h>  
inline void net_copy_ip6(void *to, const void *from)
{
	memcpy((void *)to, from, sizeof(struct in6_addr));
}
void dump_icmp6(struct icmp6_hdr *icmp6) {
  printf("------------------ ICMP6\n");
  printf("type = %x", icmp6->icmp6_type);
  if (icmp6->icmp6_type == ICMP6_ECHO_REQUEST) {
    printf("(echo request)");
  } else if (icmp6->icmp6_type == ICMP6_ECHO_REPLY) {
    printf("(echo reply)");
  }
  printf("\n");
  printf("code = %x\n", icmp6->icmp6_code);

  printf("checksum = %x\n", icmp6->icmp6_cksum);

  if ((icmp6->icmp6_type == ICMP6_ECHO_REQUEST) || (icmp6->icmp6_type == ICMP6_ECHO_REPLY)) {
    printf("icmp6_id = %x\n", icmp6->icmp6_id);
    printf("icmp6_seq = %x\n", icmp6->icmp6_seq);
  }
}
void ip6_dump_hdr(const struct ip6_hdr * header)
{
  char addrstr[INET6_ADDRSTRLEN];
  const struct ip6_frag * frag;
  uint8_t next_hdr;
  printf("****************** ipv6\n");
  printf("version = %x\n", header->ip6_vfc >> 4);
  printf("traffic class = %x\n", header->ip6_flow >> 20);
  printf("flow label = %x\n", ntohl(header->ip6_flow & 0x000fffff));
  printf("payload len = %x\n", ntohs(header->ip6_plen));
  printf("next header = %x\n", header->ip6_nxt);
  printf("hop limit = %x\n", header->ip6_hlim);
  next_hdr = header->ip6_nxt;
  if (next_hdr == IPPROTO_FRAGMENT)
  {
        frag = (const struct ip6_frag*) (header + 1);
        next_hdr = frag->ip6f_nxt;
  }
  inet_ntop(AF_INET6, &header->ip6_src, addrstr, sizeof(addrstr));
  printf("source = %s --> \n", addrstr);
  inet_ntop(AF_INET6, &header->ip6_dst, addrstr, sizeof(addrstr));
  printf("dest = %s\n", addrstr);
  printf("(%hu bytes) (next header: %hhx)\n", ntohs(header->ip6_plen), next_hdr);
  return;
}
void print_mac(unsigned int port_id)
{
    struct rte_ether_addr dev_eth_addr;
    rte_eth_macaddr_get(port_id, &dev_eth_addr);
    printf("port id %d MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n\n",
        (unsigned int) port_id,
        dev_eth_addr.addr_bytes[0],
        dev_eth_addr.addr_bytes[1],
        dev_eth_addr.addr_bytes[2],
        dev_eth_addr.addr_bytes[3],
        dev_eth_addr.addr_bytes[4],
        dev_eth_addr.addr_bytes[5]);
}
void ether_addr_dump(const char *what, const struct rte_ether_addr *ea)
{
	char buf[RTE_ETHER_ADDR_FMT_SIZE];

	rte_ether_format_addr(buf, RTE_ETHER_ADDR_FMT_SIZE, ea);
	if (what)
		printf("%s", what);
	printf("%s \n", buf);
}
struct rte_mempool * fwd_pktmbuf_pool = NULL;
int init_mbuf(struct rte_mempool * pool)
{
      fwd_pktmbuf_pool = pool; 
      return 0;
}
struct rte_mempool * get_mbufpool(uint8_t portid)
{
      return fwd_pktmbuf_pool ; 
}
struct rte_mbuf * get_mbuf(void)
{
   struct rte_mbuf *mbuf = NULL;   
   mbuf = rte_pktmbuf_alloc(fwd_pktmbuf_pool);
   //printf("mbuf addr %p and phy addr %p, and next %p \n",mbuf,rte_mem_virt2phy(mbuf), mbuf->next);
   return mbuf;
}

static int encode_arp_pkt(uint8_t *msg, uint8_t *src_mac, uint8_t *dst_mac, uint32_t sip, uint32_t dip) {

	// 1 ethhdr
	struct rte_ether_hdr *eth = (struct rte_ether_hdr *)msg;
	rte_memcpy(eth->s_addr.addr_bytes, src_mac, RTE_ETHER_ADDR_LEN);
	rte_memcpy(eth->d_addr.addr_bytes, dst_mac, RTE_ETHER_ADDR_LEN);
	eth->ether_type = htons(RTE_ETHER_TYPE_ARP);

	// 2 arp 
	struct rte_arp_hdr *arp = (struct rte_arp_hdr *)(eth + 1);
	arp->arp_hardware = htons(1);
	arp->arp_protocol = htons(RTE_ETHER_TYPE_IPV4);
	arp->arp_hlen = RTE_ETHER_ADDR_LEN;
	arp->arp_plen = sizeof(uint32_t);
	arp->arp_opcode = htons(2);

	rte_memcpy(arp->arp_data.arp_sha.addr_bytes, src_mac, RTE_ETHER_ADDR_LEN);
	rte_memcpy( arp->arp_data.arp_tha.addr_bytes, dst_mac, RTE_ETHER_ADDR_LEN);

	arp->arp_data.arp_sip = sip;
	arp->arp_data.arp_tip = dip;
	
	return 0;

}

struct rte_mbuf *send_arp(struct rte_mempool *mbuf_pool, uint8_t *src_mac,uint8_t *dst_mac, uint32_t sip, uint32_t dip) {

	const unsigned total_length = sizeof(struct rte_ether_hdr) + sizeof(struct rte_arp_hdr);

	struct rte_mbuf *mbuf = rte_pktmbuf_alloc(mbuf_pool);
	if (!mbuf) {
		rte_exit(EXIT_FAILURE, "rte_pktmbuf_alloc\n");
	}

	mbuf->pkt_len = total_length;
	mbuf->data_len = total_length;

	uint8_t *pkt_data = rte_pktmbuf_mtod(mbuf, uint8_t *);
	encode_arp_pkt(pkt_data, src_mac,dst_mac, sip, dip);

	return mbuf;
}
