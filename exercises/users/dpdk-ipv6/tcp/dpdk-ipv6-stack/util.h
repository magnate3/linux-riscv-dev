#ifndef DPDK_UTIL_H
#define DPDK_UTIL_H
#include <stdint.h>
#include <netinet/icmp6.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <netinet/ip_icmp.h>
#include <rte_ethdev.h>
#define ipaddr_eq(addr0, addr1) (memcmp((const void*)(addr0), (const void*)addr1, sizeof(struct in6_addr)) == 0)
#define ETHER_ADDR_FMT "%02x:%02x:%02x:%02x:%02x:%02x"
#define RTE_ETHER_ADDR_STR(eth_addr) \
                 (eth_addr)->addr_bytes[0],\
                 (eth_addr)->addr_bytes[1], \
                 (eth_addr)->addr_bytes[2],\
                 (eth_addr)->addr_bytes[3],\
                 (eth_addr)->addr_bytes[4],\
                 (eth_addr)->addr_bytes[5]
void ip6_dump_hdr(const struct ip6_hdr * header);
void dump_icmp6(const struct icmp6_hdr *icmp6);
void ether_addr_dump(const char *what, const struct rte_ether_addr *ea);
void print_mac(unsigned int port_id);
int init_mbuf(struct rte_mempool * pool);
struct rte_mempool * get_mbufpool(uint8_t portid);
struct rte_mbuf *send_arp(struct rte_mempool *mbuf_pool, uint8_t *src_mac,uint8_t *dst_mac, uint32_t sip, uint32_t dip);
void ip6_dump_dpdk_frag(struct rte_mbuf *mbuf);
inline void net_copy_ip6(void *to, const void *from);
struct rte_mbuf * get_mbuf(void);
void dump_rte_eth_hdr(const struct rte_ether_hdr *eth);
#endif
