#ifndef __DPDK_NDISC_H__
#define __DPDK_NDISC_H__

#include "dpdk_eth.h"
#include "dpdk_common.h"
//#include "neigh.h"
#define RTE_LOGTYPE_NEIGHBOUR RTE_LOGTYPE_USER1
/* Neigbour Discovery option types */
enum ND_OPT {
	__ND_OPT_PREFIX_INFO_END	= 0,
	ND_OPT_SOURCE_LL_ADDR		= 1,
	ND_OPT_TARGET_LL_ADDR		= 2,
	ND_OPT_PREFIX_INFO		= 3,
        ND_OPT_REDIRECT_HDR		= 4,
        //ND_OPT_MTU			= 5,
	__ND_OPT_MAX
};
#define ND_TTL              255
#define IPV6_NDISC_HOPLIMIT             255
#define IPV6_NDISC_ROUTER_SOLICITATION		133
#define IPV6_NDISC_ROUTER_ADVERTISEMENT		134
#define IPV6_NDISC_NEIGHBOUR_SOLICITATION	135
#define IPV6_NDISC_NEIGHBOUR_ADVERTISEMENT	136
#define IPV6_NDISC_REDIRECT			137
#define ZERO_IPV6_ADDR { { { 0x00, 0x00, 0x00, 0x00, \
			  0x00, 0x00, 0x00, 0x00, \
			  0x00, 0x00, 0x00, 0x00, \
			  0x00, 0x00, 0x00, 0x00 } } }
#define IN6ADDRSZ	sizeof(struct in6_addr)
#define INETHADDRSZ  (sizeof(struct rte_ether_addr))
int string_to_ip6(const char * ip6str ,struct in6_addr *result);
int ip6_is_our_addr(struct in6_addr *addr);
int ndisc_rcv(struct rte_mbuf *mbuf, struct netif_port *dev, uint16_t ip_offfset, uint16_t icmp6_offfse);

void ndisc_send_dad(struct netif_port *dev,
                    const struct in6_addr* solicit);

//void ndisc_solicit(struct neighbour_entry *neigh,
//                   const struct in6_addr *saddr);

int init_ndisc(void);
void icmp6_echo_request(void);
void icmp6_ns_request(void);
#endif /* __DPVS_NDISC_H__ */
