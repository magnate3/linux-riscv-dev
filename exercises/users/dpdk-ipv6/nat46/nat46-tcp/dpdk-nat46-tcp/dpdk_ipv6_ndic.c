#include <rte_ether.h>
#include <netinet/in.h>
#include <netinet/icmp6.h>
#include <netinet/ip6.h>
#include <linux/if_addr.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <linux/if_ether.h> 
#include "conf/common.h"
#include "dpdk_ipv6_ndic.h"
#include "dpdk_icmp6.h"
#include "dpdk_ipv6.h"
#include "dpdk_eth.h"
#include "dpdk_mbuf.h"
#include "util.h"
//#include "checksum.h"
#include "dpdk.h"

#define NDISC_OPT_SPACE(len) (((len)+2+7)&~7)
#define IP6_HDR_SIZE (sizeof(struct ip6_hdr))
#define IP6_NDISC_OPT_SPACE(len) (((len) + 2 + 7) & ~7)
typedef unsigned char uchar;
struct in6_addr net_ip6 = ZERO_IPV6_ADDR;
struct in6_addr net_link_local_ip6 = ZERO_IPV6_ADDR;
struct in6_addr gw_ip6 = ZERO_IPV6_ADDR;
struct in6_addr gw_multicast_ip6 = ZERO_IPV6_ADDR;
static void icmp6_multicast_ip6_addr(const struct in6_addr *unicast, struct in6_addr *multicast);
//44:A1:91:A4:9C:0B
struct rte_ether_addr gw_mac=
    {{0x0, 0x0, 0x0, 0x0, 0x0, 0x0}};
struct rte_ether_addr net_ethaddr=
    {{0x44, 0xA1, 0x91, 0xA4, 0x9c, 0x0B}};
struct nd_msg {
    struct icmp6_hdr    icmph;
    struct in6_addr    target;
    uint8_t            opt[0];
};

struct icmp6_nd_opt {
    uint8_t  type;
    uint8_t  len;
    struct eth_addr mac;
} __attribute__((__packed__));
#if 0
const char *ip6str = "fe80::4a57:2ff:fe64:e7a7";
const char *gwip6str= "fe80::4a57:2ff:fe64:e7ae";
#else
const char *ip6str = "2008:db8::a0a:6751";
const char *gwip6str= "2008:db8::a0a:6752";
#endif
// struct in6_addr result;
int string_to_ip6(const char * ip6str ,struct in6_addr *result)
{
    if (inet_pton(AF_INET6, ip6str, result) > 0) // success!
    {
        return 0;
    }
    return -1;
}
int ip6_is_our_addr(struct in6_addr *addr)
{
	return !memcmp(addr, &net_link_local_ip6, sizeof(struct in6_addr)) ||
	       !memcmp(addr, &net_ip6, sizeof(struct in6_addr));
}
/**
 *  * ndisc_has_option() - Check if the ND packet has the specified option set
 *  *
 *  * @ip6:	pointer to IPv6 header
 *  * @type:	option type to check
 *  * Return: 1 if ND has that option, 0 therwise
 *  */
/*
static bool ndisc_has_option(struct ip6_hdr *ip6, __u8 type)
{
	struct nd_msg *ndisc = (struct nd_msg *)(((unsigned char *)ip6) + IP6_HDR_SIZE);

	if (ip6->payload_len <= sizeof(struct icmp6_hdr))
		return 0;

	return ndisc->opt[0] == type;
}
*/
static bool ndisc_has_option(struct nd_msg *ndisc, uint8_t type)
{
	return ndisc->opt[0] == type;
}
static void ndisc_extract_enetaddr(struct nd_msg *ndisc, uchar enetaddr[6])
{
	memcpy(enetaddr, &ndisc->opt[2], 6);
}
static int
ndisc_insert_option(struct nd_msg *ndisc, int type,  uint8_t *data, int len)
{
	int space = IP6_NDISC_OPT_SPACE(len);

	ndisc->opt[0] = type;
	ndisc->opt[1] = space >> 3;
	memcpy(&ndisc->opt[2], data, len);
	len += 2;

	/* fill the remainder with 0 */
	if (space - len > 0)
		memset(&ndisc->opt[len], '\0', space - len);

	return space;
}
int ip6_add_hdr(struct ip6_hdr *ip6, struct in6_addr *src, struct in6_addr *dest,
		int nextheader, int hoplimit, int payload_len)
{
	//struct ip6_hdr *ip6 = (struct ip6_hdr *)xip;

	//ip6->version = 6;
	//ip6->priority = 0;
	//ip6->flow_lbl[0] = 0;
	//ip6->flow_lbl[1] = 0;
	//ip6->flow_lbl[2] = 0;
	//ip6->payload_len = htons(payload_len);
	//ip6->nexthdr = nextheader;
	//ip6->hop_limit = hoplimit;
        //ip6->ip6_hlim =  hoplimit;
        //printf("dump recv ipv6 _____________________");
        //ip6_dump_hdr(ip6);  
        // dest = old ip6->ip6_src,first set ip6->ip6_dst
	net_copy_ip6(&ip6->ip6_dst, dest);
	net_copy_ip6(&ip6->ip6_src, src);
        ///printf("dump send ipv6 _____________________");
        ///ip6_dump_hdr(ip6);  
	return sizeof(struct ip6_hdr);
}
static void
ip6_send_na(struct rte_mbuf *mbuf,struct ip6_hdr * ip6,uchar *eth_dst_addr, struct in6_addr *neigh_addr,
	    struct in6_addr *target,struct nd_msg * msg)
{
	__u16 len;

#if 1
	len = sizeof(struct icmp6_hdr) + IN6ADDRSZ +
	    IP6_NDISC_OPT_SPACE(INETHADDRSZ);

	//pkt = (uchar *)net_tx_packet;
	//pkt += net_set_ether(pkt, eth_dst_addr, PROT_IP6);
	//pkt += ip6_add_hdr(pkt, &net_link_local_ip6, neigh_addr,
	//		   PROT_ICMPV6, IPV6_NDISC_HOPLIMIT, len);
	//ip6_add_hdr(ip6, &net_ip6, neigh_addr,
	//		   IPPROTO_ICMPV6, IPV6_NDISC_HOPLIMIT, len);
	ip6_add_hdr(ip6, target, neigh_addr,
			   IPPROTO_ICMPV6, IPV6_NDISC_HOPLIMIT, len);
#endif

	/* ICMPv6 - NA */
	//msg = (struct nd_msg *)pkt;
	msg->icmph.icmp6_type = IPV6_NDISC_NEIGHBOUR_ADVERTISEMENT;
	msg->icmph.icmp6_code = 0;
#if 0
	memset(&msg->icmph.icmp6_cksum, 0, sizeof(__be16));
	memset(&msg->icmph.icmp6_unused, 0, sizeof(__be32));
	msg->icmph.icmp6_dataun.u_nd_advt.solicited = 1;
	msg->icmph.icmp6_dataun.u_nd_advt.override = 1;
#else
        //msg->icmph.icmp6_dataun.icmp6_un_data16[0] = 1;
        //msg->icmph.icmp6_dataun.icmp6_un_data16[1] = 1;
        //na->icmph.icmp6_type = NDISC_NEIGHBOUR_ADVERTISEMENT;
        //msg->icmph.icmp6_router = 0;
        //msg->icmph.icmp6_override = 1;
        //msg->icmph.icmp6_solicited = 1;
        msg->icmph.icmp6_dataun.icmp6_un_data32[0] =  ND_NA_FLAG_SOLICITED ;
#endif
	/* Set the target address and lltargetaddr option */
	net_copy_ip6(&msg->target, target);
	ndisc_insert_option(msg, ND_OPT_TARGET_LL_ADDR, &net_ethaddr.addr_bytes,
			    INETHADDRSZ);
        icmp6_send_csum(ip6,&msg->icmph);
        // this is a bad code
        int ret = 0;
        struct rte_ether_addr eth_addr;
        ret = rte_eth_macaddr_get(DEFAULT_PORTID, &eth_addr);
        ether_addr_dump("  nic mac addr =", &eth_addr);
        struct rte_ether_hdr * eth_h = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);
        rte_ether_addr_copy(&eth_h->s_addr, &eth_h->d_addr);
        rte_ether_addr_copy(&eth_addr, &eth_h->s_addr);
        //ether_addr_dump("  src mac addr =", &eth_h->s_addr);
        //ether_addr_dump("  dst mac addr =", &eth_h->d_addr);

	/* send it! */
	//net_send_packet(net_tx_packet, (pkt - net_tx_packet));
        ipv6_xmit(mbuf);
       //struct  rte_mbuf * arp_pkt=  send_arp(get_mbufpool(0),eth_h->s_addr.addr_bytes, eth_h->d_addr.addr_bytes,12345,123455);
       //ipv6_xmit(arp_pkt);
}
static int ndisc_recv_ns(struct rte_mbuf *mbuf,struct ip6_hdr * ip6,struct nd_msg *ndisc, struct netif_port *dev)
{
    //printf("%s not implement \n",__func__);   
    uchar neigh_eth_addr[6];
    //struct nd_msg *ndisc = (struct nd_msg *)icmp;
    if (ip6_is_our_addr(&ndisc->target) &&
	   ndisc_has_option(ndisc, ND_OPT_SOURCE_LL_ADDR)) {
	   ndisc_extract_enetaddr(ndisc, neigh_eth_addr);
	   ip6_send_na(mbuf,ip6,neigh_eth_addr, &ip6->ip6_src, &ndisc->target,ndisc);
		}
    else
    {
        rte_pktmbuf_free(mbuf);
    }
    return EDPVS_OK;
}
static int ndisc_recv_na(struct rte_mbuf *mbuf,struct ip6_hdr *ip6h, struct netif_port *dev)
{
    //printf("%s not implement \n",__func__);   
    struct eth_hdr *eth = mbuf_eth_hdr(mbuf);
    if (ipaddr_eq(&gw_ip6, &ip6h->ip6_src)
            && eth_addr_is_zero(&gw_mac)) {
        //ip6_dump_hdr(ip6h);
        //ether_addr_dump("advert mac addr: ", &eth->s_addr);
        rte_ether_addr_copy(&eth->s_addr, &gw_mac);
    }
    rte_pktmbuf_free(mbuf);
    return EDPVS_OK;
}
int ndisc_rcv(struct rte_mbuf *mbuf, struct netif_port *dev,uint16_t ip_offfset, uint16_t icmp6_offfset)
{
    //printf("%s not implement \n",__func__);   
    struct nd_msg *msg;
    int ret;
#if 0
    struct ip6_hdr *ipv6_hdr = MBUF_USERDATA(mbuf, struct ip6_hdr *, MBUF_FIELD_PROTO);

    if (mbuf_may_pull(mbuf, sizeof(struct icmp6_hdr)) != 0) {
        ret = EDPVS_NOMEM;
        goto free;
    }

    msg = (struct nd_msg *)rte_pktmbuf_mtod(mbuf, struct nd_msg *);
#else
    struct ip6_hdr *ipv6_hdr;
    ipv6_hdr =  rte_pktmbuf_mtod_offset(mbuf, struct ip6_hdr *, ip_offfset);
    msg = rte_pktmbuf_mtod_offset(mbuf, struct nd_msg*,icmp6_offfset);
#endif
    if (ipv6_hdr->ip6_hlim != 255) {
        RTE_LOG(ERR, NEIGHBOUR, "[%s] invalid hop-limit\n", __func__);
        ret = EDPVS_INVAL;
        goto free;
    }

    if (msg->icmph.icmp6_code != 0) {
        RTE_LOG(ERR, NEIGHBOUR, "[%s] invalid ICMPv6_code:%d\n", __func__,
                msg->icmph.icmp6_code);
        ret = EDPVS_INVAL;
        goto free;
    }

    switch (msg->icmph.icmp6_type) {
    case ND_NEIGHBOR_SOLICIT:
        return ndisc_recv_ns(mbuf,ipv6_hdr, msg, dev);
        //break;
    case ND_NEIGHBOR_ADVERT:
        return ndisc_recv_na(mbuf,ipv6_hdr, dev);
        //break;

    /* not support yet */
    case ND_ROUTER_SOLICIT:
    case ND_ROUTER_ADVERT:
    case ND_REDIRECT:
        ret = EDPVS_KNICONTINUE;
        break;
    default:
        ret = EDPVS_KNICONTINUE;
        break;
    }

#if 0
    /* ipv6 handler should consume mbuf */
    if (ret != EDPVS_KNICONTINUE)
        goto free;

    return EDPVS_KNICONTINUE;
#endif
free:
    rte_pktmbuf_free(mbuf);
    return ret;
}

static void icmp6_multicast_ether_addr(const struct in6_addr *daddr, struct eth_addr *mac)
{
    struct in6_addr mcast_addr;

    mac->bytes[0] = 0x33;
    mac->bytes[1] = 0x33;
    icmp6_multicast_ip6_addr(daddr, &mcast_addr);
    memcpy(mac->bytes + 2, &mcast_addr.s6_addr32[3], sizeof(uint32_t));
}
static void icmp6_ns_eth_hdr_push(struct rte_mbuf *m)
{
    struct eth_addr dmac;
    struct eth_hdr *eth = NULL;
#if 1
    icmp6_multicast_ether_addr(&gw_ip6, &dmac);

    eth = mbuf_push_eth_hdr(m);
    eth_hdr_set(eth, ETHER_TYPE_IPv6, &dmac, &net_ethaddr.addr_bytes);
#endif
}

static void icmp6_multicast_ip6_addr(const struct in6_addr *unicast, struct in6_addr *multicast)
{
    memset(multicast, 0, sizeof(struct in6_addr));
    multicast->s6_addr[0] = 0xff;
    multicast->s6_addr[1] = 0x02;
    multicast->s6_addr[11] = 0x01;
    multicast->s6_addr[12] = 0xff;
    multicast->s6_addr[13] = unicast->s6_addr[13];
    multicast->s6_addr[14] = unicast->s6_addr[14];
    multicast->s6_addr[15] = unicast->s6_addr[15];
}
static void icmp6_ns_ip6_hdr_push(struct rte_mbuf *m)
{
    uint16_t plen = 0;
    struct ip6_hdr *ip6h = NULL;

    plen = sizeof(struct nd_neighbor_solicit) + sizeof(struct icmp6_nd_opt);

    ip6h = mbuf_push_ip6_hdr(m);
    memset(ip6h, 0, sizeof(struct ip6_hdr));
    ip6h->ip6_vfc = (6 << 4);
    ip6h->ip6_hops = ND_TTL;
    //ip6h->ip6_src = net_ip6;
    net_copy_ip6(&ip6h->ip6_src,&net_ip6);
    ip6h->ip6_plen = htons(plen);
    ip6h->ip6_nxt = IPPROTO_ICMPV6;

    icmp6_multicast_ip6_addr(&gw_ip6, &ip6h->ip6_dst);
}

static void icmp6_nd_opt_set(struct icmp6_nd_opt *opt, uint8_t type, const struct eth_addr *mac)
{
    opt->type = type;
    opt->len = 1;
    eth_addr_copy(&opt->mac, mac);
}
static void icmp6_ns_hdr_push(struct rte_mbuf *m)
{
    struct ip6_hdr *ip6h = NULL;
    struct icmp6_nd_opt *opt = NULL;
    struct nd_neighbor_solicit *ns = NULL;

    ip6h = mbuf_ip6_hdr(m);
    ns = RTE_PKTMBUF_PUSH(m, struct nd_neighbor_solicit);
    memset(ns, 0, sizeof(struct nd_neighbor_solicit));
    ns->nd_ns_type = ND_NEIGHBOR_SOLICIT;
    //ns->nd_ns_target = gw_ip6;
    net_copy_ip6(&ns->nd_ns_target,&gw_ip6);
    opt = RTE_PKTMBUF_PUSH(m, struct icmp6_nd_opt);
    icmp6_nd_opt_set(opt, ND_OPT_SOURCE_LINKADDR, &net_ethaddr);

    ns->nd_ns_cksum = 0;
    ns->nd_ns_cksum = RTE_IPV6_UDPTCP_CKSUM(ip6h, ns);
}

void icmp6_ns_request(void)
{
    struct rte_mbuf *mbuf = NULL;

    mbuf = get_mbuf();
    if (mbuf == NULL) {
        return;
    }

    //printf("%s, mbuf addr %p and phy addr %p, and next %p \n",__func__,mbuf,rte_mem_virt2phy(mbuf), mbuf->next);
    icmp6_ns_eth_hdr_push(mbuf);
    icmp6_ns_ip6_hdr_push(mbuf);
    icmp6_ns_hdr_push(mbuf);
    ipv6_xmit(mbuf);
}
static void icmp6_echo_hdr_push(struct rte_mbuf *m)
{
    int i = 0;
    struct icmp6_hdr *ich = RTE_PKTMBUF_PUSH(m, struct icmp6_hdr);
    struct ip6_hdr *ip6h = NULL;
    char * data;
    ip6h = mbuf_ip6_hdr(m);
    ich->icmp6_type = ICMP6_ECHO_REQUEST;
    ich->icmp6_code = 0;
    ich->icmp6_cksum = 0;
    ich->icmp6_data16[0]= htons(getpid());	/* Identifier */
    ich->icmp6_data16[1]= htons(random());	/* Sequence Number */
    data = (char*)rte_pktmbuf_append(m,ICMP_ECHO_DATALEN);
    for(;i < ICMP_ECHO_DATALEN; ++i)
        data[i] = 'a' + i;
    //ich->icmp6_cksum = icmp6_checksum(ip6h, ich, data, ICMP_ECHO_DATALEN);
    icmp6_send_csum(ip6h,ich);
    //RTE_IPV6_UDPTCP_CKSUM(ip6h, ich);
}
static void icmp6_echo_ip6_hdr_push(struct rte_mbuf *m)
{
    uint16_t plen = 0;
    struct ip6_hdr *ip6h = NULL;
#if 0
    plen = sizeof(struct nd_neighbor_solicit) + sizeof(struct icmp6_nd_opt);
#else
    plen = sizeof(struct icmp6_hdr) + ICMP_ECHO_DATALEN;
#endif
    ip6h = mbuf_push_ip6_hdr(m);
    memset(ip6h, 0, sizeof(struct ip6_hdr));
    ip6h->ip6_vfc = (6 << 4);
    ip6h->ip6_hops = ND_TTL;
    //ip6h->ip6_src = net_ip6;
    net_copy_ip6(&ip6h->ip6_src,&net_ip6);
    net_copy_ip6(&ip6h->ip6_dst,&gw_ip6);
    ip6h->ip6_plen = htons(plen);
    ip6h->ip6_nxt = IPPROTO_ICMPV6;

}
static void icmp6_echo_eth_hdr_push(struct rte_mbuf *m)
{
    //struct eth_addr dmac;
    struct eth_hdr *eth = NULL;
#if 1
    eth = mbuf_push_eth_hdr(m);
    eth_hdr_set(eth, ETHER_TYPE_IPv6, &gw_mac.addr_bytes, &net_ethaddr.addr_bytes);
#endif
}

void icmp6_echo_request(void)
{
    struct rte_mbuf *mbuf = NULL;
    if(eth_addr_is_zero(&gw_mac)) {
    printf("%s, gw mac is zero \n",__func__);
         return;
    }
    mbuf = get_mbuf();
    if (mbuf == NULL) {
        return;
    }

    //printf("%s, mbuf addr %p and phy addr %p, and next %p \n",__func__,mbuf,rte_mem_virt2phy(mbuf), mbuf->next);
    icmp6_echo_eth_hdr_push(mbuf);
    icmp6_echo_ip6_hdr_push(mbuf);
    icmp6_echo_hdr_push(mbuf);
    ipv6_xmit(mbuf);
}
int init_ndisc(void)
{
    string_to_ip6(gwip6str, &gw_ip6);
    string_to_ip6(ip6str, &net_ip6);
    rte_eth_macaddr_get(DEFAULT_PORTID, &net_ethaddr);
    return 0;
}
