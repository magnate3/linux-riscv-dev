#include <assert.h>
#include "dpdk_ipv6.h"
#include "dpdk_ipv6_ndic.h"
#include "dpdk_icmp6.h"
#include "dpdk_eth.h"
#include "dpdk_mbuf.h"
#include "dpdk_common.h"
#include "dpdk_nat46.h"
#include "util.h"

#define ICMP6
#define RTE_LOGTYPE_ICMP6    RTE_LOGTYPE_USER1

#ifdef  CONFIG_DPDK_ICMP_DEBUG
static void icmp6_dump_hdr(const struct rte_mbuf *mbuf)
{
    //struct icmp6_hdr *ich = rte_pktmbuf_mtod(mbuf, struct icmp6_hdr *);
    //lcoreid_t lcore = rte_lcore_id();

    //fprintf(stderr, "lcore %d port %d icmp type %u code %u\n",
    //        lcore, mbuf->port, ich->icmp_type, ich->icmp_code);

    return;
}
#endif

uint16_t icmp6_csum(struct ip6_hdr *iph, struct icmp6_hdr *ich)
{
    uint32_t csum, l4_len;
    struct ip6_hdr hdr;

    /* must be linear !! */
    l4_len = ntohs(iph->ip6_plen);
    if ((void *)ich != (void *)(iph + 1))
        l4_len -= (void *)ich - (void *)(iph+1);

    memset(&hdr, 0, sizeof(struct ip6_hdr));
    hdr.ip6_nxt     = IPPROTO_ICMPV6;
    hdr.ip6_plen    = htons(l4_len);
    hdr.ip6_src     = iph->ip6_src;
    hdr.ip6_dst     = iph->ip6_dst;
    csum = rte_raw_cksum(ich, l4_len);
#ifdef DPDK_IPV6_DEBUG
    csum += rte_ipv6_phdr_cksum((struct rte_ipv6_hdr *)&hdr, 0);
#else
    csum += rte_ipv6_phdr_cksum((struct ipv6_hdr *)&hdr, 0);
#endif
    csum = ((csum & 0xffff0000) >> 16) + (csum & 0xffff);
    csum = (~csum) & 0xffff;
    if (csum == 0)
        csum = 0xffff;

    return csum;
}

static uint16_t icmp6_csum_of_mul_segs(const struct rte_mbuf *m,struct ip6_hdr *iph, struct icmp6_hdr *ich)
{
    uint32_t csum=0, l4_len;
    uint16_t csum1=0;
    struct ip6_hdr hdr;
    int ret = 0;
    /* must be linear !! */
    l4_len = ntohs(iph->ip6_plen);
    if ((void *)ich != (void *)(iph + 1))
        l4_len -= (void *)ich - (void *)(iph+1);

    memset(&hdr, 0, sizeof(struct ip6_hdr));
    hdr.ip6_nxt     = IPPROTO_ICMPV6;
    hdr.ip6_plen    = htons(l4_len);
    hdr.ip6_src     = iph->ip6_src;
    hdr.ip6_dst     = iph->ip6_dst;
    ret = ipv6_raw_cksum_mbuf(m,(char*)ich - rte_pktmbuf_mtod(m,char*) ,l4_len, &csum1);
    if(unlikely(ret))
    {
        RTE_LOG(INFO, IPV6, "mbuf(%p) raw cksum error  : \n",m);
        return 0;
    }
    csum += csum1;
#if 0 
    if(likely(0 == rte_pktmbuf_linearize(m)))
    {
        csum2 = rte_raw_cksum(ich, l4_len);
    }
    RTE_LOG(INFO, IPV6, "mbuf(%p) cksum1 0x%x, cksum2 0x%x \n",m,csum1,csum2);
#endif
#ifdef DPDK_IPV6_DEBUG
    csum += rte_ipv6_phdr_cksum((struct rte_ipv6_hdr *)&hdr, 0);
#else
    csum += rte_ipv6_phdr_cksum((struct ipv6_hdr *)&hdr, 0);
#endif
    csum = ((csum & 0xffff0000) >> 16) + (csum & 0xffff);
    csum = (~csum) & 0xffff;
    if (csum == 0)
        csum = 0xffff;
    return csum;
}
void icmp6_send_csum(struct ip6_hdr *shdr, struct icmp6_hdr *ich)
{
    uint32_t csum, l4_len;

    ich->icmp6_cksum = 0;

    l4_len = ntohs(shdr->ip6_plen);

    csum = rte_raw_cksum(ich, l4_len);
#ifdef DPDK_IPV6_DEBUG
    csum += rte_ipv6_phdr_cksum((struct rte_ipv6_hdr *)shdr, 0);
#else
    csum += rte_ipv6_phdr_cksum((struct ipv6_hdr *)&shdr, 0);
#endif
    csum = ((csum & 0xffff0000) >> 16) + (csum & 0xffff);
    csum = (~csum) & 0xffff;
    if (csum == 0)
        csum = 0xffff;

    ich->icmp6_cksum = csum;
}

/*
 *  copy from kernel
 *  an inline helper for the "simple" if statement below
 *  checks if parameter problem report is caused by an
 *  unrecognized IPv6 option that has the Option Type
 *  highest-order two bits set to 10
 */

static bool icmp6_opt_unrec(struct rte_mbuf *imbuf, uint32_t offset)
{
    uint8_t *op, optval;

    op = mbuf_header_pointer(imbuf, offset, sizeof(optval), &optval);
    if (!op)
        return true;
    return (*op & 0xC0) == 0x80;
}

/*
 * Figure out, may we reply to this packet with icmp error.
 *
 * We do not reply, if:
 *  - it was icmp error message.
 *  - it is truncated, so that it is known, that protocol is ICMPV6
 *    (i.e. in the middle of some exthdr)
 *
 *  --ANK (980726)
 */

static int icmp6_is_ineligible(struct rte_mbuf *imbuf)
{
    int ptr = sizeof(ip6_hdr);
    __u8 nexthdr = ip6_hdr(imbuf)->ip6_nxt;

    if (mbuf_may_pull(imbuf, ptr) != 0) {
        return 1;
    }

    ptr = ip6_skip_exthdr(imbuf, ptr, &nexthdr);
    if (ptr < 0)
        return 0;

    if (nexthdr == IPPROTO_ICMPV6) {
        __u8 _type, *tp;
        tp = mbuf_header_pointer(imbuf,
                ptr + offsetof(struct icmp6_hdr, icmp6_type),
                sizeof(_type), &_type);
        if (tp == NULL ||
            !(*tp & ICMP6_INFOMSG_MASK))
            return 1;
    }
    return 0;
}
#ifdef DPDK_IPV6_DEBUG
/* @imbuf is input (original) IP packet to trigger ICMP. */
void icmp6_send(struct rte_mempool *mp,struct rte_mbuf *imbuf, int type, int code, uint32_t info)
{
    struct ip6_hdr *iph = ip6_hdr(imbuf);
    eth_type_t etype = imbuf->packet_type; /* FIXME: use other field ? */
    struct in6_addr *saddr = NULL;
    struct ip6_hdr shdr;                   /* IPv6 header for sending packet */
    struct rte_mbuf *mbuf;
    struct icmp6_hdr *ich;
    struct inet_ifaddr *ifa;
    int room, err;
    int addr_type = 0;

    ifa = inet_addr_ifa_get(AF_INET6, netif_port_get(imbuf->port),
                           (union inet_addr *)&iph->ip6_dst);
    if (ifa) {
        saddr = &iph->ip6_dst;
        inet_addr_ifa_put(ifa);
    }

    addr_type = ipv6_addr_type(&iph->ip6_dst);

    /*
     * when the original ipv6 dst is l2/l3 mcast, just deal ICMP6_PACKET_TOO_BIG and
     * ICMP6_PARAM_PROB's unrecognize IPv6 option.
     */
    if (addr_type & IPV6_ADDR_MULTICAST ||  etype != ETH_PKT_HOST) {
        if (type != ICMP6_PACKET_TOO_BIG &&
            !(type == ICMP6_PARAM_PROB &&
              code == ICMP6_PARAMPROB_OPTION &&
              (icmp6_opt_unrec(imbuf, info)))) {

                RTE_LOG(DEBUG, ICMP6,
                    "%s: l2 broadcast or l3 multicast don't support the error.\n",
                     __func__);
            return;
        }
        saddr = NULL;
    }

    addr_type = ipv6_addr_type(&iph->ip6_src);
    /*
     *  Must not send error if the source does not uniquely
     *  identify a single node (RFC2463 Section 2.4).
     *  We check unspecified / multicast addresses here,
     *  and anycast addresses will be checked later.
     */
    if ((addr_type == IPV6_ADDR_ANY) || (addr_type & IPV6_ADDR_MULTICAST)) {
        RTE_LOG(DEBUG, ICMP6, "icmpv6_send: addr_any/mcast source\n");
        return;
    }

    /*
     *  In icmp6_send, never answer to a ICMP packet except the type of ICMP6_INFOMSG_MASK.
     */
    if (icmp6_is_ineligible(imbuf)) {
        RTE_LOG(DEBUG, ICMP6, "icmpv6_send: no reply to icmp error\n");
        return;
    }

    memset(&shdr, 0, sizeof(struct ip6_hdr));
    shdr.ip6_nxt = IPPROTO_ICMPV6;
    shdr.ip6_dst = iph->ip6_src;

    if (saddr) {
        shdr.ip6_src =  *saddr;
    } else {
        //inet_addr_select(AF_INET6, fl6.fl6_oif,
        //                 (union inet_addr *)&fl6.fl6_daddr, fl6.fl6_scope,
        //                 (union inet_addr *)&fl6.fl6_saddr);
        //shdr.ip6_src = fl6.fl6_saddr;
    }

    mbuf = rte_pktmbuf_alloc(mp);
    if (!mbuf) {
        RTE_LOG(DEBUG, ICMP6, "%s: no memory.\n", __func__);
        return;
    }
    mbuf->userdata = NULL;
    assert(rte_pktmbuf_headroom(mbuf) >= 128); /* for L2/L3 */
    ich = (struct icmp6_hdr*)rte_pktmbuf_append(mbuf, sizeof(struct icmp6_hdr));;
    if (!ich) {
        RTE_LOG(DEBUG, ICMP6, "%s: no room in mbuf.\n", __func__);
        rte_pktmbuf_free(mbuf);
        return;
    }
    ich->icmp6_type = type;
    ich->icmp6_code = code;
    ich->icmp6_pptr = htonl(info);  //use icmp6_pptr for store

#ifndef DPDK_IPV6_DEBUG
    /* copy as much as we can without exceeding min-MTU */
    room = min_t(int, fl6.fl6_oif->mtu, IPV6_MIN_MTU);
#else
    /* copy as much as we can without exceeding min-MTU */
    room = min_t(int, 1400, IPV6_MIN_MTU);
#endif
    room -= sizeof(struct ip6_hdr);
    room -= sizeof(struct icmp6_hdr);
    room = min_t(int, imbuf->data_len, room);

    if (!rte_pktmbuf_append(mbuf, room)) {
        RTE_LOG(DEBUG, ICMP6, "%s: no room in mbuf.\n", __func__);
        rte_pktmbuf_free(mbuf);
        return;
    }

    mbuf_copy_bits(imbuf, 0, ich + 1, room);

    shdr.ip6_plen = room + sizeof(struct icmp6_hdr);
    icmp6_send_csum(&shdr, ich);
    if ((err = ipv6_xmit(mbuf)) != EDPVS_OK) {
        RTE_LOG(DEBUG, ICMP6, "%s: ipv6_xmit: fail\n",
                __func__);
    }
    return;
}

#else

/* @imbuf is input (original) IP packet to trigger ICMP. */
void icmp6_send(struct rte_mbuf *imbuf, int type, int code, uint32_t info)
{
    struct ip6_hdr *iph = ip6_hdr(imbuf);
    eth_type_t etype = imbuf->packet_type; /* FIXME: use other field ? */
    struct in6_addr *saddr = NULL;
    struct ip6_hdr shdr;                   /* IPv6 header for sending packet */
    struct rte_mbuf *mbuf;
    struct icmp6_hdr *ich;
    struct flow6 fl6;
    struct inet_ifaddr *ifa;
    int room, err;
    int addr_type = 0;

    ifa = inet_addr_ifa_get(AF_INET6, netif_port_get(imbuf->port),
                           (union inet_addr *)&iph->ip6_dst);
    if (ifa) {
        saddr = &iph->ip6_dst;
        inet_addr_ifa_put(ifa);
    }

    addr_type = ipv6_addr_type(&iph->ip6_dst);

    /*
     * when the original ipv6 dst is l2/l3 mcast, just deal ICMP6_PACKET_TOO_BIG and
     * ICMP6_PARAM_PROB's unrecognize IPv6 option.
     */
    if (addr_type & IPV6_ADDR_MULTICAST ||  etype != ETH_PKT_HOST) {
        if (type != ICMP6_PACKET_TOO_BIG &&
            !(type == ICMP6_PARAM_PROB &&
              code == ICMP6_PARAMPROB_OPTION &&
              (icmp6_opt_unrec(imbuf, info)))) {

                RTE_LOG(DEBUG, ICMP6,
                    "%s: l2 broadcast or l3 multicast don't support the error.\n",
                     __func__);
            return;
        }
        saddr = NULL;
    }

    addr_type = ipv6_addr_type(&iph->ip6_src);
    /*
     *  Must not send error if the source does not uniquely
     *  identify a single node (RFC2463 Section 2.4).
     *  We check unspecified / multicast addresses here,
     *  and anycast addresses will be checked later.
     */
    if ((addr_type == IPV6_ADDR_ANY) || (addr_type & IPV6_ADDR_MULTICAST)) {
        RTE_LOG(DEBUG, ICMP6, "icmpv6_send: addr_any/mcast source\n");
        return;
    }

    /*
     *  In icmp6_send, never answer to a ICMP packet except the type of ICMP6_INFOMSG_MASK.
     */
    if (icmp6_is_ineligible(imbuf)) {
        RTE_LOG(DEBUG, ICMP6, "icmpv6_send: no reply to icmp error\n");
        return;
    }

    memset(&shdr, 0, sizeof(struct ip6_hdr));
    memset(&fl6, 0, sizeof(fl6));
    shdr.ip6_nxt = IPPROTO_ICMPV6;
    shdr.ip6_dst = fl6.fl6_daddr = iph->ip6_src;

    fl6.fl6_proto = IPPROTO_ICMPV6;
    fl6.fl6_oif = netif_port_get(imbuf->port);
    if (saddr) {
        shdr.ip6_src = fl6.fl6_saddr = *saddr;
    } else {
        inet_addr_select(AF_INET6, fl6.fl6_oif,
                         (union inet_addr *)&fl6.fl6_daddr, fl6.fl6_scope,
                         (union inet_addr *)&fl6.fl6_saddr);
        shdr.ip6_src = fl6.fl6_saddr;
    }

    mbuf = rte_pktmbuf_alloc(fl6.fl6_oif->mbuf_pool);
    if (!mbuf) {
        RTE_LOG(DEBUG, ICMP6, "%s: no memory.\n", __func__);
        return;
    }
    mbuf_userdata_reset(mbuf);
    assert(rte_pktmbuf_headroom(mbuf) >= 128); /* for L2/L3 */
    ich = (struct icmp6_hdr*)rte_pktmbuf_append(mbuf, sizeof(struct icmp6_hdr));;
    if (!ich) {
        RTE_LOG(DEBUG, ICMP6, "%s: no room in mbuf.\n", __func__);
        rte_pktmbuf_free(mbuf);
        return;
    }
    ich->icmp6_type = type;
    ich->icmp6_code = code;
    ich->icmp6_pptr = htonl(info);  //use icmp6_pptr for store

    /* copy as much as we can without exceeding min-MTU */
    room = min_t(int, fl6.fl6_oif->mtu, IPV6_MIN_MTU);
    room -= sizeof(struct ip6_hdr);
    room -= sizeof(struct icmp6_hdr);
    room = min_t(int, imbuf->data_len, room);

    if (!rte_pktmbuf_append(mbuf, room)) {
        RTE_LOG(DEBUG, ICMP6, "%s: no room in mbuf.\n", __func__);
        rte_pktmbuf_free(mbuf);
        return;
    }

    mbuf_copy_bits(imbuf, 0, ich + 1, room);

    shdr.ip6_plen = htons(room + sizeof(struct icmp6_hdr));
    icmp6_send_csum(&shdr, ich);

    if ((err = ipv6_xmit(mbuf, &fl6)) != EDPVS_OK) {
        RTE_LOG(DEBUG, ICMP6, "%s: ipv6_xmit: %s.\n",
                __func__, dpvs_strerror(err));
    }
    return;
}
#endif
static uint16_t icmp_cksum(struct rte_ipv4_hdr *iphdr, struct rte_icmp_hdr *icmphdr)
{
    uint16_t *addr = (uint16_t *)(void *)icmphdr;
    int count = rte_be_to_cpu_16(iphdr->total_length) - sizeof(*iphdr);

    register long sum = 0;

    while (count > 1) {
        sum += *(unsigned short *)addr++;
        count -= 2;
    }

    if (count > 0)
        sum += *(unsigned char *)addr;

    while (sum >> 16)
        sum = (sum & 0xffff) + (sum >> 16);

    return ~sum;
}
static int icmp6_echo_reply_from_other(struct rte_mbuf *mbuf, struct ip6_hdr *iph,
                            struct icmp6_hdr *ich)
{
    if (!ip6_is_our_addr(&iph->ip6_dst))
    {
        
          goto free_mbuf;
    }
    printf("seq %u, id %u \n",ntohs(ich->icmp6_data16[1]), ntohs(ich->icmp6_data16[0]));
    rte_pktmbuf_free(mbuf);
    return EDPVS_OK;
free_mbuf:
    rte_pktmbuf_free(mbuf);
    return EDPVS_INVPKT;
}
static int icmp6_echo_reply_nat(struct rte_mbuf *mbuf, struct ip6_hdr *iph,
                            struct icmp6_hdr *ich)
{
    uint16_t icmp_len, data_len;
    struct rte_mbuf *dst_mbuf;
    struct  rte_ipv4_hdr* ip4h;
    char *data =NULL, *pktbuf=NULL; 
    struct rte_icmp_hdr * ic4h;
    //struct netif_port * dev;
    if (!ip6_is_our_addr(&iph->ip6_dst))
    {
        
          goto free_mbuf;
    }
    /* must be linear !! */
    icmp_len = ntohs(iph->ip6_plen);
    if ((void *)ich != (void *)(iph + 1))
        icmp_len -= (void *)ich - (void *)(iph+1);

    data_len = icmp_len - sizeof(struct icmp6_hdr);
    //printf("%s begin,thread id %d  mbuf addr %p , port %u\n",__func__,pthread_self(),mbuf, mbuf->port);
#if 1
    dst_mbuf = get_mbuf_from_netif(netif_port_get(mbuf->port));
#else
    dev = netif_port_get(mbuf->port); 
    dst_mbuf = rte_pktmbuf_alloc(dev->mbuf_pool);
#endif
    //printf("%s ,thread id %lu dst mbuf addr %p, mbuf addr %p , port %u\n",__func__,pthread_self(),dst_mbuf,mbuf, mbuf->port);
    if(!dst_mbuf)
    {
        goto free_mbuf;
    }
    pktbuf = rte_pktmbuf_mtod(dst_mbuf, char*);
    eth_hdr_push_for_ipv4(dst_mbuf); 
    ip4h = ipv4_hdr_push_common(mbuf,dst_mbuf,data_len +  sizeof(struct rte_icmp_hdr), IPPROTO_ICMP);
    //ip4h->next_proto_id = IPPROTO_ICMP;
    //////////////////////////////////////////////////
    ic4h = RTE_PKTMBUF_PUSH(dst_mbuf, struct rte_icmp_hdr); 
    ic4h->icmp_code = ich->icmp6_code;
    ic4h->icmp_ident = ich->icmp6_data16[0];
    ic4h->icmp_seq_nb = ich->icmp6_data16[1];
    ic4h->icmp_type = RTE_IP_ICMP_ECHO_REPLY;
    printf("icmp4_id = %x   ", ic4h->icmp_ident);
    printf("icmp4_seq = %x\n", ic4h->icmp_seq_nb);
    //data = rte_pktmbuf_mtod(dst_mbuf, char*); 
    //data = (char*)rte_pktmbuf_append(dst_mbuf, data_len); 
    if(NULL == (data =rte_pktmbuf_append(dst_mbuf, data_len)))
    {
         goto free_mbuf;
    }
    rte_memcpy(data,rte_pktmbuf_mtod_offset(mbuf, char *, mbuf->l2_len + mbuf->l3_len + sizeof(struct icmp6_hdr)),data_len);
     /*
     *计算整个 icmp 报文的长度(首部长度 + 数据部分长度)
     *icmplen = sizeof(struct icmp_echo) + strlen("hello");
     *所有字段填写完成后，就开始计算校验和
     *icmp_echo->icmp_cksum = cksum((unsigned short*)sendbuf, icmplen);
     */ 
    //csum = rte_raw_cksum(ich, sizeof(struct rte_icmp_hdr) + data_len);
    /* recalc the checksum */
    ic4h->icmp_cksum = 0;
#if 0
    csum = rte_raw_cksum(data,data_len+ sizeof(struct rte_icmp_hdr));
    ic4h->icmp_cksum = (csum == 0xffff) ? csum : ~csum;
#else
    ic4h->icmp_cksum = icmp_cksum(ip4h, ic4h); 
#endif
    //printf("%s mbuf addr %p , port %u\n",__func__,mbuf, mbuf->port);
    dump_pcap(netif_port_get(mbuf->port),(u_char*)pktbuf,dst_mbuf->pkt_len);
    eth_send_packets(&dst_mbuf,mbuf->port,0,1);
    rte_pktmbuf_free(mbuf);
    return EDPVS_OK;
free_mbuf:
    rte_pktmbuf_free(mbuf);
    return EDPVS_INVPKT; 
}
#ifdef DPDK_IPV6_DEBUG
extern struct in6_addr net_ip6 ;
extern struct in6_addr net_link_local_ip6;
static int icmp6_echo_reply(struct rte_mbuf *mbuf, struct ip6_hdr *iph,
                            struct icmp6_hdr *ich)
{
   struct rte_ether_addr eth_addr;
   struct rte_ether_hdr * eth_h = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);
#if 0
    struct ip6_hdr shdr; /* IPv6 header for sending packet */
    uint32_t icmp_len;

    if (!ip6_is_our_addr(&iph->ip6_dst))
    {
        rte_pktmbuf_free(mbuf);
        return ;
    }
    printf("%s \n", __func__);
    /* must be linear !! */
    icmp_len = ntohs(iph->ip6_plen);
    if ((void *)ich != (void *)(iph + 1))
        icmp_len -= (void *)ich - (void *)(iph+1);

    /* reply */
    ich->icmp6_type = ICMP6_ECHO_REPLY;

    memset(&shdr, 0, sizeof(struct ip6_hdr));

    shdr.ip6_nxt = IPPROTO_ICMPV6;
    shdr.ip6_plen = htons(icmp_len);
    shdr.ip6_dst =  iph->ip6_src;


    if (!ipv6_addr_is_multicast(&iph->ip6_dst)) {
        shdr.ip6_src = iph->ip6_dst;
    } else {
        //shdr.ip6_src = fl6.fl6_saddr;
        net_copy_ip6(&iph->ip6_src,&net_ip6);
    }

    icmp6_send_csum(&shdr, ich);

    return ipv6_xmit(mbuf);
#else
    uint32_t icmp_len;

    if (!ip6_is_our_addr(&iph->ip6_dst))
    {
        
        rte_pktmbuf_free(mbuf);
        return EDPVS_INVPKT; 
    }
    printf("%s \n", __func__);
    /* must be linear !! */
    icmp_len = ntohs(iph->ip6_plen);
    if ((void *)ich != (void *)(iph + 1))
        icmp_len -= (void *)ich - (void *)(iph+1);

    /* reply */
    ich->icmp6_type = ICMP6_ECHO_REPLY;
    iph->ip6_dst =  iph->ip6_src;
    if (!ipv6_addr_is_multicast(&iph->ip6_dst)) {
       iph->ip6_src = iph->ip6_dst;
    } else {
        //shdr.ip6_src = fl6.fl6_saddr;
        net_copy_ip6(&iph->ip6_src,&net_ip6);
    }

    icmp6_send_csum(iph, ich);
    rte_eth_macaddr_get(DEFAULT_PORTID, &eth_addr);
    rte_ether_addr_copy(&eth_h->s_addr, &eth_h->d_addr);
    rte_ether_addr_copy(&eth_addr, &eth_h->s_addr);
    return ipv6_xmit(mbuf);
#endif
}
#else
static int icmp6_echo_reply(struct rte_mbuf *mbuf, struct ip6_hdr *iph,
                            struct icmp6_hdr *ich)
{
    struct ip6_hdr shdr; /* IPv6 header for sending packet */
    uint32_t icmp_len;
    struct flow6 fl6;

    /* must be linear !! */
    icmp_len = ntohs(iph->ip6_plen);
    if ((void *)ich != (void *)(iph + 1))
        icmp_len -= (void *)ich - (void *)(iph+1);

    /* reply */
    ich->icmp6_type = ICMP6_ECHO_REPLY;

    memset(&shdr, 0, sizeof(struct ip6_hdr));
    memset(&fl6, 0, sizeof(struct flow6));

    shdr.ip6_nxt = IPPROTO_ICMPV6;
    shdr.ip6_plen = htons(icmp_len);
    shdr.ip6_dst = fl6.fl6_daddr = iph->ip6_src;

    fl6.fl6_proto = IPPROTO_ICMPV6;
    fl6.fl6_oif = netif_port_get(mbuf->port);

    if (!ipv6_addr_is_multicast(&iph->ip6_dst)) {
        shdr.ip6_src = fl6.fl6_saddr = iph->ip6_dst;
    } else {
        inet_addr_select(AF_INET6, fl6.fl6_oif,
                         (union inet_addr *)&fl6.fl6_daddr, fl6.fl6_scope,
                         (union inet_addr *)&fl6.fl6_saddr);
        shdr.ip6_src = fl6.fl6_saddr;
    }

    icmp6_send_csum(&shdr, ich);

    return ipv6_xmit(mbuf, &fl6);
}
#endif
//int icmp6_rcv(struct rte_mbuf *mbuf)
int icmp6_rcv(struct rte_mbuf *mbuf, uint16_t ip_offfset, uint16_t icmp6_offfset)
{
#if 0
    struct ip6_hdr *iph = mbuf->userdata;
    struct icmp6_hdr *ich;

    assert(iph);

    if (mbuf_may_pull(mbuf, sizeof(struct icmp6_hdr)) != 0)
        goto drop;

    ich = rte_pktmbuf_mtod(mbuf, struct icmp6_hdr *);
    if (unlikely(!ich))
        goto drop;

    if (mbuf_may_pull(mbuf, mbuf->pkt_len) != 0)
        goto drop;
#else
    struct ip6_hdr *iph;
    struct icmp6_hdr *ich;
    iph =  rte_pktmbuf_mtod_offset(mbuf, struct ip6_hdr *, ip_offfset);
    assert(iph);
    ich = rte_pktmbuf_mtod_offset(mbuf, struct icmp6_hdr *,icmp6_offfset);
    if (unlikely(!ich))
    {
        goto drop;
    }
#endif
#if 0
    if(mbuf->nb_segs <=1)
    {
        if (icmp6_csum(iph, ich) != 0xffff)
        {
            goto drop;
        }
    }
    else
    {
        if (icmp6_csum_of_mul_segs(mbuf,iph, ich) != 0xffff)
        {
            printf("**************multiple segs cksum error, mbuf add ddr %p\n", mbuf);
            goto drop;
        }
    }
#else
    if(mbuf->nb_segs <=1)
    {
        if (0xffff!= rte_ipv6_udptcp_cksum((struct rte_ipv6_hdr *)iph, ich))
        {
            goto drop;
        }
    }
    else
    {
        if (icmp6_csum_of_mul_segs(mbuf,iph, ich) != 0xffff)
        {
            printf("**************multiple segs cksum error, mbuf add ddr %p\n", mbuf);
            goto drop;
        }
    }
#endif
#ifdef  CONFIG_DPDK_ICMP_DEBUG
    //ip6_dump_hdr(iph);  
    //dump_icmp6(ich);
#endif
    switch (ich->icmp6_type) {
        case ICMP6_ECHO_REPLY:
#if 0
            printf("************** icmp6 echo reply********************* \n");
            return icmp6_echo_reply_nat(mbuf, iph, ich);
#else
            return icmp6_echo_reply_from_other(mbuf, iph, ich);
#endif
        case ICMP6_ECHO_REQUEST:
            printf("************** icmp6 echo request ********************* \n");
            return icmp6_echo_reply(mbuf, iph, ich);

        case ND_ROUTER_SOLICIT:
        case ND_ROUTER_ADVERT:
        case ND_NEIGHBOR_SOLICIT:
        case ND_NEIGHBOR_ADVERT:
        case ND_REDIRECT:
            return ndisc_rcv(mbuf, netif_port_get(mbuf->port),ip_offfset,icmp6_offfset);

        default :
            ///printf("************** icmp6 type(%d) can not be processed\n",ich->icmp6_type);
            rte_pktmbuf_free(mbuf);
            return EDPVS_INVPKT;
    }

drop:
    rte_pktmbuf_free(mbuf);
    return EDPVS_INVPKT;
}

