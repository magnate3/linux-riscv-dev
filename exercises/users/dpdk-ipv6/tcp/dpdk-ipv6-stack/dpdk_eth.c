#include <assert.h>
#include <netinet/in.h>
#include <netinet/icmp6.h>
#include <netinet/ip6.h>
#include <sys/types.h>
#include <pthread.h>
#include <rte_cycles.h>
#include "dpdk_eth.h"
#include "list.h"
#include "help.h"
#include "dpdk_mbuf.h"
#include "dpdk_ip.h"
#include "dpdk_icmp6.h"
#include "dpdk_nat46.h"
#include "util.h"
#include "ip_range.h"
#include "dpdk_reassembly.h"
#define NETIF_PKT_PREFETCH_OFFSET   3
#define NETIF_PORT_TABLE_BITS 8
#define NETIF_PORT_TABLE_BUCKETS (1 << NETIF_PORT_TABLE_BITS)
#define NETIF_PORT_TABLE_MASK (NETIF_PORT_TABLE_BUCKETS - 1)
extern struct rte_ether_addr cli_mac;
extern struct in6_addr net_ip6 ;
extern struct in6_addr gw_ip6 ;
extern struct rte_ether_addr gw_mac;
extern struct rte_ether_addr net_ethaddr;
uint32_t server_ip_addr = RTE_IPV4(10,10,103,251);
static uint16_t g_nports;
static struct list_head port_tab[NETIF_PORT_TABLE_BUCKETS];
static struct list_head port_ntab[NETIF_PORT_TABLE_BUCKETS];

uint32_t max_flow_num = DEF_FLOW_NUM;
uint32_t max_flow_ttl = DEF_FLOW_TTL;
#define is_multicast_ipv4_addr(ipv4_addr) \
	(((rte_be_to_cpu_32((ipv4_addr)) >> 24) & 0x000000FF) == 0xE0)
uint8_t g_dev_tx_offload_ipv4_cksum=0;
uint8_t g_dev_tx_offload_tcpudp_cksum=0;
void dumpFile(const u_char *pkt, int len, time_t tv_sec, suseconds_t tv_usec);

static void open_pcap(struct netif_port *dev,const char *fname)
{
    char name[IFNAMSIZ +16];
    snprintf(name,IFNAMSIZ +16,"%s-txrx.pcap",fname);
    dev->dumper = pcap_dump_open(pcap_open_dead(DLT_EN10MB, 1600), name);
    if (NULL == dev->dumper)
    {
        printf("dumper is NULL\n");
        return;
    }
}

void dump_pcap( const struct netif_port *dev, const u_char *pkt, int len)
{
    struct timeval tv;
    struct pcap_pkthdr hdr;
    gettimeofday(&tv, NULL);
    hdr.ts.tv_sec = tv.tv_sec;
    hdr.ts.tv_usec = tv.tv_usec;
    hdr.caplen = len;
    hdr.len = len; 
    pcap_dump((u_char*)(dev->dumper), &hdr, pkt); 
}
struct inet_ifaddr *inet_addr_ifa_get(int af, const struct netif_port *dev,
                                      union inet_addr *addr)
{
    struct inet_ifaddr *ifa=NULL;

    printf("%s not implement \n",__func__);   
    return ifa;
}

void inet_addr_ifa_put(struct inet_ifaddr *ifa)
{
    //ifa_put(ifa);
    printf("%s not implement \n",__func__);   
}
static uint16_t
ipv4_hdr_cksum(struct rte_ipv4_hdr *ip_h)
{
	uint16_t *v16_h;
	uint32_t ip_cksum;

	/*
 * 	 * Compute the sum of successive 16-bit words of the IPv4 header,
 * 	 	 * skipping the checksum field of the header.
 * 	 	 	 */
	v16_h = (unaligned_uint16_t *) ip_h;
	ip_cksum = v16_h[0] + v16_h[1] + v16_h[2] + v16_h[3] +
		v16_h[4] + v16_h[6] + v16_h[7] + v16_h[8] + v16_h[9];

	/* reduce 32 bit checksum to 16 bits and complement it */
	ip_cksum = (ip_cksum & 0xffff) + (ip_cksum >> 16);
	ip_cksum = (ip_cksum & 0xffff) + (ip_cksum >> 16);
	ip_cksum = (~ip_cksum) & 0x0000FFFF;
	return (ip_cksum == 0) ? 0xFFFF : (uint16_t) ip_cksum;
}
uint16_t eth_send_packets(struct rte_mbuf **mbuf, uint16_t port_id,uint16_t queue_id, const uint16_t burst_num) {
        //printf("send packets port %u, queue %u \n", port_id, queue_id);
	uint16_t nb_tx = rte_eth_tx_burst(port_id, queue_id,mbuf, burst_num);
        uint16_t j = 0;
	if (unlikely(nb_tx < burst_num)) {
            for(j = nb_tx; j < burst_num; ++j)
		rte_pktmbuf_free(mbuf[j]);
	}
	return nb_tx;
}
static inline eth_type_t eth_type_parse(const struct rte_ether_hdr *eth_hdr,
                                        const struct netif_port *dev)
{
    if (eth_addr_equal(&dev->addr, &eth_hdr->d_addr))
        return ETH_PKT_HOST;

    if (rte_is_multicast_ether_addr(&eth_hdr->d_addr)) {
        if (rte_is_broadcast_ether_addr(&eth_hdr->d_addr))
            return ETH_PKT_BROADCAST;
        else
            return ETH_PKT_MULTICAST;
    }

    return ETH_PKT_OTHERHOST;
}
static inline uint32_t
reverse_ip_addr(const uint32_t ip_addr)
{
    return RTE_IPV4((uint8_t)(ip_addr & 0xff),
                (uint8_t)((ip_addr >> 8) & 0xff),
                (uint8_t)((ip_addr >> 16) & 0xff),
                (uint8_t)((ip_addr >> 24) & 0xff));
}
static const char *
arp_op_name(uint16_t arp_op)
{
	switch (arp_op) {
	case RTE_ARP_OP_REQUEST:
		return "ARP Request";
	case RTE_ARP_OP_REPLY:
		return "ARP Reply";
	case RTE_ARP_OP_REVREQUEST:
		return "Reverse ARP Request";
	case RTE_ARP_OP_REVREPLY:
		return "Reverse ARP Reply";
	case RTE_ARP_OP_INVREQUEST:
		return "Peer Identify Request";
	case RTE_ARP_OP_INVREPLY:
		return "Peer Identify Reply";
	default:
		break;
	}
	return "Unkwown ARP op";
}
static int process_arp(struct rte_mbuf *mbuf, struct rte_ether_hdr * eth_h)
{
     
     int l2_len = mbuf->l2_len;
     struct rte_arp_hdr * arp_h = (struct rte_arp_hdr *) ((char *)eth_h + l2_len);
     uint16_t arp_op;
     uint16_t arp_pro;
     uint32_t ip_addr;
     struct rte_ether_addr eth_addr;
     struct netif_port *dev = netif_port_get(mbuf->port);
     arp_op = RTE_BE_TO_CPU_16(arp_h->arp_opcode);
     arp_pro = RTE_BE_TO_CPU_16(arp_h->arp_protocol);
     ip_addr = arp_h->arp_data.arp_tip;
     if (NULL == dev)
     {
          goto free_mbuf;
     }
     
     if (reverse_ip_addr(ip_addr) == server_ip_addr)
     {
         	    printf("  ARP:  hrd=%d proto=0x%04x hln=%d "
				       "pln=%d op=%u (%s)\n",
				       RTE_BE_TO_CPU_16(arp_h->arp_hardware),
				       arp_pro, arp_h->arp_hlen,
				       arp_h->arp_plen, arp_op,
				       arp_op_name(arp_op));

     }
     else
     {
          goto free_mbuf;
     }
     if ((RTE_BE_TO_CPU_16(arp_h->arp_hardware) !=
			     RTE_ARP_HRD_ETHER) ||
			    (arp_pro != RTE_ETHER_TYPE_IPV4) ||
			    (arp_h->arp_hlen != 6) ||
			    (arp_h->arp_plen != 4)
			    ) 
     {
          goto free_mbuf;
     }
     if (arp_op != RTE_ARP_OP_REQUEST) {
          goto free_mbuf;
     }
     rte_ether_addr_copy(&eth_h->s_addr, &cli_mac);
     rte_ether_addr_copy(&eth_h->s_addr, &eth_h->d_addr);
     rte_ether_addr_copy(&dev->addr, &eth_h->s_addr);
     arp_h->arp_opcode = rte_cpu_to_be_16(RTE_ARP_OP_REPLY);
     rte_ether_addr_copy(&arp_h->arp_data.arp_tha,
					&eth_addr);
     rte_ether_addr_copy(&arp_h->arp_data.arp_sha,
					&arp_h->arp_data.arp_tha);
     rte_ether_addr_copy(&eth_h->s_addr,
					&arp_h->arp_data.arp_sha);
     /* Swap IP addresses in ARP payload */
     ip_addr = arp_h->arp_data.arp_sip;
     arp_h->arp_data.arp_sip = arp_h->arp_data.arp_tip;
     arp_h->arp_data.arp_tip = ip_addr;
     eth_send_packets(&mbuf, mbuf->port,0,1);
     return 0;
free_mbuf:
     rte_pktmbuf_free(mbuf);
     return 0;
}
static int process_icmp4(struct rte_mbuf *mbuf,struct rte_ether_hdr * eth_h)
{
    struct rte_ipv4_hdr *ip_h;
    struct rte_icmp_hdr *icmp_h; 
    struct rte_ether_addr eth_addr;
    uint32_t ip_addr;
    uint32_t cksum;
     int l2_len = mbuf->l2_len;
    ip_h = (struct rte_ipv4_hdr *) ((char *)eth_h + l2_len);
    if (ip_h->next_proto_id != IPPROTO_ICMP) {
          goto free_mbuf;
    }
    icmp_h = (struct rte_icmp_hdr *) ((char *)ip_h + sizeof(struct rte_ipv4_hdr));
    if (!((icmp_h->icmp_type == RTE_IP_ICMP_ECHO_REQUEST) && (icmp_h->icmp_code == 0))) {
          goto free_mbuf;
    }
    rte_ether_addr_copy(&eth_h->s_addr, &eth_addr);
    rte_ether_addr_copy(&eth_h->d_addr, &eth_h->s_addr);
    rte_ether_addr_copy(&eth_addr, &eth_h->d_addr);
    ip_addr = ip_h->src_addr;
    if (is_multicast_ipv4_addr(ip_h->dst_addr)) {
	uint32_t ip_src;
	ip_src = rte_be_to_cpu_32(ip_addr);
	if ((ip_src & 0x00000003) == 1)
		ip_src = (ip_src & 0xFFFFFFFC) | 0x00000002;
	else
		ip_src = (ip_src & 0xFFFFFFFC) | 0x00000001;
	ip_h->src_addr = rte_cpu_to_be_32(ip_src);
	ip_h->dst_addr = ip_addr;
	ip_h->hdr_checksum = ipv4_hdr_cksum(ip_h);
	} else {
        ip_h->src_addr = ip_h->dst_addr;
        ip_h->dst_addr = ip_addr;
		}
	icmp_h->icmp_type = RTE_IP_ICMP_ECHO_REPLY;
	cksum = ~icmp_h->icmp_cksum & 0xffff;
	cksum += ~htons(RTE_IP_ICMP_ECHO_REQUEST << 8) & 0xffff;
	cksum += htons(RTE_IP_ICMP_ECHO_REPLY << 8);
	cksum = (cksum & 0xffff) + (cksum >> 16);
	cksum = (cksum & 0xffff) + (cksum >> 16);
	icmp_h->icmp_cksum = ~cksum;
     eth_send_packets(&mbuf, mbuf->port,0,1);
     return 0;
free_mbuf:
	rte_pktmbuf_free(mbuf);
        return EDPVS_INVPKT; 
}
/*
 *     dest->transport_in_handler = lb_ipv4_icmp_nat_in_handler;
 *     dest->transport_out_handler = lb_ipv4_icmp_nat_out_handler;
 */
static int process_icmp46(struct rte_mbuf *mbuf,struct rte_ether_hdr * eth_h)
{
    struct rte_ipv4_hdr *ip_h;
    struct rte_icmp_hdr *icmp_h; 
    int l2_len = mbuf->l2_len;
    struct icmp6_hdr* ic6h;
    struct ip6_hdr *ip6h = NULL;
    struct rte_mbuf *dst_mbuf;
    char *pktbuf =NULL; 
    char  *data =NULL; 
    uint16_t data_len = 0;
    ip_h = (struct rte_ipv4_hdr *) ((char *)eth_h + l2_len);
    if (ip_h->next_proto_id != IPPROTO_ICMP) {
         goto free_mbuf;
    }
    icmp_h = (struct rte_icmp_hdr *) ((char *)ip_h + sizeof(struct rte_ipv4_hdr));
    if (!((icmp_h->icmp_type == RTE_IP_ICMP_ECHO_REQUEST) && (icmp_h->icmp_code == 0))) {
        goto free_mbuf;
    }
    //must do rte_pktmbuf_adjrte_pktmbuf_adj
    if (unlikely(NULL == rte_pktmbuf_adj(mbuf, sizeof(struct rte_ether_hdr))))
    {
        goto free_mbuf;
    }
    data_len = mbuf->pkt_len - ip4_hdrlen(mbuf) - sizeof(struct rte_icmp_hdr);
#if 1
    mbuf->l3_len = sizeof(struct rte_ipv4_hdr);
#else
    mbuf->l3_len = ip4_hdrlen(mbuf);
#endif
    
    dst_mbuf = get_mbuf_from_netif(netif_port_get(mbuf->port));
    if(!dst_mbuf)
    {
        goto free_mbuf;
    }
    pktbuf = rte_pktmbuf_mtod(dst_mbuf, char *); 
    eth_hdr_push(dst_mbuf); 
    ip6h = ipv6_hdr_push_common(mbuf,dst_mbuf,data_len + sizeof(struct icmp6_hdr)); 
    ip6h->ip6_nxt = IPPROTO_ICMPV6;
    ic6h = RTE_PKTMBUF_PUSH(dst_mbuf, struct icmp6_hdr); 
    data = (char*)rte_pktmbuf_append(dst_mbuf, data_len); 
    switch(icmp_h->icmp_type){
         case RTE_IP_ICMP_ECHO_REQUEST:  
                ic6h->icmp6_type = ICMP6_ECHO_REQUEST;
                ic6h->icmp6_code = icmp_h->icmp_code;
                ic6h->icmp6_cksum = 0;
                ic6h->icmp6_data16[0]= icmp_h->icmp_ident;	/* Identifier */
                ic6h->icmp6_data16[1]=  icmp_h->icmp_seq_nb;	/* Sequence Number */
                //dump_icmp6(ic6h);
		break;
         default:
                goto free_mbuf;
    } 
    rte_memcpy(data,rte_pktmbuf_mtod_offset(mbuf, char *, ip4_hdrlen(mbuf) + sizeof(struct rte_icmp_hdr)),data_len);
    icmp6_send_csum(ip6h,ic6h);
    //printf("************dump nat46 icmp request  %d**************",  pthread_self());
    ip6_dump_hdr(ip6h);
    //dump_pcap(netif_port_get(mbuf->port),(u_char*)pktbuf,dst_mbuf->pkt_len);
    eth_send_packets(&dst_mbuf, mbuf->port,0,1);
    rte_pktmbuf_free(mbuf);
    return 0;
free_mbuf:
	rte_pktmbuf_free(mbuf);
        return EDPVS_INVPKT; 
    return 0;
}
static int process_ipv4(struct rte_mbuf *mbuf,struct rte_ether_hdr * eth_h)
{
     //return process_icmp4(mbuf,eth_h);
     return process_icmp46(mbuf,eth_h);
}
static inline int netif_deliver_mbuf(struct rte_mbuf *mbuf, uint16_t packet_type)
{
    //struct pkt_type *pt;
    //int err;
    uint16_t eth_type;
    //uint16_t data_off;
    //struct netif_port * dev;
    struct rte_ether_hdr *eth_hdr;
    assert(mbuf->port <= NETIF_MAX_PORTS);
    eth_hdr = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);
    eth_type = RTE_BE_TO_CPU_16(eth_hdr->ether_type);
    mbuf->l2_len = sizeof(struct rte_ether_hdr);
    /* Reply to ARP requests */
    if (eth_type == RTE_ETHER_TYPE_ARP) {
          process_arp(mbuf,eth_hdr);
    }
    else if (eth_type == RTE_ETHER_TYPE_IPV4) {
         //printf("%s ipv4, tid %x, mbuf %p , mbuf port %u \n",__func__, pthread_self(),mbuf, mbuf->port);
         process_ipv4(mbuf,eth_hdr);
    }
    else if (eth_type == RTE_ETHER_TYPE_IPV6) {
          //printf("%s ipv6, tid %x, mbuf %p , mbuf port %u \n",__func__, pthread_self(),mbuf, mbuf->port);
          dpdk_ip_proto_process_raw(mbuf);
    }
    else
    {
        rte_pktmbuf_free(mbuf);
    }
#if 0
    /* Remove ether_hdr at the beginning of an mbuf */
    data_off = mbuf->data_off;
    if (unlikely(NULL == rte_pktmbuf_adj(mbuf, sizeof(struct rte_ether_hdr))))
    {
        rte_pktmbuf_free(mbuf);
        return EDPVS_INVPKT; 
    }
#endif
    return EDPVS_OK;
}
void lcore_process_packets(struct rte_mbuf **mbufs,const uint16_t count, uint16_t portid)
{
    uint16_t i,t;
    struct netif_port *dev;
    struct rte_ether_hdr *eth_hdr;
     /* prefetch packets */
#if 1
    for (t = 0; t < count && t < NETIF_PKT_PREFETCH_OFFSET; t++)
        rte_prefetch0(rte_pktmbuf_mtod(mbufs[t], void *));
#endif
    /* L2 filter */
    for (i = 0; i < count; ++i){
        struct rte_mbuf *mbuf = mbufs[i];
        mbuf->port = portid;
        //printf("i: %u, count %u, mbuf %p , mbuf port %u,portid %u \n",i,count, mbuf, mbuf->port, portid);
#if 1
         if (t < count) {
            rte_prefetch0(rte_pktmbuf_mtod(mbufs[t], void *));
            t++;
        }
#endif
#if 1
        //printf("mbuf addr %p, mbuf port %u, %d \n",mbuf, mbuf->port, i);
        dev = netif_port_get(mbuf->port);  
        if (unlikely(!dev)) {
              rte_pktmbuf_free(mbuf);
              continue;
            }
#endif
        eth_hdr = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);
        /* reuse mbuf.packet_type, it was RTE_PTYPE_XXX */
         mbuf->packet_type = eth_type_parse(eth_hdr, dev);
#if 0 
        /*
 *          * handle VLAN
 *                   * if HW offload vlan strip, it's still need vlan module
 *                            * to act as VLAN filter.
 *                                     */
        if (eth_hdr->ether_type == htons(ETH_P_8021Q) ||
            mbuf->ol_flags & PKT_RX_VLAN_STRIPPED) {

            if (vlan_rcv(mbuf, netif_port_get(mbuf->port)) != EDPVS_OK) {
                rte_pktmbuf_free(mbuf);
                lcore_stats[cid].dropped++;
                continue;
            }

            dev = netif_port_get(mbuf->port);
            if (unlikely(!dev)) {
                rte_pktmbuf_free(mbuf);
                lcore_stats[cid].dropped++;
                continue;
            }

            eth_hdr = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);
        } 
#endif // vlan
        netif_deliver_mbuf(mbuf,mbuf->packet_type);
    }
}
static void init_addr(struct netif_port * dev)
{
     dev->ipv6 = true;
     dev->local_ip.in6 =  net_ip6;
     dev->gateway_ip.in6 =  gw_ip6;
     rte_ether_addr_copy(&net_ethaddr, &dev->local_mac);
     // not init
     //rte_ether_addr_copy(gw_mac, &dev->gateway_mac);
}
void init_netif(struct netif_port * dev ,struct rte_mempool      *mbuf_pool,uint16_t portid, int queue_num)
{
    uint64_t frag_cycles;
    //ipaddr_t ip;
    int socket = 0;
    frag_cycles = (rte_get_tsc_hz() + MS_PER_S - 1) / MS_PER_S *
                max_flow_ttl;
    dev->mbuf_pool = mbuf_pool;
    dev->id = portid;
    dev->queue_num = queue_num;
    rte_eth_macaddr_get(portid, &dev->addr);
    snprintf(dev->name,IFNAMSIZ -1, "eth%u",portid);
    open_pcap(dev,dev->name);
#if 0
    ip.in6 = net_ip6;
    if (ip_range_init(&dev->server_ip_range, ip, 1) < 0) {
        printf("bad server ip range \n");
        exit(0);
    }
#endif
    init_addr(dev);
    if ((dev->frag_tbl = rte_ip_frag_table_create(max_flow_num,
                        IP_FRAG_TBL_BUCKET_ENTRIES, max_flow_num, frag_cycles,
                        socket)) == NULL) {
                RTE_LOG(ERR, IP_RSMBL, "ip_frag_tbl_creat failed \n");
                exit(0);
    }
   
}
struct rte_mempool *get_netif_mempool(uint16_t portid)
{
     struct netif_port * dev = netif_port_get(portid);
     if(NULL == dev)
         return NULL;
     return dev->mbuf_pool;
}
struct rte_mbuf * get_mbuf_from_netif(struct netif_port * dev)
{
   struct rte_mbuf *mbuf = NULL;   
   mbuf = rte_pktmbuf_alloc(dev->mbuf_pool);
   //printf("mbuf addr %p and phy addr %p, and next %p \n",mbuf,rte_mem_virt2phy(mbuf), mbuf->next);
   return mbuf;
}
static inline int port_tab_hashkey(portid_t id)
{
    return id & NETIF_PORT_TABLE_MASK;
}

static unsigned int port_ntab_hashkey(const char *name, size_t len)
{
    unsigned int i;
    unsigned int hash=1315423911;
    for (i = 0; i < len; i++)
    {
        if (name[i] == '\0')
            break;
        hash^=((hash<<5)+name[i]+(hash>>2));
    }

    return (hash % NETIF_PORT_TABLE_BUCKETS);
}

static inline void port_tab_init(void)
{
    int i;
    for (i = 0; i < NETIF_PORT_TABLE_BUCKETS; i++)
        INIT_LIST_HEAD(&port_tab[i]);
}

static inline void port_ntab_init(void)
{
    int i;
    for (i = 0; i < NETIF_PORT_TABLE_BUCKETS; i++)
        INIT_LIST_HEAD(&port_ntab[i]);
}
int netif_port_register(struct netif_port *port)
{
    struct netif_port *cur;
    int hash, nhash;
    int err = EDPVS_OK;

    if (unlikely(NULL == port))
        return EDPVS_INVAL;

    hash = port_tab_hashkey(port->id);
    list_for_each_entry(cur, &port_tab[hash], list) {
        if (cur->id == port->id || strcmp(cur->name, port->name) == 0) {
            return EDPVS_EXIST;
        }
    }

    nhash = port_ntab_hashkey(port->name, sizeof(port->name));
    list_for_each_entry(cur, &port_ntab[hash], nlist) {
        if (cur->id == port->id || strcmp(cur->name, port->name) == 0) {
            return EDPVS_EXIST;
        }
    }

    list_add_tail(&port->list, &port_tab[hash]);
    list_add_tail(&port->nlist, &port_ntab[nhash]);
    g_nports++;
    return err;
}
struct netif_port* netif_port_get(uint16_t id)
{
    int hash = port_tab_hashkey(id);
    struct netif_port *port;
    //printf("port id %u\n",id);
    assert(id <= NETIF_MAX_PORTS);

    list_for_each_entry(port, &port_tab[hash], list) {
        if (port->id == id) {
            return port;
        }
    }
    printf("dev is NULL \n");
    return NULL;
}

static struct netif_port* netif_port_get_by_name(const char *name)
{
    int nhash;
    struct netif_port *port;

    if (!name || strlen(name) <= 0)
        return NULL;

    nhash = port_ntab_hashkey(name, strlen(name));
    list_for_each_entry(port, &port_ntab[nhash], nlist) {
        if (!strcmp(port->name, name)) {
            return port;
        }
    }

    return NULL;
}
struct rte_mempool *port_get_mbuf_pool(uint16_t portid, uint16_t queueid)
{
      return get_netif_mempool(portid);
}
int init_dpdk_eth_mod(void)
{
    port_tab_init();
    port_ntab_init();
    return 0;
}
