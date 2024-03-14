/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <string.h>
#include <sys/queue.h>
#include <stdarg.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <sys/param.h>
#include <arpa/inet.h>

#include <rte_common.h>
#include <rte_byteorder.h>
#include <rte_log.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_eal.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_udp.h>
#include <rte_string_fns.h>
#include <rte_lpm.h>
#include <rte_lpm6.h>

#include <rte_ip_frag.h>
#include "help.h"
#define MAX_PKT_BURST 32


#define RTE_LOGTYPE_IP_RSMBL RTE_LOGTYPE_USER1

#define MAX_JUMBO_PKT_LEN  9600

#define	BUF_SIZE	RTE_MBUF_DEFAULT_DATAROOM
#define	MBUF_DATA_SIZE	RTE_MBUF_DEFAULT_BUF_SIZE

#define NB_MBUF 8192
#define MEMPOOL_CACHE_SIZE 256

/* allow max jumbo frame 9.5 KB */
#define JUMBO_FRAME_MAX_SIZE	0x2600

#define	MAX_FLOW_NUM	UINT16_MAX
#define	MIN_FLOW_NUM	1
#define	DEF_FLOW_NUM	0x1000

/* TTL numbers are in ms. */
#define	MAX_FLOW_TTL	(3600 * MS_PER_S)
#define	MIN_FLOW_TTL	1
#define	DEF_FLOW_TTL	MS_PER_S

#define MAX_FRAG_NUM RTE_LIBRTE_IP_FRAG_MAX_FRAG

/* Should be power of two. */
#define	IP_FRAG_TBL_BUCKET_ENTRIES	16

static uint32_t max_flow_num = DEF_FLOW_NUM;
static uint32_t max_flow_ttl = DEF_FLOW_TTL;

#define BURST_TX_DRAIN_US 100 /* TX drain every ~100us */

#define NB_SOCKETS 8

/* Configure how many packets ahead to prefetch, when reading packets */
#define PREFETCH_OFFSET	3
#define CONNECT_81_PORT 0
#define CONNECT_82_PORT 1
#define SRV_81_IP  "10.10.103.81"
#define SRV_82_IP  "100.20.0.82"
static struct rte_ether_addr srv81_ether_addr =
    {{0x48,0x57,0x02,0x64,0xea,0x1e}};

static struct rte_ether_addr srv82_ether_addr =
    {{0x48,0x57,0x02,0x64,0xe7,0xad}};
/*
 * Configurable number of RX/TX ring descriptors
 */
#define RTE_TEST_RX_DESC_DEFAULT 1024
#define RTE_TEST_TX_DESC_DEFAULT 1024

static uint16_t nb_rxd = RTE_TEST_RX_DESC_DEFAULT;
static uint16_t nb_txd = RTE_TEST_TX_DESC_DEFAULT;

/* ethernet addresses of ports */
static struct rte_ether_addr ports_eth_addr[RTE_MAX_ETHPORTS];

#ifndef IPv4_BYTES
#define IPv4_BYTES_FMT "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8
#define IPv4_BYTES(addr) \
		(uint8_t) (((addr) >> 24) & 0xFF),\
		(uint8_t) (((addr) >> 16) & 0xFF),\
		(uint8_t) (((addr) >> 8) & 0xFF),\
		(uint8_t) ((addr) & 0xFF)
#endif

#ifndef IPv6_BYTES
#define IPv6_BYTES_FMT "%02x%02x:%02x%02x:%02x%02x:%02x%02x:"\
                       "%02x%02x:%02x%02x:%02x%02x:%02x%02x"
#define IPv6_BYTES(addr) \
	addr[0],  addr[1], addr[2],  addr[3], \
	addr[4],  addr[5], addr[6],  addr[7], \
	addr[8],  addr[9], addr[10], addr[11],\
	addr[12], addr[13],addr[14], addr[15]
#endif

#define IPV6_ADDR_LEN 16

/* mask of enabled ports */
static uint32_t enabled_port_mask = 0;

static int rx_queue_per_lcore = 1;

struct mbuf_table {
	uint32_t len;
	uint32_t head;
	uint32_t tail;
	struct rte_mbuf *m_table[0];
};

struct rx_queue {
	struct rte_ip_frag_tbl *frag_tbl;
	struct rte_mempool *pool;
	struct rte_lpm *lpm;
	struct rte_lpm6 *lpm6;
	uint16_t portid;
};

struct tx_lcore_stat {
	uint64_t call;
	uint64_t drop;
	uint64_t queue;
	uint64_t send;
};

#define MAX_RX_QUEUE_PER_LCORE 16
#define MAX_TX_QUEUE_PER_PORT 16
#define MAX_RX_QUEUE_PER_PORT 128

struct lcore_queue_conf {
	uint16_t n_rx_queue;
	struct rx_queue rx_queue_list[MAX_RX_QUEUE_PER_LCORE];
	uint16_t tx_queue_id[RTE_MAX_ETHPORTS];
	struct rte_ip_frag_death_row death_row;
	struct mbuf_table *tx_mbufs[RTE_MAX_ETHPORTS];
	struct tx_lcore_stat tx_stat;
} __rte_cache_aligned;
static struct lcore_queue_conf lcore_queue_conf[RTE_MAX_LCORE];

static struct rte_eth_conf port_conf = {
	.rxmode = {
		.mq_mode        = ETH_MQ_RX_RSS,
		.max_rx_pkt_len = JUMBO_FRAME_MAX_SIZE,
		.split_hdr_size = 0,
		.offloads = (DEV_RX_OFFLOAD_CHECKSUM |
			     DEV_RX_OFFLOAD_JUMBO_FRAME),
	},
	.rx_adv_conf = {
			.rss_conf = {
				.rss_key = NULL,
				.rss_hf = ETH_RSS_IP,
		},
	},
	.txmode = {
		.mq_mode = ETH_MQ_TX_NONE,
		.offloads = (DEV_TX_OFFLOAD_IPV4_CKSUM |
			     DEV_TX_OFFLOAD_MULTI_SEGS),
	},
};

/*
 * IPv4 forwarding table
 */
struct l3fwd_ipv4_route {
	uint32_t ip;
	uint8_t  depth;
	uint8_t  if_out;
};

struct l3fwd_ipv4_route l3fwd_ipv4_route_array[] = {
		//{RTE_IPV4(100,10,0,0), 16, 0},
		{RTE_IPV4(10,10,0,0), 16, 0},
		{RTE_IPV4(100,20,0,0), 16, 1},
		{RTE_IPV4(100,30,0,0), 16, 2},
		{RTE_IPV4(100,40,0,0), 16, 3},
		{RTE_IPV4(100,50,0,0), 16, 4},
		{RTE_IPV4(100,60,0,0), 16, 5},
		{RTE_IPV4(100,70,0,0), 16, 6},
		{RTE_IPV4(100,80,0,0), 16, 7},
};

/*
 * IPv6 forwarding table
 */

struct l3fwd_ipv6_route {
	uint8_t ip[IPV6_ADDR_LEN];
	uint8_t depth;
	uint8_t if_out;
};

static struct l3fwd_ipv6_route l3fwd_ipv6_route_array[] = {
	{{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 0},
	{{2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 1},
	{{3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 2},
	{{4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 3},
	{{5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 4},
	{{6,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 5},
	{{7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 6},
	{{8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, 48, 7},
};

#define LPM_MAX_RULES         1024
#define LPM6_MAX_RULES         1024
#define LPM6_NUMBER_TBL8S (1 << 16)

struct rte_lpm6_config lpm6_config = {
		.max_rules = LPM6_MAX_RULES,
		.number_tbl8s = LPM6_NUMBER_TBL8S,
		.flags = 0
};

static struct rte_lpm *socket_lpm[RTE_MAX_NUMA_NODES];
static struct rte_lpm6 *socket_lpm6[RTE_MAX_NUMA_NODES];

#ifdef RTE_LIBRTE_IP_FRAG_TBL_STAT
#define TX_LCORE_STAT_UPDATE(s, f, v)   ((s)->f += (v))
#else
#define TX_LCORE_STAT_UPDATE(s, f, v)   do {} while (0)
#endif /* RTE_LIBRTE_IP_FRAG_TBL_STAT */

/*
 * If number of queued packets reached given threahold, then
 * send burst of packets on an output interface.
 */
static inline uint32_t
send_burst(struct lcore_queue_conf *qconf, uint32_t thresh, uint16_t port)
{
	uint32_t fill, len, k, n;
	struct mbuf_table *txmb;

	txmb = qconf->tx_mbufs[port];
	len = txmb->len;

	if ((int32_t)(fill = txmb->head - txmb->tail) < 0)
		fill += len;

	if (fill >= thresh) {
		n = RTE_MIN(len - txmb->tail, fill);

		k = rte_eth_tx_burst(port, qconf->tx_queue_id[port],
			txmb->m_table + txmb->tail, (uint16_t)n);

		TX_LCORE_STAT_UPDATE(&qconf->tx_stat, call, 1);
		TX_LCORE_STAT_UPDATE(&qconf->tx_stat, send, k);

		fill -= k;
		if ((txmb->tail += k) == len)
			txmb->tail = 0;
	}

	return fill;
}

/* Enqueue a single packet, and send burst if queue is filled */
static inline int
send_single_packet(struct rte_mbuf *m, uint16_t port)
{
	uint32_t fill, lcore_id, len;
	struct lcore_queue_conf *qconf;
	struct mbuf_table *txmb;

	lcore_id = rte_lcore_id();
	qconf = &lcore_queue_conf[lcore_id];

	txmb = qconf->tx_mbufs[port];
	len = txmb->len;

	fill = send_burst(qconf, MAX_PKT_BURST, port);

	if (fill == len - 1) {
		TX_LCORE_STAT_UPDATE(&qconf->tx_stat, drop, 1);
		rte_pktmbuf_free(txmb->m_table[txmb->tail]);
		if (++txmb->tail == len)
			txmb->tail = 0;
	}

	TX_LCORE_STAT_UPDATE(&qconf->tx_stat, queue, 1);
	txmb->m_table[txmb->head] = m;
	if(++txmb->head == len)
		txmb->head = 0;

	return 0;
}
static void print_ip(int ip)
{
    unsigned char bytes[4];
    bytes[0] = ip & 0xFF;
    bytes[1] = (ip >> 8) & 0xFF;
    bytes[2] = (ip >> 16) & 0xFF;
    bytes[3] = (ip >> 24) & 0xFF;
    printf("%d.%d.%d.%d\n", bytes[3], bytes[2], bytes[1], bytes[0]);
}
static inline void
read_and_print_ipv4_info(struct rte_mbuf *m)
{

        uint32_t ip_dst = 0;
        uint32_t ip81= 0, ip82 = 0;
        uint32_t ip_src; //20170830
        uint16_t flag_offset, ip_ofs, ip_flag;
        uint16_t total_len; //20170902 16 bits unsign int total_length
        struct rte_ether_hdr * eth_h = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
        struct rte_ipv4_hdr* ip_hdr = (struct rte_ipv4_hdr *)(struct rte_ipv4_hdr *)(eth_h + 1);
        inet_pton(AF_INET,  SRV_81_IP, (void *)&ip81);
        inet_pton(AF_INET,  SRV_82_IP, (void *)&ip82);
        ip81 = ntohl(ip81);
        ip82 = ntohl(ip82);
        ip_dst = ntohl(ip_hdr->dst_addr);
        if(ip_dst == ip81 || ip_dst == ip82)
         {
             printf("ip_dst: "); 
             print_ip(ip_dst); 
             ip_src = rte_be_to_cpu_32(ip_hdr->src_addr);//20170830
             printf("ip_src: "); //20170904
             print_ip(ip_src);
             flag_offset = rte_be_to_cpu_16(ip_hdr->fragment_offset);
             ip_ofs = (uint16_t)(flag_offset & RTE_IPV4_HDR_OFFSET_MASK);
             ip_flag = (uint16_t)(flag_offset & RTE_IPV4_HDR_MF_FLAG);
              total_len = rte_cpu_to_be_16(ip_hdr->total_length);//20170831 rte_cpu_to_be_16

             printf("total_length: ");
             printf("%" PRIu16 "\n" ,total_len);//20170831
             printf("more frag: ");
             printf("%" PRIu16 "\n" ,ip_flag);//20170831

             printf("offset: ");
             printf("%" PRIu16 "\n" ,ip_ofs);//20170831
             printf("\n");
         }
}
static inline void print_mac_info(struct rte_mbuf *pkt)
{
    struct rte_ether_hdr *eth_hdr;
    eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
#if 0
    printf("Pkt src MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
                   " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 " \n",
                   eth_hdr->s_addr.addr_bytes[0], eth_hdr->s_addr.addr_bytes[1],
                   eth_hdr->s_addr.addr_bytes[2], eth_hdr->s_addr.addr_bytes[3],
                   eth_hdr->s_addr.addr_bytes[4], eth_hdr->s_addr.addr_bytes[5]);
#endif
    printf("src mac: ");
    print_mac_addr(eth_hdr->s_addr);
    printf("  dst mac: ");
    print_mac_addr(eth_hdr->d_addr);
    printf("\n");
}
static inline void debug_frag(struct rte_mbuf *m)
{
     printf("pkt len %u \n", m->pkt_len);
     struct rte_mbuf * head = m;
     int i = 0;
     while(head) {
         printf("********* mbuf data len %u , ", head->data_len);
         printf("mbuf data is:   \n");
         for(i = 0; i < head->data_len; ++i) 
         {
            printf("%2X", (rte_pktmbuf_mtod(head,unsigned char *)[i]));
         }
         printf("\n");
         head = head->next;
     }     
}
static inline void
reassemble(struct rte_mbuf *m, uint16_t portid, uint32_t queue,
	struct lcore_queue_conf *qconf, uint64_t tms)
{
	struct rte_ether_hdr *eth_hdr;
	struct rte_ip_frag_tbl *tbl;
	struct rte_ip_frag_death_row *dr;
	struct rx_queue *rxq;
	void *d_addr_bytes;
	uint32_t next_hop;
	uint16_t dst_port;

	rxq = &qconf->rx_queue_list[queue];

	eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);

	dst_port = portid;

        //print_mac_info(m);
	/* if packet is IPv4 */
        //printf(">>>>>>>>>>>>> m->packet_type %x \n", m->packet_type);
	if (RTE_ETH_IS_IPV4_HDR(m->packet_type)) {
		struct rte_ipv4_hdr *ip_hdr;
		uint32_t ip_dst;

		ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);

#if 0
                 read_and_print_ipv4_info(m);
#endif
		 /* if it is a fragmented packet, then try to reassemble. */
		if (rte_ipv4_frag_pkt_is_fragmented(ip_hdr)) {
			struct rte_mbuf *mo;

			tbl = rxq->frag_tbl;
			dr = &qconf->death_row;

			/* prepare mbuf: setup l2_len/l3_len. */
			m->l2_len = sizeof(*eth_hdr);
			m->l3_len = sizeof(*ip_hdr);

			/* process this fragment. */
			mo = rte_ipv4_frag_reassemble_packet(tbl, dr, m, tms, ip_hdr);
			if (mo == NULL)
                        {
                                printf(" a frag recv  and wait for reassemble\n");
				/* no packet to send out. */
				return;

                        }
                        printf("reassembled frags and forward \n");
			/* we have our packet reassembled. */
			if (mo != m) {
				m = mo;
				eth_hdr = rte_pktmbuf_mtod(m,
					struct rte_ether_hdr *);
				ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
			}

#if 0
                        read_and_print_ipv4_info(m);
#endif
			/* update offloading flags */
			m->ol_flags |= (PKT_TX_IPV4 | PKT_TX_IP_CKSUM);
		}
#if 0
                debug_frag(m);
#endif
		ip_dst = rte_be_to_cpu_32(ip_hdr->dst_addr);

		/* Find destination port */
		if (rte_lpm_lookup(rxq->lpm, ip_dst, &next_hop) == 0 &&
				(enabled_port_mask & 1 << next_hop) != 0) {
			dst_port = next_hop;
		}

		eth_hdr->ether_type = rte_be_to_cpu_16(RTE_ETHER_TYPE_IPV4);
	} else if (RTE_ETH_IS_IPV6_HDR(m->packet_type)) {
		/* if packet is IPv6 */
		struct ipv6_extension_fragment *frag_hdr;
		struct rte_ipv6_hdr *ip_hdr;

		ip_hdr = (struct rte_ipv6_hdr *)(eth_hdr + 1);

		frag_hdr = rte_ipv6_frag_get_ipv6_fragment_header(ip_hdr);

		if (frag_hdr != NULL) {
			struct rte_mbuf *mo;

			tbl = rxq->frag_tbl;
			dr  = &qconf->death_row;

			/* prepare mbuf: setup l2_len/l3_len. */
			m->l2_len = sizeof(*eth_hdr);
			m->l3_len = sizeof(*ip_hdr) + sizeof(*frag_hdr);

			mo = rte_ipv6_frag_reassemble_packet(tbl, dr, m, tms, ip_hdr, frag_hdr);
			if (mo == NULL)
				return;

			if (mo != m) {
				m = mo;
				eth_hdr = rte_pktmbuf_mtod(m,
							struct rte_ether_hdr *);
				ip_hdr = (struct rte_ipv6_hdr *)(eth_hdr + 1);
			}
		}

		/* Find destination port */
		if (rte_lpm6_lookup(rxq->lpm6, ip_hdr->dst_addr,
						&next_hop) == 0 &&
				(enabled_port_mask & 1 << next_hop) != 0) {
			dst_port = next_hop;
		}

		eth_hdr->ether_type = rte_be_to_cpu_16(RTE_ETHER_TYPE_IPV6);
	}
	/* if packet wasn't IPv4 or IPv6, it's forwarded to the port it came from */

#if 0
	/* 02:00:00:00:00:xx */
	d_addr_bytes = &eth_hdr->d_addr.addr_bytes[0];
	*((uint64_t *)d_addr_bytes) = 0x000000000002 + ((uint64_t)dst_port << 40);
#else
                if(CONNECT_81_PORT == portid && CONNECT_82_PORT == dst_port)
                {
		RTE_LOG(INFO, IP_RSMBL, "portid=%u recv packet from 81 server , and next hop %u \n", 
				portid, dst_port);
                }
                if(CONNECT_82_PORT  == portid && CONNECT_81_PORT == dst_port)
                {
		RTE_LOG(INFO, IP_RSMBL, "portid=%u recv packet from 82 server, and next hop %u \n", 
				portid, dst_port);
                }
                if(CONNECT_81_PORT == dst_port) {
                     rte_ether_addr_copy(&srv81_ether_addr, &eth_hdr->d_addr);
                }
                else if(CONNECT_82_PORT == dst_port) {
                    rte_ether_addr_copy(&srv82_ether_addr, &eth_hdr->d_addr);
                }
       //printf("recv from port %u , and send to dst port %u \n",portid, dst_port);
#endif
	/* src addr */
	rte_ether_addr_copy(&ports_eth_addr[dst_port], &eth_hdr->s_addr);
        //print_mac_info(m);
        //read_and_print_ipv4_info(m);
	send_single_packet(m, dst_port);
}

/* main processing loop */
static int
main_loop(__attribute__((unused)) void *dummy)
{
	struct rte_mbuf *pkts_burst[MAX_PKT_BURST];
	unsigned lcore_id;
	uint64_t diff_tsc, cur_tsc, prev_tsc;
	int i, j, nb_rx;
	uint16_t portid;
	struct lcore_queue_conf *qconf;
	const uint64_t drain_tsc = (rte_get_tsc_hz() + US_PER_S - 1) / US_PER_S * BURST_TX_DRAIN_US;

	prev_tsc = 0;

	lcore_id = rte_lcore_id();
	qconf = &lcore_queue_conf[lcore_id];

	if (qconf->n_rx_queue == 0) {
		RTE_LOG(INFO, IP_RSMBL, "lcore %u has nothing to do\n", lcore_id);
		return 0;
	}

	RTE_LOG(INFO, IP_RSMBL, "entering main loop on lcore %u\n", lcore_id);

	for (i = 0; i < qconf->n_rx_queue; i++) {

		portid = qconf->rx_queue_list[i].portid;
		RTE_LOG(INFO, IP_RSMBL, " -- lcoreid=%u portid=%u\n", lcore_id,
			portid);
	}

	while (1) {

		cur_tsc = rte_rdtsc();

		/*
		 * TX burst queue drain
		 */
		diff_tsc = cur_tsc - prev_tsc;
		if (unlikely(diff_tsc > drain_tsc)) {

			/*
			 * This could be optimized (use queueid instead of
			 * portid), but it is not called so often
			 */
			for (portid = 0; portid < RTE_MAX_ETHPORTS; portid++) {
				if ((enabled_port_mask & (1 << portid)) != 0)
					send_burst(qconf, 1, portid);
			}

			prev_tsc = cur_tsc;
		}

		/*
		 * Read packet from RX queues
		 */
		for (i = 0; i < qconf->n_rx_queue; ++i) {

			portid = qconf->rx_queue_list[i].portid;

			nb_rx = rte_eth_rx_burst(portid, 0, pkts_burst,
				MAX_PKT_BURST);

			/* Prefetch first packets */
			for (j = 0; j < PREFETCH_OFFSET && j < nb_rx; j++) {
				rte_prefetch0(rte_pktmbuf_mtod(
						pkts_burst[j], void *));
			}

			/* Prefetch and forward already prefetched packets */
			for (j = 0; j < (nb_rx - PREFETCH_OFFSET); j++) {
				rte_prefetch0(rte_pktmbuf_mtod(pkts_burst[
					j + PREFETCH_OFFSET], void *));
				reassemble(pkts_burst[j], portid,
					i, qconf, cur_tsc);
			}

			/* Forward remaining prefetched packets */
			for (; j < nb_rx; j++) {
				reassemble(pkts_burst[j], portid,
					i, qconf, cur_tsc);
			}

                        if((&qconf->death_row)->cnt > 0){
                        printf(" ********* frag death %u \n ",  (&qconf->death_row)->cnt);
                        }
			rte_ip_frag_free_death_row(&qconf->death_row,
				PREFETCH_OFFSET);
		}
	}
}

/* display usage */
static void
print_usage(const char *prgname)
{
	printf("%s [EAL options] -- -p PORTMASK [-q NQ]"
		"  [--max-pkt-len PKTLEN]"
		"  [--maxflows=<flows>]  [--flowttl=<ttl>[(s|ms)]]\n"
		"  -p PORTMASK: hexadecimal bitmask of ports to configure\n"
		"  -q NQ: number of RX queues per lcore\n"
		"  --maxflows=<flows>: optional, maximum number of flows "
		"supported\n"
		"  --flowttl=<ttl>[(s|ms)]: optional, maximum TTL for each "
		"flow\n",
		prgname);
}

static uint32_t
parse_flow_num(const char *str, uint32_t min, uint32_t max, uint32_t *val)
{
	char *end;
	uint64_t v;

	/* parse decimal string */
	errno = 0;
	v = strtoul(str, &end, 10);
	if (errno != 0 || *end != '\0')
		return -EINVAL;

	if (v < min || v > max)
		return -EINVAL;

	*val = (uint32_t)v;
	return 0;
}

static int
parse_flow_ttl(const char *str, uint32_t min, uint32_t max, uint32_t *val)
{
	char *end;
	uint64_t v;

	static const char frmt_sec[] = "s";
	static const char frmt_msec[] = "ms";

	/* parse decimal string */
	errno = 0;
	v = strtoul(str, &end, 10);
	if (errno != 0)
		return -EINVAL;

	if (*end != '\0') {
		if (strncmp(frmt_sec, end, sizeof(frmt_sec)) == 0)
			v *= MS_PER_S;
		else if (strncmp(frmt_msec, end, sizeof (frmt_msec)) != 0)
			return -EINVAL;
	}

	if (v < min || v > max)
		return -EINVAL;

	*val = (uint32_t)v;
	return 0;
}

static int
parse_portmask(const char *portmask)
{
	char *end = NULL;
	unsigned long pm;

	/* parse hexadecimal string */
	pm = strtoul(portmask, &end, 16);
	if ((portmask[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;

	if (pm == 0)
		return -1;

	return pm;
}

static int
parse_nqueue(const char *q_arg)
{
	char *end = NULL;
	unsigned long n;

	printf("%p\n", q_arg);

	/* parse hexadecimal string */
	n = strtoul(q_arg, &end, 10);
	if ((q_arg[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;
	if (n == 0)
		return -1;
	if (n >= MAX_RX_QUEUE_PER_LCORE)
		return -1;

	return n;
}

/* Parse the argument given in the command line of the application */
static int
parse_args(int argc, char **argv)
{
	int opt, ret;
	char **argvopt;
	int option_index;
	char *prgname = argv[0];
	static struct option lgopts[] = {
		{"max-pkt-len", 1, 0, 0},
		{"maxflows", 1, 0, 0},
		{"flowttl", 1, 0, 0},
		{NULL, 0, 0, 0}
	};

	argvopt = argv;

	while ((opt = getopt_long(argc, argvopt, "p:q:",
				lgopts, &option_index)) != EOF) {

		switch (opt) {
		/* portmask */
		case 'p':
			enabled_port_mask = parse_portmask(optarg);
			if (enabled_port_mask == 0) {
				printf("invalid portmask\n");
				print_usage(prgname);
				return -1;
			}
			break;

		/* nqueue */
		case 'q':
			rx_queue_per_lcore = parse_nqueue(optarg);
			if (rx_queue_per_lcore < 0) {
				printf("invalid queue number\n");
				print_usage(prgname);
				return -1;
			}
			break;

		/* long options */
		case 0:
			if (!strncmp(lgopts[option_index].name,
					"maxflows", 8)) {
				if ((ret = parse_flow_num(optarg, MIN_FLOW_NUM,
						MAX_FLOW_NUM,
						&max_flow_num)) != 0) {
					printf("invalid value: \"%s\" for "
						"parameter %s\n",
						optarg,
						lgopts[option_index].name);
					print_usage(prgname);
					return ret;
				}
			}

			if (!strncmp(lgopts[option_index].name, "flowttl", 7)) {
				if ((ret = parse_flow_ttl(optarg, MIN_FLOW_TTL,
						MAX_FLOW_TTL,
						&max_flow_ttl)) != 0) {
					printf("invalid value: \"%s\" for "
						"parameter %s\n",
						optarg,
						lgopts[option_index].name);
					print_usage(prgname);
					return ret;
				}
			}

			break;

		default:
			print_usage(prgname);
			return -1;
		}
	}

	if (optind >= 0)
		argv[optind-1] = prgname;

	ret = optind-1;
	optind = 1; /* reset getopt lib */
	return ret;
}

static void
print_ethaddr(const char *name, const struct rte_ether_addr *eth_addr)
{
	char buf[RTE_ETHER_ADDR_FMT_SIZE];
	rte_ether_format_addr(buf, RTE_ETHER_ADDR_FMT_SIZE, eth_addr);
	printf("%s%s", name, buf);
}

/* Check the link status of all ports in up to 9s, and print them finally */
static void
check_all_ports_link_status(uint32_t port_mask)
{
#define CHECK_INTERVAL 100 /* 100ms */
#define MAX_CHECK_TIME 90 /* 9s (90 * 100ms) in total */
	uint16_t portid;
	uint8_t count, all_ports_up, print_flag = 0;
	struct rte_eth_link link;
	int ret;

	printf("\nChecking link status");
	fflush(stdout);
	for (count = 0; count <= MAX_CHECK_TIME; count++) {
		all_ports_up = 1;
		RTE_ETH_FOREACH_DEV(portid) {
			if ((port_mask & (1 << portid)) == 0)
				continue;
			memset(&link, 0, sizeof(link));
			ret = rte_eth_link_get_nowait(portid, &link);
			if (ret < 0) {
				all_ports_up = 0;
				if (print_flag == 1)
					printf("Port %u link get failed: %s\n",
						portid, rte_strerror(-ret));
				continue;
			}
			/* print link status if flag set */
			if (print_flag == 1) {
				if (link.link_status)
					printf(
					"Port%d Link Up. Speed %u Mbps - %s\n",
						portid, link.link_speed,
				(link.link_duplex == ETH_LINK_FULL_DUPLEX) ?
					("full-duplex") : ("half-duplex\n"));
				else
					printf("Port %d Link Down\n", portid);
				continue;
			}
			/* clear all_ports_up flag if any link down */
			if (link.link_status == ETH_LINK_DOWN) {
				all_ports_up = 0;
				break;
			}
		}
		/* after finally printing all link status, get out */
		if (print_flag == 1)
			break;

		if (all_ports_up == 0) {
			printf(".");
			fflush(stdout);
			rte_delay_ms(CHECK_INTERVAL);
		}

		/* set the print_flag if all ports up or timeout */
		if (all_ports_up == 1 || count == (MAX_CHECK_TIME - 1)) {
			print_flag = 1;
			printf("\ndone\n");
		}
	}
}

static int
init_routing_table(void)
{
	struct rte_lpm *lpm;
	struct rte_lpm6 *lpm6;
	int socket, ret;
	unsigned i;

	for (socket = 0; socket < RTE_MAX_NUMA_NODES; socket++) {
		if (socket_lpm[socket]) {
			lpm = socket_lpm[socket];
			/* populate the LPM table */
			for (i = 0; i < RTE_DIM(l3fwd_ipv4_route_array); i++) {
				ret = rte_lpm_add(lpm,
					l3fwd_ipv4_route_array[i].ip,
					l3fwd_ipv4_route_array[i].depth,
					l3fwd_ipv4_route_array[i].if_out);

				if (ret < 0) {
					RTE_LOG(ERR, IP_RSMBL, "Unable to add entry %i to the l3fwd "
						"LPM table\n", i);
					return -1;
				}

				RTE_LOG(INFO, IP_RSMBL, "Socket %i: adding route " IPv4_BYTES_FMT
						"/%d (port %d)\n",
					socket,
					IPv4_BYTES(l3fwd_ipv4_route_array[i].ip),
					l3fwd_ipv4_route_array[i].depth,
					l3fwd_ipv4_route_array[i].if_out);
			}
		}

		if (socket_lpm6[socket]) {
			lpm6 = socket_lpm6[socket];
			/* populate the LPM6 table */
			for (i = 0; i < RTE_DIM(l3fwd_ipv6_route_array); i++) {
				ret = rte_lpm6_add(lpm6,
					l3fwd_ipv6_route_array[i].ip,
					l3fwd_ipv6_route_array[i].depth,
					l3fwd_ipv6_route_array[i].if_out);

				if (ret < 0) {
					RTE_LOG(ERR, IP_RSMBL, "Unable to add entry %i to the l3fwd "
						"LPM6 table\n", i);
					return -1;
				}

				RTE_LOG(INFO, IP_RSMBL, "Socket %i: adding route " IPv6_BYTES_FMT
						"/%d (port %d)\n",
					socket,
					IPv6_BYTES(l3fwd_ipv6_route_array[i].ip),
					l3fwd_ipv6_route_array[i].depth,
					l3fwd_ipv6_route_array[i].if_out);
			}
		}
	}
	return 0;
}

static int
setup_port_tbl(struct lcore_queue_conf *qconf, uint32_t lcore, int socket,
	uint32_t port)
{
	struct mbuf_table *mtb;
	uint32_t n;
	size_t sz;

	n = RTE_MAX(max_flow_num, 2UL * MAX_PKT_BURST);
	sz = sizeof (*mtb) + sizeof (mtb->m_table[0]) *  n;

	if ((mtb = rte_zmalloc_socket(__func__, sz, RTE_CACHE_LINE_SIZE,
			socket)) == NULL) {
		RTE_LOG(ERR, IP_RSMBL, "%s() for lcore: %u, port: %u "
			"failed to allocate %zu bytes\n",
			__func__, lcore, port, sz);
		return -1;
	}

	mtb->len = n;
	qconf->tx_mbufs[port] = mtb;

	return 0;
}

static int
setup_queue_tbl(struct rx_queue *rxq, uint32_t lcore, uint32_t queue)
{
	int socket;
	uint32_t nb_mbuf;
	uint64_t frag_cycles;
	char buf[RTE_MEMPOOL_NAMESIZE];

	socket = rte_lcore_to_socket_id(lcore);
	if (socket == SOCKET_ID_ANY)
		socket = 0;

	frag_cycles = (rte_get_tsc_hz() + MS_PER_S - 1) / MS_PER_S *
		max_flow_ttl;

	if ((rxq->frag_tbl = rte_ip_frag_table_create(max_flow_num,
			IP_FRAG_TBL_BUCKET_ENTRIES, max_flow_num, frag_cycles,
			socket)) == NULL) {
		RTE_LOG(ERR, IP_RSMBL, "ip_frag_tbl_create(%u) on "
			"lcore: %u for queue: %u failed\n",
			max_flow_num, lcore, queue);
		return -1;
	}

	/*
	 * At any given moment up to <max_flow_num * (MAX_FRAG_NUM)>
	 * mbufs could be stored int the fragment table.
	 * Plus, each TX queue can hold up to <max_flow_num> packets.
	 */

	nb_mbuf = RTE_MAX(max_flow_num, 2UL * MAX_PKT_BURST) * MAX_FRAG_NUM;
	nb_mbuf *= (port_conf.rxmode.max_rx_pkt_len + BUF_SIZE - 1) / BUF_SIZE;
	nb_mbuf *= 2; /* ipv4 and ipv6 */
	nb_mbuf += nb_rxd + nb_txd;

	nb_mbuf = RTE_MAX(nb_mbuf, (uint32_t)NB_MBUF);

	snprintf(buf, sizeof(buf), "mbuf_pool_%u_%u", lcore, queue);

	rxq->pool = rte_pktmbuf_pool_create(buf, nb_mbuf, MEMPOOL_CACHE_SIZE, 0,
					    MBUF_DATA_SIZE, socket);
	if (rxq->pool == NULL) {
		RTE_LOG(ERR, IP_RSMBL,
			"rte_pktmbuf_pool_create(%s) failed", buf);
		return -1;
	}

	return 0;
}

static int
init_mem(void)
{
	char buf[PATH_MAX];
	struct rte_lpm *lpm;
	struct rte_lpm6 *lpm6;
	struct rte_lpm_config lpm_config;
	int socket;
	unsigned lcore_id;

	/* traverse through lcores and initialize structures on each socket */

	for (lcore_id = 0; lcore_id < RTE_MAX_LCORE; lcore_id++) {

		if (rte_lcore_is_enabled(lcore_id) == 0)
			continue;

		socket = rte_lcore_to_socket_id(lcore_id);

		if (socket == SOCKET_ID_ANY)
			socket = 0;

		if (socket_lpm[socket] == NULL) {
			RTE_LOG(INFO, IP_RSMBL, "Creating LPM table on socket %i\n", socket);
			snprintf(buf, sizeof(buf), "IP_RSMBL_LPM_%i", socket);

			lpm_config.max_rules = LPM_MAX_RULES;
			lpm_config.number_tbl8s = 256;
			lpm_config.flags = 0;

			lpm = rte_lpm_create(buf, socket, &lpm_config);
			if (lpm == NULL) {
				RTE_LOG(ERR, IP_RSMBL, "Cannot create LPM table\n");
				return -1;
			}
			socket_lpm[socket] = lpm;
		}

		if (socket_lpm6[socket] == NULL) {
			RTE_LOG(INFO, IP_RSMBL, "Creating LPM6 table on socket %i\n", socket);
			snprintf(buf, sizeof(buf), "IP_RSMBL_LPM_%i", socket);

			lpm6 = rte_lpm6_create(buf, socket, &lpm6_config);
			if (lpm6 == NULL) {
				RTE_LOG(ERR, IP_RSMBL, "Cannot create LPM table\n");
				return -1;
			}
			socket_lpm6[socket] = lpm6;
		}
	}

	return 0;
}

static void
queue_dump_stat(void)
{
	uint32_t i, lcore;
	const struct lcore_queue_conf *qconf;

	for (lcore = 0; lcore < RTE_MAX_LCORE; lcore++) {
		if (rte_lcore_is_enabled(lcore) == 0)
			continue;

		qconf = &lcore_queue_conf[lcore];
		for (i = 0; i < qconf->n_rx_queue; i++) {

			fprintf(stdout, " -- lcoreid=%u portid=%u "
				"frag tbl stat:\n",
				lcore,  qconf->rx_queue_list[i].portid);
			rte_ip_frag_table_statistics_dump(stdout,
					qconf->rx_queue_list[i].frag_tbl);
			fprintf(stdout, "TX bursts:\t%" PRIu64 "\n"
				"TX packets _queued:\t%" PRIu64 "\n"
				"TX packets dropped:\t%" PRIu64 "\n"
				"TX packets send:\t%" PRIu64 "\n",
				qconf->tx_stat.call,
				qconf->tx_stat.queue,
				qconf->tx_stat.drop,
				qconf->tx_stat.send);
		}
	}
}

static void
signal_handler(int signum)
{
	queue_dump_stat();
	if (signum != SIGUSR1)
		rte_exit(0, "received signal: %d, exiting\n", signum);
}

/* Parse packet type of a packet by SW */
static inline void
parse_ptype(struct rte_mbuf *m)
{
        struct rte_ether_hdr *eth_hdr;
        uint32_t packet_type = RTE_PTYPE_UNKNOWN;
        uint16_t ether_type;

        eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
        ether_type = eth_hdr->ether_type;
        if (ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4))
                packet_type |= RTE_PTYPE_L3_IPV4_EXT_UNKNOWN;
        else if (ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV6))
                packet_type |= RTE_PTYPE_L3_IPV6_EXT_UNKNOWN;

        m->packet_type = packet_type;
}

/* callback function to detect packet type for a queue of a port */
static uint16_t
cb_parse_ptype(uint16_t port __rte_unused, uint16_t queue __rte_unused,
                   struct rte_mbuf *pkts[], uint16_t nb_pkts,
                   uint16_t max_pkts __rte_unused,
                   void *user_param __rte_unused)
{
        uint16_t i;

        for (i = 0; i < nb_pkts; ++i)
                parse_ptype(pkts[i]);

        return nb_pkts;
}
/* Check L3 packet type detection capablity of the NIC port */
static int
check_ptype(int portid)
{
        int i, ret;
        int ptype_l3_ipv4 = 0, ptype_l3_ipv6 = 0;
        uint32_t ptype_mask = RTE_PTYPE_L3_MASK;

        ret = rte_eth_dev_get_supported_ptypes(portid, ptype_mask, NULL, 0);
        if (ret <= 0)
                return 0;

        uint32_t ptypes[ret];

        ret = rte_eth_dev_get_supported_ptypes(portid, ptype_mask, ptypes, ret);
        for (i = 0; i < ret; ++i) {
                if (ptypes[i] & RTE_PTYPE_L3_IPV4)
                        ptype_l3_ipv4 = 1;
                if (ptypes[i] & RTE_PTYPE_L3_IPV6)
                        ptype_l3_ipv6 = 1;
        }

        if (ptype_l3_ipv4 == 0)
                printf("port %d cannot parse RTE_PTYPE_L3_IPV4\n", portid);

        if (ptype_l3_ipv6 == 0)
                printf("port %d cannot parse RTE_PTYPE_L3_IPV6\n", portid);

        if (ptype_l3_ipv4 && ptype_l3_ipv6)
                return 1;

        return 0;

}
int
main(int argc, char **argv)
{
	struct lcore_queue_conf *qconf;
	struct rte_eth_dev_info dev_info;
	struct rte_eth_txconf *txconf;
	struct rx_queue *rxq;
	int ret, socket;
	unsigned nb_ports;
	uint16_t queueid;
	unsigned lcore_id = 0, rx_lcore_id = 0;
	uint32_t n_tx_queue, nb_lcores;
	uint16_t portid;

	/* init EAL */
	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Invalid EAL parameters\n");
	argc -= ret;
	argv += ret;

	/* parse application arguments (after the EAL ones) */
	ret = parse_args(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Invalid IP reassembly parameters\n");

	nb_ports = rte_eth_dev_count_avail();
	if (nb_ports == 0)
		rte_exit(EXIT_FAILURE, "No ports found!\n");

	nb_lcores = rte_lcore_count();

	/* initialize structures (mempools, lpm etc.) */
	if (init_mem() < 0)
		rte_panic("Cannot initialize memory structures!\n");

	/* check if portmask has non-existent ports */
	if (enabled_port_mask & ~(RTE_LEN2MASK(nb_ports, unsigned)))
		rte_exit(EXIT_FAILURE, "Non-existent ports in portmask!\n");

	/* initialize all ports */
	RTE_ETH_FOREACH_DEV(portid) {
		struct rte_eth_rxconf rxq_conf;
		struct rte_eth_conf local_port_conf = port_conf;

		/* skip ports that are not enabled */
		if ((enabled_port_mask & (1 << portid)) == 0) {
			printf("\nSkipping disabled port %d\n", portid);
			continue;
		}

		qconf = &lcore_queue_conf[rx_lcore_id];

		/* limit the frame size to the maximum supported by NIC */
		ret = rte_eth_dev_info_get(portid, &dev_info);
		if (ret != 0)
			rte_exit(EXIT_FAILURE,
				"Error during getting device (port %u) info: %s\n",
				portid, strerror(-ret));

		local_port_conf.rxmode.max_rx_pkt_len = RTE_MIN(
		    dev_info.max_rx_pktlen,
		    local_port_conf.rxmode.max_rx_pkt_len);

		/* get the lcore_id for this port */
		while (rte_lcore_is_enabled(rx_lcore_id) == 0 ||
			   qconf->n_rx_queue == (unsigned)rx_queue_per_lcore) {

			rx_lcore_id++;
			if (rx_lcore_id >= RTE_MAX_LCORE)
				rte_exit(EXIT_FAILURE, "Not enough cores\n");

			qconf = &lcore_queue_conf[rx_lcore_id];
		}

		socket = rte_lcore_to_socket_id(portid);
		if (socket == SOCKET_ID_ANY)
			socket = 0;

		queueid = qconf->n_rx_queue;
		rxq = &qconf->rx_queue_list[queueid];
		rxq->portid = portid;
		rxq->lpm = socket_lpm[socket];
		rxq->lpm6 = socket_lpm6[socket];

		ret = rte_eth_dev_adjust_nb_rx_tx_desc(portid, &nb_rxd,
						       &nb_txd);
		if (ret < 0)
			rte_exit(EXIT_FAILURE,
				 "Cannot adjust number of descriptors: err=%d, port=%d\n",
				 ret, portid);

		if (setup_queue_tbl(rxq, rx_lcore_id, queueid) < 0)
			rte_exit(EXIT_FAILURE, "Failed to set up queue table\n");
		qconf->n_rx_queue++;

		/* init port */
		printf("Initializing port %d ... ", portid );
		fflush(stdout);

		n_tx_queue = nb_lcores;
		if (n_tx_queue > MAX_TX_QUEUE_PER_PORT)
			n_tx_queue = MAX_TX_QUEUE_PER_PORT;
		if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
			local_port_conf.txmode.offloads |=
				DEV_TX_OFFLOAD_MBUF_FAST_FREE;

		local_port_conf.rx_adv_conf.rss_conf.rss_hf &=
			dev_info.flow_type_rss_offloads;
		if (local_port_conf.rx_adv_conf.rss_conf.rss_hf !=
				port_conf.rx_adv_conf.rss_conf.rss_hf) {
			printf("Port %u modified RSS hash function based on hardware support,"
				"requested:%#"PRIx64" configured:%#"PRIx64"\n",
				portid,
				port_conf.rx_adv_conf.rss_conf.rss_hf,
				local_port_conf.rx_adv_conf.rss_conf.rss_hf);
		}

		ret = rte_eth_dev_configure(portid, 1, (uint16_t)n_tx_queue,
					    &local_port_conf);
		if (ret < 0) {
			printf("\n");
			rte_exit(EXIT_FAILURE, "Cannot configure device: "
				"err=%d, port=%d\n",
				ret, portid);
		}

		/* init one RX queue */
		rxq_conf = dev_info.default_rxconf;
		rxq_conf.offloads = local_port_conf.rxmode.offloads;
		ret = rte_eth_rx_queue_setup(portid, 0, nb_rxd,
					     socket, &rxq_conf,
					     rxq->pool);
		if (ret < 0) {
			printf("\n");
			rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup: "
				"err=%d, port=%d\n",
				ret, portid);
		}

		ret = rte_eth_macaddr_get(portid, &ports_eth_addr[portid]);
		if (ret < 0) {
			printf("\n");
			rte_exit(EXIT_FAILURE,
				"rte_eth_macaddr_get: err=%d, port=%d\n",
				ret, portid);
		}

		print_ethaddr(" Address:", &ports_eth_addr[portid]);
		printf("\n");

		/* init one TX queue per couple (lcore,port) */
		queueid = 0;
		for (lcore_id = 0; lcore_id < RTE_MAX_LCORE; lcore_id++) {
			if (rte_lcore_is_enabled(lcore_id) == 0)
				continue;

			socket = (int) rte_lcore_to_socket_id(lcore_id);

			printf("txq=%u,%d,%d ", lcore_id, queueid, socket);
			fflush(stdout);

			txconf = &dev_info.default_txconf;
			txconf->offloads = local_port_conf.txmode.offloads;

			ret = rte_eth_tx_queue_setup(portid, queueid, nb_txd,
					socket, txconf);
			if (ret < 0)
				rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup: err=%d, "
					"port=%d\n", ret, portid);

			qconf = &lcore_queue_conf[lcore_id];
			qconf->tx_queue_id[portid] = queueid;
			setup_port_tbl(qconf, lcore_id, socket, portid);
			queueid++;
		}
		printf("\n");
	}

	printf("\n");

	/* start ports */
	RTE_ETH_FOREACH_DEV(portid) {
		if ((enabled_port_mask & (1 << portid)) == 0) {
			continue;
		}
		/* Start device */
		ret = rte_eth_dev_start(portid);
		if (ret < 0)
			rte_exit(EXIT_FAILURE, "rte_eth_dev_start: err=%d, port=%d\n",
				ret, portid);

		ret = rte_eth_promiscuous_enable(portid);
		if (ret != 0)
			rte_exit(EXIT_FAILURE,
				"rte_eth_promiscuous_enable: err=%s, port=%d\n",
				rte_strerror(-ret), portid);

                if (check_ptype(portid) == 0) {
                        rte_eth_add_rx_callback(portid, 0, cb_parse_ptype, NULL);
                        printf("Add Rx callback function to detect L3 packet type by SW :"
                                " port = %d\n", portid);
                }
	}

	if (init_routing_table() < 0)
		rte_exit(EXIT_FAILURE, "Cannot init routing table\n");

	check_all_ports_link_status(enabled_port_mask);

	signal(SIGUSR1, signal_handler);
	signal(SIGTERM, signal_handler);
	signal(SIGINT, signal_handler);

	/* launch per-lcore init on every lcore */
	rte_eal_mp_remote_launch(main_loop, NULL, CALL_MASTER);
	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		if (rte_eal_wait_lcore(lcore_id) < 0)
			return -1;
	}

	return 0;
}
