/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2017 Mellanox Technologies, Ltd
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/queue.h>
#include <netinet/in.h>
#include <setjmp.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>

#include <rte_version.h>
#include <rte_eal.h>
#include <rte_common.h>
#include <rte_malloc.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_net.h>
#include <rte_cycles.h>
#include "flow_blocks.h"
#include "dpdk.h"
#ifdef RTE_NET_I40E
#include <rte_pmd_i40e.h>
#endif
#define RTE_PORT_ALL            (~(uint16_t)0x0)
static volatile bool force_quit;

static uint16_t port_id;
static uint8_t selected_queue = 1;
uint16_t nr_queues = 8;
struct rte_mempool *mbuf_pool;
struct rte_flow *flow;
#define SRC_IP ((0<<24) + (0<<16) + (0<<8) + 0) /* src ip = 0.0.0.0 */
#define DEST_IP ((192<<24) + (168<<16) + (1<<8) + 1) /* dest ip = 192.168.1.1 */
#define FULL_MASK 0xffffffff /* full mask */
#define EMPTY_MASK 0x0 /* empty mask */
#define IPV4_ADDR(a, b, c, d)           (((d & 0xff) << 24) | ((c & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff))

#define IP_BUF_LEN 24
#define NUM_OF_DESC (512)
#define I40E_RSS_HKEY_LEN 52
struct rte_eth_rss_conf rss_conf;
const char * dst_ip = "10.11.11.65";
enum print_warning {
        ENABLED_WARN = 0,
        DISABLED_WARN
};
static const char *
ip_proto_name(uint16_t ip_proto)
{
	static const char * ip_proto_names[] = {
		"IP6HOPOPTS", /**< IP6 hop-by-hop options */
		"ICMP",       /**< control message protocol */
		"IGMP",       /**< group mgmt protocol */
		"GGP",        /**< gateway^2 (deprecated) */
		"IPv4",       /**< IPv4 encapsulation */

		"UNASSIGNED",
		"TCP",        /**< transport control protocol */
		"ST",         /**< Stream protocol II */
		"EGP",        /**< exterior gateway protocol */
		"PIGP",       /**< private interior gateway */

		"RCC_MON",    /**< BBN RCC Monitoring */
		"NVPII",      /**< network voice protocol*/
		"PUP",        /**< pup */
		"ARGUS",      /**< Argus */
		"EMCON",      /**< EMCON */

		"XNET",       /**< Cross Net Debugger */
		"CHAOS",      /**< Chaos*/
		"UDP",        /**< user datagram protocol */
		"MUX",        /**< Multiplexing */
		"DCN_MEAS",   /**< DCN Measurement Subsystems */

		"HMP",        /**< Host Monitoring */
		"PRM",        /**< Packet Radio Measurement */
		"XNS_IDP",    /**< xns idp */
		"TRUNK1",     /**< Trunk-1 */
		"TRUNK2",     /**< Trunk-2 */

		"LEAF1",      /**< Leaf-1 */
		"LEAF2",      /**< Leaf-2 */
		"RDP",        /**< Reliable Data */
		"IRTP",       /**< Reliable Transaction */
		"TP4",        /**< tp-4 w/ class negotiation */

		"BLT",        /**< Bulk Data Transfer */
		"NSP",        /**< Network Services */
		"INP",        /**< Merit Internodal */
		"SEP",        /**< Sequential Exchange */
		"3PC",        /**< Third Party Connect */

		"IDPR",       /**< InterDomain Policy Routing */
		"XTP",        /**< XTP */
		"DDP",        /**< Datagram Delivery */
		"CMTP",       /**< Control Message Transport */
		"TPXX",       /**< TP++ Transport */

		"ILTP",       /**< IL transport protocol */
		"IPv6_HDR",   /**< IP6 header */
		"SDRP",       /**< Source Demand Routing */
		"IPv6_RTG",   /**< IP6 routing header */
		"IPv6_FRAG",  /**< IP6 fragmentation header */

		"IDRP",       /**< InterDomain Routing*/
		"RSVP",       /**< resource reservation */
		"GRE",        /**< General Routing Encap. */
		"MHRP",       /**< Mobile Host Routing */
		"BHA",        /**< BHA */

		"ESP",        /**< IP6 Encap Sec. Payload */
		"AH",         /**< IP6 Auth Header */
		"INLSP",      /**< Integ. Net Layer Security */
		"SWIPE",      /**< IP with encryption */
		"NHRP",       /**< Next Hop Resolution */

		"UNASSIGNED",
		"UNASSIGNED",
		"UNASSIGNED",
		"ICMPv6",     /**< ICMP6 */
		"IPv6NONEXT", /**< IP6 no next header */

		"Ipv6DSTOPTS",/**< IP6 destination option */
		"AHIP",       /**< any host internal protocol */
		"CFTP",       /**< CFTP */
		"HELLO",      /**< "hello" routing protocol */
		"SATEXPAK",   /**< SATNET/Backroom EXPAK */

		"KRYPTOLAN",  /**< Kryptolan */
		"RVD",        /**< Remote Virtual Disk */
		"IPPC",       /**< Pluribus Packet Core */
		"ADFS",       /**< Any distributed FS */
		"SATMON",     /**< Satnet Monitoring */

		"VISA",       /**< VISA Protocol */
		"IPCV",       /**< Packet Core Utility */
		"CPNX",       /**< Comp. Prot. Net. Executive */
		"CPHB",       /**< Comp. Prot. HeartBeat */
		"WSN",        /**< Wang Span Network */

		"PVP",        /**< Packet Video Protocol */
		"BRSATMON",   /**< BackRoom SATNET Monitoring */
		"ND",         /**< Sun net disk proto (temp.) */
		"WBMON",      /**< WIDEBAND Monitoring */
		"WBEXPAK",    /**< WIDEBAND EXPAK */

		"EON",        /**< ISO cnlp */
		"VMTP",       /**< VMTP */
		"SVMTP",      /**< Secure VMTP */
		"VINES",      /**< Banyon VINES */
		"TTP",        /**< TTP */

		"IGP",        /**< NSFNET-IGP */
		"DGP",        /**< dissimilar gateway prot. */
		"TCF",        /**< TCF */
		"IGRP",       /**< Cisco/GXS IGRP */
		"OSPFIGP",    /**< OSPFIGP */

		"SRPC",       /**< Strite RPC protocol */
		"LARP",       /**< Locus Address Resolution */
		"MTP",        /**< Multicast Transport */
		"AX25",       /**< AX.25 Frames */
		"4IN4",       /**< IP encapsulated in IP */

		"MICP",       /**< Mobile Int.ing control */
		"SCCSP",      /**< Semaphore Comm. security */
		"ETHERIP",    /**< Ethernet IP encapsulation */
		"ENCAP",      /**< encapsulation header */
		"AES",        /**< any private encr. scheme */

		"GMTP",       /**< GMTP */
		"IPCOMP",     /**< payload compression (IPComp) */
		"UNASSIGNED",
		"UNASSIGNED",
		"PIM",        /**< Protocol Independent Mcast */
	};

	if (ip_proto < sizeof(ip_proto_names) / sizeof(ip_proto_names[0]))
		return ip_proto_names[ip_proto];
	switch (ip_proto) {
#ifdef IPPROTO_PGM
	case IPPROTO_PGM:  /**< PGM */
		return "PGM";
#endif
	case IPPROTO_SCTP:  /**< Stream Control Transport Protocol */
		return "SCTP";
#ifdef IPPROTO_DIVERT
	case IPPROTO_DIVERT: /**< divert pseudo-protocol */
		return "DIVERT";
#endif
	case IPPROTO_RAW: /**< raw IP packet */
		return "RAW";
	default:
		break;
	}
	return "UNASSIGNED";
}
/* helper functions */
static inline struct rte_ipv4_hdr *ip4_hdr(const struct rte_mbuf *mbuf)
{
    /* can only invoked at L3 */
    return rte_pktmbuf_mtod_offset(mbuf, struct rte_ipv4_hdr *,sizeof(struct rte_ether_hdr));
}

static inline uint16_t ip4_hdrlen(const struct rte_mbuf *mbuf)
{
    return (ip4_hdr(mbuf)->version_ihl & 0xf) << 2;
}
static inline void
print_ether_addr(const char *what, struct rte_ether_addr *eth_addr)
{
	char buf[RTE_ETHER_ADDR_FMT_SIZE];
	rte_ether_format_addr(buf, RTE_ETHER_ADDR_FMT_SIZE, eth_addr);
	printf("%s%s", what, buf);
}
static void ip_format_addr(char *buf, uint16_t size,const uint32_t ip_addr)
{
    snprintf(buf, size, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8 ,
             (uint8_t)((ip_addr >> 24) & 0xff),
             (uint8_t)((ip_addr >> 16) & 0xff),
             (uint8_t)((ip_addr >> 8) & 0xff),
             (uint8_t)((ip_addr)&0xff));
}
void udpdk_dump_eth(struct rte_mbuf *m)
{
     struct rte_ether_hdr *eth_hdr;
     struct rte_ipv4_hdr *ip_hdr;
     uint32_t ip;
     uint16_t eth_type, total_length;
     uint16_t hdr_len; 
     char buf[IP_BUF_LEN] = {0};
     eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
     eth_type = rte_cpu_to_be_16(eth_hdr->ether_type);
     inet_pton(AF_INET, dst_ip, &ip);
     if (eth_type ==  RTE_ETHER_TYPE_IPV4)
     {
          ip_hdr = (struct rte_ipv4_hdr *)((char *)eth_hdr + sizeof(struct rte_ether_hdr));
          total_length = rte_be_to_cpu_16(ip_hdr->total_length);
          if(ip != ip_hdr->dst_addr){
              return;
          }
          printf("ip packet id %d \t,",rte_be_to_cpu_32(ip_hdr->packet_id));
          memset(buf,IP_BUF_LEN,0);
          ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(ip_hdr->src_addr)); 
          printf("src ip : %s, ",buf);
          memset(buf,IP_BUF_LEN,0);
          ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(ip_hdr->dst_addr)); 
          printf("dst ip : %s",buf);
          hdr_len = ip4_hdrlen(m);
          printf(" payload len %u, ip hdrlen %u, next proto %s \n",total_length , hdr_len,ip_proto_name(ip_hdr->next_proto_id));
          if(ip_hdr->next_proto_id == IPPROTO_UDP)
          {
              struct rte_udp_hdr * udp_hdr = (struct rte_udp_hdr *)(ip_hdr + 1);
              printf("udp dst port %u \n",rte_be_to_cpu_16(udp_hdr->dst_port));
          }
      }
}
static int
get_fdir_info(uint16_t port_id, struct rte_eth_fdir_info *fdir_info,
                    struct rte_eth_fdir_stats *fdir_stat)
{
        int ret = -ENOTSUP;

#ifdef RTE_NET_I40E
        if (ret == -ENOTSUP) {
                ret = rte_pmd_i40e_get_fdir_info(port_id, fdir_info);
                if (!ret)
                        ret = rte_pmd_i40e_get_fdir_stats(port_id, fdir_stat);
        }
#endif
#ifdef RTE_NET_IXGBE
        if (ret == -ENOTSUP) {
                ret = rte_pmd_ixgbe_get_fdir_info(port_id, fdir_info);
                if (!ret)
                        ret = rte_pmd_ixgbe_get_fdir_stats(port_id, fdir_stat);
        }
#endif
        switch (ret) {
        case 0:
                break;
        case -ENOTSUP:
                fprintf(stderr, "\n FDIR is not supported on port %-2d\n",
                        port_id);
                break;
        default:
                fprintf(stderr, "programming error: (%s)\n", strerror(-ret));
                break;
        }
        return ret;
}

static int port_id_is_invalid(uint16_t port_id, enum print_warning warning)
{
        uint16_t pid;

        if (port_id == (uint16_t)RTE_PORT_ALL)
                return 0;

        RTE_ETH_FOREACH_DEV(pid)
                if (port_id == pid)
                        return 0;

        if (warning == ENABLED_WARN)
                fprintf(stderr, "Invalid port %d\n", port_id);

        return 1;
}
void fdir_get_infos(uint16_t port_id)
{
        struct rte_eth_fdir_stats fdir_stat;
        struct rte_eth_fdir_info fdir_info;

        static const char *fdir_stats_border = "########################";

        if (port_id_is_invalid(port_id, ENABLED_WARN))
                return;

        memset(&fdir_info, 0, sizeof(fdir_info));
        memset(&fdir_stat, 0, sizeof(fdir_stat));
        if (get_fdir_info(port_id, &fdir_info, &fdir_stat))
                return;

        printf("\n  %s FDIR infos for port %-2d     %s\n",
               fdir_stats_border, port_id, fdir_stats_border);
        printf("  MODE: ");
        if (fdir_info.mode == RTE_FDIR_MODE_PERFECT)
                printf("  PERFECT\n");
        else if (fdir_info.mode == RTE_FDIR_MODE_PERFECT_MAC_VLAN)
                printf("  PERFECT-MAC-VLAN\n");
        else if (fdir_info.mode == RTE_FDIR_MODE_PERFECT_TUNNEL)
                printf("  PERFECT-TUNNEL\n");
        else if (fdir_info.mode == RTE_FDIR_MODE_SIGNATURE)
                printf("  SIGNATURE\n");
        else
                printf("  DISABLE\n");
}
static void
main_loop(void)
{
	struct rte_mbuf *mbufs[32];
	struct rte_ether_hdr *eth_hdr;
	struct rte_flow_error error;
	uint16_t nb_rx;
	uint16_t i;
	uint16_t j;

	while (!force_quit) {
		for (i = 0; i < nr_queues; i++) {
			nb_rx = rte_eth_rx_burst(port_id,
						i, mbufs, 32);
			if (nb_rx) {
				for (j = 0; j < nb_rx; j++) {
					struct rte_mbuf *m = mbufs[j];

					eth_hdr = rte_pktmbuf_mtod(m,
							struct rte_ether_hdr *);
#if RTE_VERSION >= RTE_VERSION_NUM(20, 0, 0, 0)
					print_ether_addr("src=",
							&eth_hdr->src_addr);
					print_ether_addr(" - dst=",
							&eth_hdr->dst_addr);
#else
					print_ether_addr("src=",
							&eth_hdr->s_addr);
					print_ether_addr(" - dst=",
							&eth_hdr->d_addr);
#endif
					printf(" - queue=0x%x    ", (unsigned int)i);
                                        udpdk_dump_eth(m);
                                        uint64_t ol_flags = m->ol_flags;
	                                if (ol_flags & PKT_RX_RSS_HASH) {
	                                	printf(" - RSS hash=0x%x", (unsigned int) m->hash.rss);
	                                	printf(" - RSS queue=0x%x", (unsigned int) i);
	                                }
	                                if (ol_flags & PKT_RX_FDIR) {
	                                	printf(" - FDIR matched ");
	                                	if (ol_flags & PKT_RX_FDIR_ID)
	                                		printf("ID=0x%x", m->hash.fdir.hi);
	                                	else if (ol_flags & PKT_RX_FDIR_FLX)
	                                		printf("flex bytes=0x%08x %08x",
	                                		       m->hash.fdir.hi, m->hash.fdir.lo);
	                                	else
	                                		printf("hash=0x%x ID=0x%x ",
	                                		       m->hash.fdir.hash, m->hash.fdir.id);
	                                }
					printf("\n");
					rte_pktmbuf_free(m);
				}
			}
		}
	}

	/* closing and releasing resources */
	rte_flow_flush(port_id, &error);
	rte_eth_dev_stop(port_id);
	rte_eth_dev_close(port_id);
}

#define CHECK_INTERVAL 1000  /* 100ms */
#define MAX_REPEAT_TIMES 90  /* 9s (90 * 100ms) in total */

static void
assert_link_status(void)
{
	struct rte_eth_link link;
	uint8_t rep_cnt = MAX_REPEAT_TIMES;
	int link_get_err = -EINVAL;

	memset(&link, 0, sizeof(link));
	do {
		link_get_err = rte_eth_link_get(port_id, &link);
		if (link_get_err == 0 && link.link_status == ETH_LINK_UP)
			break;
		rte_delay_ms(CHECK_INTERVAL);
	} while (--rep_cnt);

	if (link_get_err < 0)
		rte_exit(EXIT_FAILURE, ":: error: link get is failing: %s\n",
			 rte_strerror(-link_get_err));
	if (link.link_status == ETH_LINK_DOWN)
		rte_exit(EXIT_FAILURE, ":: error: link is still down\n");
}
#if CONFIG_FDIR
static enum rte_fdir_mode g_fdir_mode = RTE_FDIR_MODE_PERFECT;
static int config_fdir_conf(struct rte_fdir_conf *fdir_conf)
{
    int shift;

#if 0
    /* how many mask bits needed? */
    for (shift = 0; (0x1<<shift) < g_slave_lcore_num; shift++)
        ;
    if (shift >= 16)
        return -1;

    fdir_conf->mask.dst_port_mask = htons(~((~0x0) << shift));
#endif
    fdir_conf->mode = g_fdir_mode;

    return 0;
}
#endif
static void
init_port(void)
{
	int ret;
	uint16_t i;
	struct rte_eth_conf port_conf = {
#if RTE_VERSION < RTE_VERSION_NUM(20, 0, 0, 0)
		.rxmode = {
                        .mq_mode = ETH_MQ_RX_NONE,
			.split_hdr_size = 0,
		},
#else
                .rxmode = {
                //.mq_mode = RTE_ETH_MQ_RX_NONE,
                .max_lro_pkt_size = 1024,
                //.max_rx_pkt_len = RTE_ETHER_MAX_LEN,
                },
#endif
		.txmode = {
			.offloads =
				DEV_TX_OFFLOAD_VLAN_INSERT |
				DEV_TX_OFFLOAD_IPV4_CKSUM  |
				DEV_TX_OFFLOAD_UDP_CKSUM   |
				DEV_TX_OFFLOAD_TCP_CKSUM   |
				DEV_TX_OFFLOAD_SCTP_CKSUM  |
				DEV_TX_OFFLOAD_TCP_TSO,
		},
	};
	struct rte_eth_txconf txq_conf;
	struct rte_eth_rxconf rxq_conf;
	struct rte_eth_dev_info dev_info;
        struct rte_eth_conf dev_conf;
	ret = rte_eth_dev_info_get(port_id, &dev_info);
	if (ret != 0)
		rte_exit(EXIT_FAILURE,
			"Error during getting device (port %u) info: %s\n",
			port_id, strerror(-ret));

#if 0
        rss_init();
        rss_config_port(&dev_conf,&dev_info);
#endif
        if(dev_info.flow_type_rss_offloads & RTE_ETH_RSS_IPV4) {
            printf("support RTE_ETH_RSS_IPV4 \n");
        }
        if(dev_info.flow_type_rss_offloads &  RTE_ETH_RSS_FRAG_IPV4) {
            printf("support  RTE_ETH_RSS_FRAG_IPV4\n");
        }
        if(dev_info.flow_type_rss_offloads &  RTE_ETH_RSS_NONFRAG_IPV4_UDP) {
            printf("support  RTE_ETH_RSS_NONFRAG_IPV4_UDP\n");
        }
        if(dev_info.flow_type_rss_offloads &  RTE_ETH_RSS_NONFRAG_IPV4_TCP) {
            printf("support  RTE_ETH_RSS_NONFRAG_IPV4_TCP\n");
        }
	port_conf.txmode.offloads &= dev_info.tx_offload_capa;
#if 0
        port_conf.rx_adv_conf.rss_conf = dev_conf.rx_adv_conf.rss_conf;
        port_conf.rxmode.mq_mode = dev_conf.rxmode.mq_mode; 
#endif
	printf(":: initializing port: %d\n", port_id);
	ret = rte_eth_dev_configure(port_id,
				nr_queues, nr_queues, &port_conf);
	if (ret < 0) {
		rte_exit(EXIT_FAILURE,
			":: cannot configure device: err=%d, port=%u\n",
			ret, port_id);
	}

	rxq_conf = dev_info.default_rxconf;
	rxq_conf.offloads = port_conf.rxmode.offloads;
	for (i = 0; i < nr_queues; i++) {
                //printf("setup %u queue \n",i);
		ret = rte_eth_rx_queue_setup(port_id, i, NUM_OF_DESC,
				     rte_eth_dev_socket_id(port_id),
				     &rxq_conf,
				     mbuf_pool);
		if (ret < 0) {
			rte_exit(EXIT_FAILURE,
				":: Rx queue setup failed: err=%d, port=%u\n",
				ret, port_id);
		}
	}

	txq_conf = dev_info.default_txconf;
	txq_conf.offloads = port_conf.txmode.offloads;

	for (i = 0; i < nr_queues; i++) {
		ret = rte_eth_tx_queue_setup(port_id, i, NUM_OF_DESC,
				rte_eth_dev_socket_id(port_id),
				&txq_conf);
		if (ret < 0) {
			rte_exit(EXIT_FAILURE,
				":: Tx queue setup failed: err=%d, port=%u\n",
				ret, port_id);
		}
	}

	ret = rte_eth_promiscuous_enable(port_id);
	if (ret != 0)
		rte_exit(EXIT_FAILURE,
			":: promiscuous mode enable failed: err=%s, port=%u\n",
			rte_strerror(-ret), port_id);

	ret = rte_eth_dev_start(port_id);
	if (ret < 0) {
		rte_exit(EXIT_FAILURE,
			"rte_eth_dev_start:err=%d, port=%u\n",
			ret, port_id);
	}

	assert_link_status();

	printf(":: initializing port: %d done\n", port_id);
}

static void
signal_handler(int signum)
{
	if (signum == SIGINT || signum == SIGTERM) {
		printf("\n\nSignal %d received, preparing to exit...\n",
				signum);
		force_quit = true;
	}
}

int
main(int argc, char **argv)
{
	int ret;
        uint32_t ip;
	uint16_t nr_ports;
	struct rte_flow_error error;

        struct rte_ether_addr server_ether_addr;
        char buf[IP_BUF_LEN] = {0};
	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, ":: invalid EAL arguments\n");

	force_quit = false;
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

        inet_pton(AF_INET, dst_ip, &ip);
	nr_ports = rte_eth_dev_count_avail();
	if (nr_ports == 0)
		rte_exit(EXIT_FAILURE, ":: no Ethernet ports found\n");
	port_id = 0;
	if (nr_ports != 1) {
		printf(":: warn: %d ports detected, but we use only one: port %u\n",
			nr_ports, port_id);
	}
        ret = rte_eth_macaddr_get(port_id, &server_ether_addr);
        if (ret != 0)
             rte_exit(EXIT_FAILURE, "macaddr get failed\n");
        print_ether_addr("server mac: ", &server_ether_addr);
        printf("\n");
	mbuf_pool = rte_pktmbuf_pool_create("mbuf_pool", 4096, 128, 0,
					    RTE_MBUF_DEFAULT_BUF_SIZE,
					    rte_socket_id());
	if (mbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");

	init_port();
        rte_flow_flush(port_id, NULL);
        flow_new(port_id, 3,0,ip,5000);
	main_loop();
        //generate_udp_fdir_rule(port_id, 4,rte_be_to_cpu_32(ip),5000);
	return 0;
}
