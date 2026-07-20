#include <signal.h>
#include <stdbool.h>
#include <getopt.h>

#include <rte_byteorder.h>
#include <rte_log.h>
#include <rte_common.h>
#include <rte_config.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <rte_log.h>
#include "help.h"
#include <rte_cycles.h>
#define US_PER_S 1000000

#define APP "pingpong"

uint32_t PINGPONG_LOG_LEVEL = RTE_LOG_DEBUG;

/* the client side */
//48:57:02:64:ea:1e
/*
static struct rte_ether_addr client_ether_addr =
    {{0x48, 0x57, 0x02, 0x64, 0xea, 0x1e}};
static uint32_t client_ip_addr = RTE_IPV4(10,10,103,81);
*/
/* the server side */
// 44:a1:91:a4:9b:eb
static struct rte_ether_addr server_ether_addr =
    {{0x44, 0xa1, 0x91, 0xa4, 0x9b, 0xeb}};
static uint32_t server_ip_addr = RTE_IPV4(10,10,103,251);

//static uint16_t cfg_udp_src = 1000;
//static uint16_t cfg_udp_dst = 1001;

#define MAX_PKT_BURST 32
#define MEMPOOL_CACHE_SIZE 128

/*
 * Configurable number of RX/TX ring descriptors
 */
#define RTE_TEST_RX_DESC_DEFAULT 1024
#define RTE_TEST_TX_DESC_DEFAULT 4
static uint16_t nb_rxd = RTE_TEST_RX_DESC_DEFAULT;
static uint16_t nb_txd = RTE_TEST_TX_DESC_DEFAULT;

int RTE_LOGTYPE_PINGPONG;

struct rte_mempool *pingpong_pktmbuf_pool = NULL;

static volatile bool force_quit;

/* enabled port */
static uint16_t portid = 0;
/* number of packets */
static uint64_t nb_pkts = 100;
/* server mode */
static bool server_mode = false;

static struct rte_eth_dev_tx_buffer *tx_buffer;

void reply_to_icmp_echo_rqsts(void);
void server_loop(void);
static struct rte_eth_conf port_conf = {
    .rxmode = {
        .split_hdr_size = 0,
    },
    .txmode = {
        .mq_mode = ETH_MQ_TX_NONE,
    },
};

/* Per-port statistics struct */
struct pingpong_port_statistics
{
    uint64_t tx;
    uint64_t rx;
    uint64_t *rtt;
    uint64_t dropped;
} __rte_cache_aligned;
struct pingpong_port_statistics port_statistics;

static void rte_ether_format_addr(char *buf, uint16_t size,
                      const struct rte_ether_addr *eth_addr)
{
        snprintf(buf, size, "%02X:%02X:%02X:%02X:%02X:%02X",
                 eth_addr->addr_bytes[0],
                 eth_addr->addr_bytes[1],
                 eth_addr->addr_bytes[2],
                 eth_addr->addr_bytes[3],
                 eth_addr->addr_bytes[4],
                 eth_addr->addr_bytes[5]);
}
static inline void
initlize_port_statistics(void)
{
    port_statistics.tx = 0;
    port_statistics.rx = 0;
    port_statistics.rtt = malloc(sizeof(uint64_t) * nb_pkts);
    port_statistics.dropped = 0;
}

static inline void
destroy_port_statistics(void)
{
    free(port_statistics.rtt);
}

static inline void
print_port_statistics(void)
{
    uint64_t i, min_rtt, max_rtt, sum_rtt, avg_rtt;
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "====== ping-pong statistics =====\n");
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "tx %" PRIu64 " ping packets\n", port_statistics.tx);
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "rx %" PRIu64 " pong packets\n", port_statistics.rx);
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "dopped %" PRIu64 " packets\n", port_statistics.dropped);

    min_rtt = 999999999;
    max_rtt = 0;
    sum_rtt = 0;
    avg_rtt = 0;
    for (i = 0; i < nb_pkts; i++)
    {
        sum_rtt += port_statistics.rtt[i];
        if (port_statistics.rtt[i] < min_rtt)
            min_rtt = port_statistics.rtt[i];
        if (port_statistics.rtt[i] > max_rtt)
            max_rtt = port_statistics.rtt[i];
    }
    avg_rtt = sum_rtt / nb_pkts;
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "min rtt: %" PRIu64 " us\n", min_rtt);
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "max rtt: %" PRIu64 " us\n", max_rtt);
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "average rtt: %" PRIu64 " us\n", avg_rtt);
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "=================================\n");
}

static const char short_options[] =
    "p:" /* portmask */
    "n:" /* number of packets */
    "s"  /* server mode */
    ;

#define IP_DEFTTL 64 /* from RFC 1340. */
#define IP_VERSION 0x40
#define IP_HDRLEN 0x05 /* default IP header length == five 32-bits words. */
#define IP_VHL_DEF (IP_VERSION | IP_HDRLEN)
#define IP_ADDR_FMT_SIZE 15

static inline void
ip_format_addr(char *buf, uint16_t size,
               const uint32_t ip_addr)
{
    snprintf(buf, size, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8 "\n",
             (uint8_t)((ip_addr >> 24) & 0xff),
             (uint8_t)((ip_addr >> 16) & 0xff),
             (uint8_t)((ip_addr >> 8) & 0xff),
             (uint8_t)((ip_addr)&0xff));
}

static inline uint32_t
reverse_ip_addr(const uint32_t ip_addr)
{
    return RTE_IPV4((uint8_t)(ip_addr & 0xff),
                (uint8_t)((ip_addr >> 8) & 0xff),
                (uint8_t)((ip_addr >> 16) & 0xff),
                (uint8_t)((ip_addr >> 24) & 0xff));
}

static void
signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM)
    {
        rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "\n\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}

/* display usage */
static void
pingpong_usage(const char *prgname)
{
    printf("%s [EAL options] --"
           "\t-p PORTID: port to configure\n"
           "\t\t\t\t\t-n PACKETS: number of packets\n"
           "\t\t\t\t\t-s: enable server mode\n",
           prgname);
}

/* Parse the argument given in the command line of the application */
static int
pingpong_parse_args(int argc, char **argv)
{
    int opt, ret;
    char *prgname = argv[0];

    while ((opt = getopt(argc, argv, short_options)) != EOF)
    {
        switch (opt)
        {
        /* port id */
        case 'p':
            portid = (uint16_t)strtol(optarg, NULL, 10);
            break;

        case 'n':
            nb_pkts = (uint64_t)strtoull(optarg, NULL, 10);
            break;

        case 's':
            server_mode = true;
            break;

        default:
            pingpong_usage(prgname);
            return -1;
        }
    }

    if (optind >= 0)
        argv[optind - 1] = prgname;

    ret = optind - 1;
    optind = 1; /* reset getopt lib */
    return ret;
}

static inline uint16_t
ip_sum(const unaligned_uint16_t *hdr, int hdr_len)
{
    uint32_t sum = 0;

    while (hdr_len > 1)
    {
        sum += *hdr++;
        if (sum & 0x80000000)
            sum = (sum & 0xFFFF) + (sum >> 16);
        hdr_len -= 2;
    }

    while (sum >> 16)
        sum = (sum & 0xFFFF) + (sum >> 16);

    return ~sum;
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

static void
ipv4_addr_to_dot(uint32_t be_ipv4_addr, char *buf)
{
	uint32_t ipv4_addr;

	ipv4_addr = rte_be_to_cpu_32(be_ipv4_addr);
	sprintf(buf, "%d.%d.%d.%d", (ipv4_addr >> 24) & 0xFF,
		(ipv4_addr >> 16) & 0xFF, (ipv4_addr >> 8) & 0xFF,
		ipv4_addr & 0xFF);
}

static void
ether_addr_dump(const char *what, const struct rte_ether_addr *ea)
{
	char buf[RTE_ETHER_ADDR_FMT_SIZE];

	rte_ether_format_addr(buf, RTE_ETHER_ADDR_FMT_SIZE, ea);
	if (what)
		printf("%s", what);
	printf("%s", buf);
}

static void
ipv4_addr_dump(const char *what, uint32_t be_ipv4_addr)
{
	char buf[16];

	ipv4_addr_to_dot(be_ipv4_addr, buf);
	if (what)
		printf("%s", what);
	printf("%s", buf);
}

static uint16_t
ipv4_hdr_cksum(struct rte_ipv4_hdr *ip_h)
{
	uint16_t *v16_h;
	uint32_t ip_cksum;

	/*
	 * Compute the sum of successive 16-bit words of the IPv4 header,
	 * skipping the checksum field of the header.
	 */
	v16_h = (unaligned_uint16_t *) ip_h;
	ip_cksum = v16_h[0] + v16_h[1] + v16_h[2] + v16_h[3] +
		v16_h[4] + v16_h[6] + v16_h[7] + v16_h[8] + v16_h[9];

	/* reduce 32 bit checksum to 16 bits and complement it */
	ip_cksum = (ip_cksum & 0xffff) + (ip_cksum >> 16);
	ip_cksum = (ip_cksum & 0xffff) + (ip_cksum >> 16);
	ip_cksum = (~ip_cksum) & 0x0000FFFF;
	return (ip_cksum == 0) ? 0xFFFF : (uint16_t) ip_cksum;
}

#define is_multicast_ipv4_addr(ipv4_addr) \
	(((rte_be_to_cpu_32((ipv4_addr)) >> 24) & 0x000000FF) == 0xE0)
/*
 * Receive a burst of packets, lookup for ICMP echo requests, and, if any,
 * send back ICMP echo replies.
 */
void reply_to_icmp_echo_rqsts(void)
{
	struct rte_mbuf *pkts_burst[MAX_PKT_BURST];
	struct rte_mbuf *pkt;
	struct rte_ether_hdr *eth_h;
	struct rte_vlan_hdr *vlan_h;
	struct rte_arp_hdr  *arp_h;
	struct rte_ipv4_hdr *ip_h;
	struct rte_icmp_hdr *icmp_h;
	struct rte_ether_addr eth_addr;
	uint32_t retry;
	uint32_t ip_addr;
	uint16_t nb_rx;
	uint16_t nb_tx;
	uint16_t nb_replies;
	uint16_t eth_type;
	uint16_t vlan_id;
	uint16_t arp_op;
	uint16_t arp_pro;
	uint32_t cksum;
	uint8_t  i;
	int l2_len;
#ifdef RTE_TEST_PMD_RECORD_CORE_CYCLES
	uint64_t start_tsc;
	uint64_t end_tsc;
	uint64_t core_cycles;
#endif

#ifdef RTE_TEST_PMD_RECORD_CORE_CYCLES
	start_tsc = rte_rdtsc();
#endif
        uint32_t burst_tx_retry_num = 3;
        int verbose_level = 0;
        uint32_t burst_tx_delay_time = 1;
	/*
	 * First, receive a burst of packets.
	 */
	nb_rx = rte_eth_rx_burst(portid, 0, pkts_burst, MAX_PKT_BURST); 
	if (unlikely(nb_rx == 0))
		return;

	nb_replies = 0;
	for (i = 0; i < nb_rx; i++) {
		if (likely(i < nb_rx - 1))
			rte_prefetch0(rte_pktmbuf_mtod(pkts_burst[i + 1],
						       void *));
		pkt = pkts_burst[i];
		eth_h = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
		eth_type = RTE_BE_TO_CPU_16(eth_h->ether_type);
		l2_len = sizeof(struct rte_ether_hdr);
		//ether_addr_dump("  ETH:  src=", &eth_h->s_addr);
		//ether_addr_dump(" dst=", &eth_h->d_addr);
	
		if (eth_type == RTE_ETHER_TYPE_VLAN) {
			vlan_h = (struct rte_vlan_hdr *)
				((char *)eth_h + sizeof(struct rte_ether_hdr));
			l2_len  += sizeof(struct rte_vlan_hdr);
			eth_type = rte_be_to_cpu_16(vlan_h->eth_proto);
			if (verbose_level > 0) {
				vlan_id = rte_be_to_cpu_16(vlan_h->vlan_tci)
					& 0xFFF;
				printf(" [vlan id=%u]", vlan_id);
			}
		}
		if (verbose_level > 0) {
			printf(" type=0x%04x\n", eth_type);
		}

		/* Reply to ARP requests */
		if (eth_type == RTE_ETHER_TYPE_ARP) {
			arp_h = (struct rte_arp_hdr *) ((char *)eth_h + l2_len);
			arp_op = RTE_BE_TO_CPU_16(arp_h->arp_opcode);
			arp_pro = RTE_BE_TO_CPU_16(arp_h->arp_protocol);
			ip_addr = arp_h->arp_data.arp_tip;
                        if (reverse_ip_addr(ip_addr) == server_ip_addr)
                        {
			    ipv4_addr_dump(" tip=", ip_addr);
			    printf("  ARP:  hrd=%d proto=0x%04x hln=%d "
				       "pln=%d op=%u (%s)\n",
				       RTE_BE_TO_CPU_16(arp_h->arp_hardware),
				       arp_pro, arp_h->arp_hlen,
				       arp_h->arp_plen, arp_op,
				       arp_op_name(arp_op));
		        }
                        else
                        {
			    rte_pktmbuf_free(pkt);
                            continue;
                        }
			if ((RTE_BE_TO_CPU_16(arp_h->arp_hardware) !=
			     RTE_ARP_HRD_ETHER) ||
			    (arp_pro != RTE_ETHER_TYPE_IPV4) ||
			    (arp_h->arp_hlen != 6) ||
			    (arp_h->arp_plen != 4)
			    ) {
				rte_pktmbuf_free(pkt);
				if (verbose_level > 0)
					printf("\n");
				continue;
			}
			if (verbose_level > 0) {
				rte_ether_addr_copy(&arp_h->arp_data.arp_sha,
						&eth_addr);
				ether_addr_dump("        sha=", &eth_addr);
				ip_addr = arp_h->arp_data.arp_sip;
				ipv4_addr_dump(" sip=", ip_addr);
				printf("\n");
				rte_ether_addr_copy(&arp_h->arp_data.arp_tha,
						&eth_addr);
				ether_addr_dump("        tha=", &eth_addr);
				ip_addr = arp_h->arp_data.arp_tip;
				ipv4_addr_dump(" tip=", ip_addr);
				printf("\n");
			}
			if (arp_op != RTE_ARP_OP_REQUEST) {
				rte_pktmbuf_free(pkt);
				continue;
			}

			/*
			 * Build ARP reply.
			 */

			/* Use source MAC address as destination MAC address. */
			rte_ether_addr_copy(&eth_h->s_addr, &eth_h->d_addr);
			/* Set source MAC address with MAC address of TX port */
			rte_ether_addr_copy(&server_ether_addr,
					&eth_h->s_addr);

			arp_h->arp_opcode = rte_cpu_to_be_16(RTE_ARP_OP_REPLY);
			rte_ether_addr_copy(&arp_h->arp_data.arp_tha,
					&eth_addr);
			rte_ether_addr_copy(&arp_h->arp_data.arp_sha,
					&arp_h->arp_data.arp_tha);
                        // eth_h->s_addr is the addr of server_ether_addr
			rte_ether_addr_copy(&eth_h->s_addr,
					&arp_h->arp_data.arp_sha);

			/* Swap IP addresses in ARP payload */
			ip_addr = arp_h->arp_data.arp_sip;
			arp_h->arp_data.arp_sip = arp_h->arp_data.arp_tip;
			arp_h->arp_data.arp_tip = ip_addr;
			pkts_burst[nb_replies++] = pkt;
			continue;
		}

		if (eth_type != RTE_ETHER_TYPE_IPV4) {
			rte_pktmbuf_free(pkt);
			continue;
		}
		ip_h = (struct rte_ipv4_hdr *) ((char *)eth_h + l2_len);
		if (verbose_level > 0) {
			ipv4_addr_dump("  IPV4: src=", ip_h->src_addr);
			ipv4_addr_dump(" dst=", ip_h->dst_addr);
			printf(" proto=%d (%s)\n",
			       ip_h->next_proto_id,
			       ip_proto_name(ip_h->next_proto_id));
		}

		/*
		 * Check if packet is a ICMP echo request.
		 */
		icmp_h = (struct rte_icmp_hdr *) ((char *)ip_h +
					      sizeof(struct rte_ipv4_hdr));
		if (! ((ip_h->next_proto_id == IPPROTO_ICMP) &&
		       (icmp_h->icmp_type == RTE_IP_ICMP_ECHO_REQUEST) &&
		       (icmp_h->icmp_code == 0))) {
			rte_pktmbuf_free(pkt);
			continue;
		}

		if (verbose_level > 0)
			printf("  ICMP: echo request seq id=%d\n",
			       rte_be_to_cpu_16(icmp_h->icmp_seq_nb));

		/*
		 * Prepare ICMP echo reply to be sent back.
		 * - switch ethernet source and destinations addresses,
		 * - use the request IP source address as the reply IP
		 *    destination address,
		 * - if the request IP destination address is a multicast
		 *   address:
		 *     - choose a reply IP source address different from the
		 *       request IP source address,
		 *     - re-compute the IP header checksum.
		 *   Otherwise:
		 *     - switch the request IP source and destination
		 *       addresses in the reply IP header,
		 *     - keep the IP header checksum unchanged.
		 * - set RTE_IP_ICMP_ECHO_REPLY in ICMP header.
		 * ICMP checksum is computed by assuming it is valid in the
		 * echo request and not verified.
		 */
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
                        /*
                        if (reverse_ip_addr(ip_h->dst_addr) != server_ip_addr)
                        {
			    rte_pktmbuf_free(pkt);
                        }
                        */
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
		pkts_burst[nb_replies++] = pkt;
	}

	/* Send back ICMP echo replies, if any. */
	if (nb_replies > 0) {
		nb_tx = rte_eth_tx_burst(portid, 0, pkts_burst,
					 nb_replies);
		/*
		 * Retry if necessary
		 */
		if (unlikely(nb_tx < nb_replies)) {
			retry = 0;
			while (nb_tx < nb_replies &&
					retry++ < burst_tx_retry_num) {
				rte_delay_us(burst_tx_delay_time);
				nb_tx += rte_eth_tx_burst(portid,
						0,
						&pkts_burst[nb_tx],
						nb_replies - nb_tx);
			}
		}
		if (unlikely(nb_tx < nb_replies)) {
			do {
				rte_pktmbuf_free(pkts_burst[nb_tx]);
			} while (++nb_tx < nb_replies);
		}
	}

#ifdef RTE_TEST_PMD_RECORD_CORE_CYCLES
	end_tsc = rte_rdtsc();
	core_cycles = (end_tsc - start_tsc);
#endif
}


void server_loop(void)
{
     /* wait for pong */
    while (!force_quit)
    {
        reply_to_icmp_echo_rqsts( );
    }
}

static int
pong_launch_one_lcore(__attribute__((unused)) void *dummy)
{
    server_loop();
    return 0;
}

int main(int argc, char **argv)
{
    int ret;
    uint16_t nb_ports;
    unsigned int nb_mbufs;
    unsigned int nb_lcores;
    unsigned int lcore_id;

    /* init EAL */
    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");
    argc -= ret;
    argv += ret;

#if 0
    /* init log */
    RTE_LOGTYPE_PINGPONG = rte_log_register(APP);
#endif
    rte_set_log_level(PINGPONG_LOG_LEVEL);
    
    nb_lcores = rte_lcore_count();
    if (nb_lcores < 2)
        rte_exit(EXIT_FAILURE, "Number of CPU cores should be no less than 2.");

    nb_ports = rte_eth_dev_count();
    if (nb_ports == 0)
        rte_exit(EXIT_FAILURE, "No Ethernet ports, bye...\n");

    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "%u port(s) available\n", nb_ports);

    /* parse application arguments (after the EAL ones) */
    ret = pingpong_parse_args(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Invalid pingpong arguments\n");
    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "Enabled port: %u\n", portid);
    if (portid > nb_ports - 1)
        rte_exit(EXIT_FAILURE, "Invalid port id %u, port id should be in range [0, %u]\n", portid, nb_ports - 1);

    force_quit = false;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    nb_mbufs = RTE_MAX((unsigned int)(nb_ports * (nb_rxd + nb_txd + MAX_PKT_BURST + MEMPOOL_CACHE_SIZE)), 8192U);
    pingpong_pktmbuf_pool = rte_pktmbuf_pool_create("mbuf_pool", nb_mbufs,
                                                    MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
                                                    rte_socket_id());
    if (pingpong_pktmbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");

    struct rte_eth_rxconf rxq_conf;
    struct rte_eth_txconf txq_conf;
    struct rte_eth_conf local_port_conf = port_conf;
    struct rte_eth_dev_info dev_info;

    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "Initializing port %u...\n", portid);
    fflush(stdout);

    rte_eth_macaddr_get(portid,(struct ether_addr *) &server_ether_addr);
    /* init port */
    rte_eth_dev_info_get(portid, &dev_info);
    //if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
    //    local_port_conf.txmode.offloads |=
    //        DEV_TX_OFFLOAD_MBUF_FAST_FREE;

    ret = rte_eth_dev_configure(portid, 1, 1, &local_port_conf);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n",
                 ret, portid);

#if 0
    ret = rte_eth_dev_adjust_nb_rx_tx_desc(portid, &nb_rxd,
                                           &nb_txd);
#endif
    if (ret < 0)
        rte_exit(EXIT_FAILURE,
                 "Cannot adjust number of descriptors: err=%d, port=%u\n",
                 ret, portid);

    /* init one RX queue */
    fflush(stdout);
    rxq_conf = dev_info.default_rxconf;

    //rxq_conf.offloads = local_port_conf.rxmode.offloads;
    ret = rte_eth_rx_queue_setup(portid, 0, nb_rxd,
                                 rte_eth_dev_socket_id(portid),
                                 &rxq_conf,
                                 pingpong_pktmbuf_pool);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n",
                 ret, portid);

    /* init one TX queue on each port */
    fflush(stdout);
    txq_conf = dev_info.default_txconf;
    //txq_conf.offloads = local_port_conf.txmode.offloads;
    ret = rte_eth_tx_queue_setup(portid, 0, nb_txd,
                                 rte_eth_dev_socket_id(portid),
                                 &txq_conf);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup:err=%d, port=%u\n",
                 ret, portid);

    /* Initialize TX buffers */
    tx_buffer = rte_zmalloc_socket("tx_buffer",
                                   RTE_ETH_TX_BUFFER_SIZE(MAX_PKT_BURST), 0,
                                   rte_eth_dev_socket_id(portid));
    if (tx_buffer == NULL)
        rte_exit(EXIT_FAILURE, "Cannot allocate buffer for tx on port %u\n",
                 portid);

#if 0
    rte_eth_tx_buffer_init(tx_buffer, MAX_PKT_BURST);

    ret = rte_eth_tx_buffer_set_err_callback(tx_buffer,
                                             rte_eth_tx_buffer_count_callback,
                                             &port_statistics.dropped);
    if (ret < 0)
        rte_exit(EXIT_FAILURE,
                 "Cannot set error callback for tx buffer on port %u\n",
                 portid);
#endif
    /* Start device */
    ret = rte_eth_dev_start(portid);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n",
                 ret, portid);

    /* initialize port stats */
    initlize_port_statistics();

    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "Initilize port %u done.\n", portid);

    lcore_id = rte_get_next_lcore(0, true, false);

    ret = 0;
    
    rte_eal_remote_launch(pong_launch_one_lcore, NULL, lcore_id);
   
    

    if (rte_eal_wait_lcore(lcore_id) < 0)
    {
        ret = -1;
    }

    rte_eth_dev_stop(portid);
    rte_eth_dev_close(portid);
    destroy_port_statistics();
    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "Bye.\n");

    return 0;
}
