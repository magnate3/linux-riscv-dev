/* Simple IPv6 TCP SYN packets generator.
 *
 * Author: Jirka Setnicka <setnicka@seznam.cz>
 *
 * Reused parts from DPDK example basic forwarder (examples/skeleton/basicfwd.c)
 */

#include <stdint.h>
#include <signal.h>
#include <stdbool.h>
#include <getopt.h>

#include <arpa/inet.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_random.h>

#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32
#define IP_DEFTTL  64   /* from RFC 1340. */

static const struct rte_eth_conf port_conf_default = {
	.rxmode = {.max_rx_pkt_len = RTE_ETHER_MAX_LEN},
};

static volatile bool keepRunning = true;

static uint16_t dst_port = 80;
struct in6_addr dst_address;


////////////////////////////////////////////////////////////////////////////////
uint16_t send_packets(struct rte_mempool *mp, uint16_t port_id);
void intHandler(int __attribute((unused))dummy);

////////////////////////////////////////////////////////////////////////////////

uint16_t send_packets(struct rte_mempool *mp, uint16_t port_id) {
	struct rte_mbuf *buf[BURST_SIZE];
        uint8_t i;
        uint16_t j;
	for (i = 0; i < BURST_SIZE; i++) {
		buf[i] = rte_pktmbuf_alloc(mp);

		// 1. Append L2 header and set it
		struct rte_ether_hdr *eth_hdr = (struct rte_ether_hdr*)rte_pktmbuf_append(buf[i], sizeof(struct rte_ether_hdr));
		eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV6);
		// Use local MAC address ???
		// rte_eth_macaddr_get(port_id, &eth_hdr->s_addr);

		// 2. Append L3 IPv6 header and set it
		struct rte_ipv6_hdr *ip_hdr = (struct rte_ipv6_hdr*)rte_pktmbuf_append(buf[i], sizeof(struct rte_ipv6_hdr));
		// Given dst_addr
		rte_memcpy(&ip_hdr->dst_addr, &dst_address.__in6_u.__u6_addr8, 16);
		// Random src_addr
		uint64_t l = rte_rand();
		uint64_t u = rte_rand();
		rte_memcpy(&ip_hdr->src_addr, &l, 8);
		rte_memcpy(&ip_hdr->src_addr[8], &u, 8);

		ip_hdr->vtc_flow = rte_cpu_to_be_32(6 << 28);
		ip_hdr->hop_limits = IP_DEFTTL;
		ip_hdr->proto = IPPROTO_TCP;
		ip_hdr->payload_len = rte_cpu_to_be_16(sizeof(struct rte_tcp_hdr));

		// 3. Append L4 TCP SYN header and set it
		struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr*)rte_pktmbuf_append(buf[i], sizeof(struct rte_tcp_hdr));
		tcp_hdr->src_port = rte_rand() & 0xffff; // random source port
		tcp_hdr->dst_port = rte_cpu_to_be_16(dst_port);
		tcp_hdr->tcp_flags = 1 << 1; // SYN
		tcp_hdr->cksum = 0;

		// 4. Compute checksum:
		tcp_hdr->cksum = rte_ipv6_udptcp_cksum(ip_hdr, tcp_hdr);
	}

	// Send it all in burst
	uint16_t nb_tx = rte_eth_tx_burst(port_id, 0, buf, BURST_SIZE);

	// Free any unsent packets
	if (unlikely(nb_tx < BURST_SIZE)) {
		for (j = nb_tx; j < BURST_SIZE; j++)
		rte_pktmbuf_free(buf[j]);
	}

	return nb_tx;
}

/*
 * Initializes a given port using global settings and with the TX buffers
 * coming from the mbuf_pool passed as a parameter.
 *
 * Source: part of DPDK examples/skeleton/basicfwd.c
 */
static inline int
port_init(uint16_t port)
{
	struct rte_eth_conf port_conf = port_conf_default;
	const uint16_t rx_rings = 0, tx_rings = 1;
	uint16_t nb_txd = TX_RING_SIZE;
	int retval;
	uint16_t q;
	struct rte_eth_dev_info dev_info;
	struct rte_eth_txconf txconf;

	if (!rte_eth_dev_is_valid_port(port))
		return -1;

	rte_eth_dev_info_get(port, &dev_info);
	if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
		port_conf.txmode.offloads |=
			DEV_TX_OFFLOAD_MBUF_FAST_FREE;

	/* Configure the Ethernet device. */
	retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
	if (retval != 0)
		return retval;

	txconf = dev_info.default_txconf;
	txconf.offloads = port_conf.txmode.offloads;
	/* Allocate and set up 1 TX queue per Ethernet port. */
	for (q = 0; q < tx_rings; q++) {
		retval = rte_eth_tx_queue_setup(port, q, nb_txd,
				rte_eth_dev_socket_id(port), &txconf);
		if (retval < 0)
			return retval;
	}

	/* Start the Ethernet port. */
	retval = rte_eth_dev_start(port);
	if (retval < 0)
		return retval;

	/* Display the port MAC address. */
	struct rte_ether_addr addr;
	rte_eth_macaddr_get(port, &addr);
	printf("Port %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
			   " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",
			port,
			addr.addr_bytes[0], addr.addr_bytes[1],
			addr.addr_bytes[2], addr.addr_bytes[3],
			addr.addr_bytes[4], addr.addr_bytes[5]);

	return 0;
}

// SIGINT handler
void intHandler(int __attribute((unused))dummy) { keepRunning = false; }

int
main(int argc, char *argv[])
{
	struct rte_mempool *mbuf_pool;
	int nb_ports = 1;
	uint16_t portid;
	uint32_t gen_count = 0;
	uint32_t tx_count = 0;

	signal(SIGINT, intHandler);


	// 1. Init Environment Abstraction Layer (EAL)
	int ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

	// 2. Parse rest of rags (and skip initial -- if there)
	argc -= ret;
	argv += ret;
	int opt;
	while ((opt = getopt(argc, argv, "t:p:")) != -1) {
		switch (opt) {
		case 't':
			printf("Parsing address %s\n", optarg);
			if (inet_pton(AF_INET6, optarg, &dst_address) != 1) {
				fprintf(stderr, "IPv6 address in wrong format\n");
				exit(EXIT_FAILURE);
			}
			break;
		case 'p':
			dst_port = atoi(optarg);
			break;
		default: /* '?' */
			fprintf(stderr, "Usage: [-t target_address] [-p target_port]\n");
			exit(EXIT_FAILURE);
		}
	}

	// 3. Rest of initialization
	/* Creates a new mempool in memory to hold the mbufs. */
	mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports,
		MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
	if (mbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

	/* Initialize all ports. */
	RTE_ETH_FOREACH_DEV(portid)
		if (port_init(portid) != 0)
			rte_exit(EXIT_FAILURE, "Cannot init port %"PRIu16 "\n",
					portid);


	// 4. Main loop
	while (keepRunning) {
		RTE_ETH_FOREACH_DEV(portid) {
			gen_count += BURST_SIZE;
			tx_count += send_packets(mbuf_pool, portid);
		}
		if (gen_count % (BURST_SIZE << 16) == 0) printf("Gen: %d\tTx: %d\r", gen_count, tx_count);
	}

	printf("\rGen: %d\tTx: %d\n", gen_count, tx_count);
	return 0;
}
