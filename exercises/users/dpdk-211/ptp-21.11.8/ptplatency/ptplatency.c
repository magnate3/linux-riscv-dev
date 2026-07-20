/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2021 Red Hat
 */

/*
 * This application is a simple Layer 2 PTP v2 latency measurement tool
 */

#include <stdint.h>
#include <inttypes.h>
#include <rte_version.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_ip.h>
#include <limits.h>
#include <sys/time.h>
#include <getopt.h>
#include <signal.h>

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024

#define NUM_MBUFS            8191
#define MBUF_CACHE_SIZE       250

/* Values for the PTP messageType field. */
#define SYNC                  0x0
#define DELAY_REQ             0x1
#define PDELAY_REQ            0x2
#define PDELAY_RESP           0x3
#define FOLLOW_UP             0x8
#define DELAY_RESP            0x9
#define PDELAY_RESP_FOLLOW_UP 0xA
#define ANNOUNCE              0xB
#define SIGNALING             0xC
#define MANAGEMENT            0xD

#define NSEC_PER_SEC        1000000000L
#define KERNEL_TIME_ADJUST_LIMIT  20000
#define PTP_PROTOCOL             0x88F7

rte_atomic32_t send_ptp_frame_counter = {0};

struct rte_mempool *mbuf_pool;
uint32_t ptp_enabled_port_mask;
uint8_t ptp_enabled_port_nb;
static uint8_t ptp_enabled_ports[RTE_MAX_ETHPORTS];

static const struct rte_eth_conf port_conf_default = {
	//.rxmode = {
	//	.max_rx_pkt_len = RTE_ETHER_MAX_LEN,
	//},
#if RTE_VERSION >= RTE_VERSION_NUM(21, 11, 0, 0)
		.rxmode = {.mtu = RTE_ETHER_MAX_LEN,},
#else
		.rxmode = {.max_rx_pkt_len = RTE_ETHER_MAX_LEN,},
#endif
};

static struct rte_ether_addr ether_dest = {
	/* Defaults to PTP multicast addr */
	.addr_bytes = {0x01, 0x1b, 0x19, 0x0, 0x0, 0x0}
};

/* Structs used for PTP handling. */
struct tstamp {
	uint16_t   sec_msb;
	uint32_t   sec_lsb;
	uint32_t   ns;
}  __rte_packed;

struct clock_id {
	uint8_t id[8];
};

struct port_id {
	struct clock_id        clock_id;
	uint16_t               port_number;
}  __rte_packed;

struct ptp_header {
	uint8_t              msg_type;
	uint8_t              ver;
	uint16_t             message_length;
	uint8_t              domain_number;
	uint8_t              reserved1;
	uint8_t              flag_field[2];
	int64_t              correction;
	uint32_t             reserved2;
	struct port_id       source_port_id;
	uint16_t             seq_id;
	uint8_t              control;
	int8_t               log_message_interval;
} __rte_packed;

struct sync_msg {
	struct ptp_header   hdr;
	struct tstamp       origin_tstamp;
} __rte_packed;

struct follow_up_msg {
	struct ptp_header   hdr;
	struct tstamp       precise_origin_tstamp;
	uint8_t             suffix[0];
} __rte_packed;

struct delay_req_msg {
	struct ptp_header   hdr;
	struct tstamp       origin_tstamp;
} __rte_packed;

struct delay_resp_msg {
	struct ptp_header    hdr;
	struct tstamp        rx_tstamp;
	struct port_id       req_port_id;
	uint8_t              suffix[0];
} __rte_packed;

struct ptp_message {
	union {
		struct ptp_header          header;
		struct sync_msg            sync;
		struct delay_req_msg       delay_req;
		struct follow_up_msg       follow_up;
		struct delay_resp_msg      delay_resp;
	} __rte_packed;
};

#define DELTA_SAMPLES 20
struct ptpv2_data_slave_ordinary {
	struct rte_mbuf *m;
	struct timespec ts_rx_sync;
	struct timespec ts_fup;
	struct clock_id client_clock_id;
	struct clock_id master_clock_id;
	struct timeval new_adj;
	int64_t delta;
	int64_t delta_max;
	int64_t delta_acc[DELTA_SAMPLES];
	uint16_t portid;
	uint16_t seqID_SYNC;
	uint16_t seqID_FOLLOWUP;
	uint8_t ptpset;
	uint8_t kernel_time_set;
	uint16_t current_ptp_port;
	uint16_t ptp_frame_size;
};

static struct ptpv2_data_slave_ordinary ptp_data;

static inline uint64_t timespec64_to_ns(const struct timespec *ts)
{
	return ((uint64_t) ts->tv_sec * NSEC_PER_SEC) + ts->tv_nsec;
}

static struct timeval
ns_to_timeval(int64_t nsec)
{
	struct timespec t_spec = {0, 0};
	struct timeval t_eval = {0, 0};
	int32_t rem;

	if (nsec == 0)
		return t_eval;
	rem = nsec % NSEC_PER_SEC;
	t_spec.tv_sec = nsec / NSEC_PER_SEC;

	if (rem < 0) {
		t_spec.tv_sec--;
		rem += NSEC_PER_SEC;
	}

	t_spec.tv_nsec = rem;
	t_eval.tv_sec = t_spec.tv_sec;
	t_eval.tv_usec = t_spec.tv_nsec / 1000;

	return t_eval;
}

/*
static void
port_ieee1588_rx_timestamp_check(uint16_t pi, uint32_t index)
{
	struct timespec timestamp = {0, 0};

	if (rte_eth_timesync_read_rx_timestamp(pi, &timestamp, index) < 0) {
		printf("Port %u RX timestamp registers not valid\n", pi);
		return;
	}
	printf("Port %u RX timestamp value %lu s %lu ns\n",
		pi, timestamp.tv_sec, timestamp.tv_nsec);
}
*/

#define MAX_TX_TMST_WAIT_MICROSECS 1000 /**< 1 milli-second */

static void
port_ieee1588_tx_timestamp_check(uint16_t pi, struct timespec *timestamp)
{
	unsigned wait_us = 0;

	while ((rte_eth_timesync_read_tx_timestamp(pi, timestamp) < 0) &&
	       (wait_us < MAX_TX_TMST_WAIT_MICROSECS)) {
		rte_delay_us(1);
		wait_us++;
	}
	if (wait_us >= MAX_TX_TMST_WAIT_MICROSECS) {
		printf("Port %u TX timestamp registers not valid after "
		       "%u micro-seconds\n",
		       pi, MAX_TX_TMST_WAIT_MICROSECS);
		return;
	}
	/*
	printf("Port %u TX timestamp value %lu s %lu ns validated after "
	       "%u micro-second%s\n",
	       pi, timestamp->tv_sec, timestamp->tv_nsec, wait_us,
	       (wait_us == 1) ? "" : "s");
		   */
}


/*
 * Initializes a given port using global settings and with the RX buffers
 * coming from the mbuf_pool passed as a parameter.
 */
static inline int
port_init(uint16_t port, struct rte_mempool *mbuf_pool)
{
	struct rte_eth_dev_info dev_info;
	struct rte_eth_conf port_conf = port_conf_default;
	const uint16_t rx_rings = 1;
	const uint16_t tx_rings = 1;
	int retval;
	uint16_t q;
	uint16_t nb_rxd = RX_RING_SIZE;
	uint16_t nb_txd = TX_RING_SIZE;

	if (!rte_eth_dev_is_valid_port(port))
		return -1;

	retval = rte_eth_dev_info_get(port, &dev_info);
	if (retval != 0) {
		printf("Error during getting device (port %u) info: %s\n",
				port, strerror(-retval));

		return retval;
	}

	if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
		port_conf.txmode.offloads |=
			DEV_TX_OFFLOAD_MBUF_FAST_FREE;
	/* Force full Tx path in the driver, required for IEEE1588 */
	port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MULTI_SEGS;

	/* Configure the Ethernet device. */
	retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
	if (retval != 0)
		return retval;

	retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
	if (retval != 0)
		return retval;

	/* Allocate and set up 1 RX queue per Ethernet port. */
	for (q = 0; q < rx_rings; q++) {
		retval = rte_eth_rx_queue_setup(port, q, nb_rxd,
				rte_eth_dev_socket_id(port), NULL, mbuf_pool);

		if (retval < 0)
			return retval;
	}

	/* Allocate and set up 1 TX queue per Ethernet port. */
	for (q = 0; q < tx_rings; q++) {
		struct rte_eth_txconf *txconf;

		txconf = &dev_info.default_txconf;
		txconf->offloads = port_conf.txmode.offloads;

		retval = rte_eth_tx_queue_setup(port, q, nb_txd,
				rte_eth_dev_socket_id(port), txconf);
		if (retval < 0)
			return retval;
	}

	/* Start the Ethernet port. */
	retval = rte_eth_dev_start(port);
	if (retval < 0)
		return retval;

	/* Enable timesync timestamping for the Ethernet device */
	retval = rte_eth_timesync_enable(port);
	if (retval < 0) {
		printf("Timesync enable failed: %d\n", retval);
		return retval;
	}

	/* Enable RX in promiscuous mode for the Ethernet device. */
	retval = rte_eth_promiscuous_enable(port);
	if (retval != 0) {
		printf("Promiscuous mode enable failed: %s\n",
			rte_strerror(-retval));
		return retval;
	}

	return 0;
}

static void
print_sync_delay(struct ptpv2_data_slave_ordinary *ptp_data) {
	int64_t nsec;
	struct timespec net_time, sys_time;
	int64_t delta, delta_avg = 0;
	uint64_t t1 = 0;
	uint64_t t2 = 0;

	for (uint16_t portid = 1; portid < ptp_enabled_port_nb; portid++) {
		struct timespec time_p0, time_p1;
		int64_t p0p1_diff = 0;
		rte_eth_timesync_read_time(0, &time_p0);
		rte_eth_timesync_read_time(portid, &time_p1);
	
		p0p1_diff = timespec64_to_ns( &time_p0) - timespec64_to_ns(&time_p1);
		printf("\nTime P0 P%d Diff: %ldns", portid, p0p1_diff);
	}
	for (uint16_t portid = 1; portid < ptp_enabled_port_nb; portid++) {
		uint64_t t0, t1;
		rte_eth_read_clock(0, &t0);
		rte_eth_read_clock(portid, &t1);

		printf("\nClock P0 P%d Diff: %ld", portid, t0-t1);
	}

	t1 = timespec64_to_ns(&ptp_data->ts_rx_sync);
	t2 = timespec64_to_ns(&ptp_data->ts_fup);
	delta = t1 - t2;
	
	printf("\nSYNC Seq ID: %d \t FUP Seq ID: %d", ptp_data->seqID_SYNC, ptp_data->seqID_FOLLOWUP);
	printf("\nT1 - Sync RX ts.  %lds %ldns",
			(ptp_data->ts_rx_sync.tv_sec),
			(ptp_data->ts_rx_sync.tv_nsec));

	printf("\nT2 - FUP  RX ts.  %lds %ldns",
			(ptp_data->ts_fup.tv_sec),
			(ptp_data->ts_fup.tv_nsec));

	
	printf("\nRound trip delay: %ldns", delta);
	ptp_data->delta_acc[ptp_data->seqID_SYNC%DELTA_SAMPLES] = delta>0?delta:-delta;
	for (int i = 0 ; i < DELTA_SAMPLES ; i++) {
		delta_avg += ptp_data->delta_acc[i];
	}

	delta_avg = (int64_t) delta_avg / DELTA_SAMPLES;
	printf("\t AVG (last %d samples): %ldns", DELTA_SAMPLES, delta_avg);

	delta = delta > 0 ? delta : -delta;
	ptp_data->delta_max = delta > ptp_data->delta_max ? delta : ptp_data->delta_max;
	printf("\nMax round trip delay: %ldns",  ptp_data->delta_max);
	
	clock_gettime(CLOCK_REALTIME, &sys_time);
	rte_eth_timesync_read_time(ptp_data->current_ptp_port, &net_time);

	time_t ts = net_time.tv_sec;

	printf("\n\nComparison between Linux kernel Time and PTP:");

	printf("\nCurrent PTP Time: %.24s %.9ld ns",
			ctime(&ts), net_time.tv_nsec);

	nsec = (int64_t)timespec64_to_ns(&net_time) -
			(int64_t)timespec64_to_ns(&sys_time);
	ptp_data->new_adj = ns_to_timeval(nsec);

	gettimeofday(&ptp_data->new_adj, NULL);

	time_t tp = ptp_data->new_adj.tv_sec;

	printf("\nCurrent SYS Time: %.24s %.6ld us",
				ctime(&tp), ptp_data->new_adj.tv_usec);

	printf("\nDelta between PTP and Linux Kernel time:   %.9"PRId64" ns\n",
				nsec);

	
	printf("[Ctrl+C to quit]\n");
	/* Clear screen and put cursor in column 1, row 1 */
	printf("\033[2J\033[1;1H");

}

/*
 * Parse the PTP SYNC message.
 */
static void
parse_sync(struct ptpv2_data_slave_ordinary *ptp_data, uint16_t rx_tstamp_idx)
{
	struct ptp_header *ptp_hdr;

	ptp_hdr = (struct ptp_header *)(rte_pktmbuf_mtod(ptp_data->m, char *)
			+ sizeof(struct rte_ether_hdr));
	ptp_data->seqID_SYNC = rte_be_to_cpu_16(ptp_hdr->seq_id);

	if (ptp_data->ptpset == 0) {
		rte_memcpy(&ptp_data->master_clock_id,
				&ptp_hdr->source_port_id.clock_id,
				sizeof(struct clock_id));
		ptp_data->ptpset = 1;
	}

	if (memcmp(&ptp_data->master_clock_id, &ptp_hdr->source_port_id.clock_id,
		   sizeof(struct clock_id)) == 0) {
		if (ptp_data->ptpset == 1) {
			rte_eth_timesync_read_rx_timestamp(ptp_data->portid,
					&ptp_data->ts_rx_sync, rx_tstamp_idx);
		}
	}

}

/*
 * Parse the PTP FOLLOWUP message and copy origin precision time
 */
static void
parse_fup(struct ptpv2_data_slave_ordinary *ptp_data)
{
	struct rte_ether_addr eth_addr;
	struct ptp_header *ptp_hdr;
	struct ptp_message *ptp_msg;
	struct tstamp *origin_tstamp;
	struct rte_mbuf *m = ptp_data->m;
	int ret;

	ptp_hdr = (struct ptp_header *)(rte_pktmbuf_mtod(m, char *)
			+ sizeof(struct rte_ether_hdr));
	if (memcmp(&ptp_data->master_clock_id,
			&ptp_hdr->source_port_id.clock_id,
			sizeof(struct clock_id)) != 0)
		return;

	ptp_data->seqID_FOLLOWUP = rte_be_to_cpu_16(ptp_hdr->seq_id);
	ptp_msg = (struct ptp_message *) (rte_pktmbuf_mtod(m, char *) +
					  sizeof(struct rte_ether_hdr));

	origin_tstamp = &ptp_msg->follow_up.precise_origin_tstamp;
	ptp_data->ts_fup.tv_nsec = ntohl(origin_tstamp->ns);
	ptp_data->ts_fup.tv_sec =
		((uint64_t)ntohl(origin_tstamp->sec_lsb)) |
		(((uint64_t)ntohs(origin_tstamp->sec_msb)) << 32);

	if (ptp_data->seqID_FOLLOWUP == ptp_data->seqID_SYNC) {
		ret = rte_eth_macaddr_get(ptp_data->portid, &eth_addr);
		if (ret != 0) {
			printf("\nCore %u: port %u failed to get MAC address: %s\n",
				rte_lcore_id(), ptp_data->portid,
				rte_strerror(-ret));
			return;
		}
	}
}

/* This function processes PTP packets, implementing slave PTP IEEE1588 L2
 * functionality.
 */
static void
parse_ptp_frames(uint16_t portid, struct rte_mbuf *m) {
	struct ptp_header *ptp_hdr;
	struct rte_ether_hdr *eth_hdr;
	uint16_t eth_type;

	eth_hdr = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
	eth_type = rte_be_to_cpu_16(eth_hdr->ether_type);

	if (eth_type == PTP_PROTOCOL) {
		ptp_data.m = m;
		ptp_data.portid = portid;
		ptp_hdr = (struct ptp_header *)(rte_pktmbuf_mtod(m, char *)
					+ sizeof(struct rte_ether_hdr));

		switch (ptp_hdr->msg_type) {
		case SYNC:
			parse_sync(&ptp_data, m->timesync);
			break;
		case FOLLOW_UP:
			parse_fup(&ptp_data);
			print_sync_delay(&ptp_data);
			break;
		default:
			break;
		}
	}
}


static struct rte_mbuf *
allocate_ptp_frame(uint16_t portid, int msg_type, int seq_ptp_counter) {
	struct rte_mbuf *created_pkt;
	struct rte_ether_hdr *eth_hdr;
	struct rte_ether_addr eth_addr;
	struct clock_id *client_clkid;
	struct ptp_message *ptp_msg;
	struct rte_ether_addr eth_multicast = ether_dest;
	size_t pkt_size;

	created_pkt = rte_pktmbuf_alloc(mbuf_pool);
	pkt_size = sizeof(struct rte_ether_hdr) + sizeof(struct follow_up_msg);
	created_pkt->data_len = pkt_size;
	created_pkt->pkt_len = pkt_size;
	eth_hdr = rte_pktmbuf_mtod(created_pkt, struct rte_ether_hdr *);
	memset(eth_hdr, 0, pkt_size);
	rte_eth_macaddr_get(portid, &eth_addr);
	rte_ether_addr_copy(&eth_addr, &eth_hdr->src_addr);

	/* Set destination address, defaults to 01-1B-19-00-00-00. */
	rte_ether_addr_copy(&eth_multicast, &eth_hdr->dst_addr);

	eth_hdr->ether_type = htons(PTP_PROTOCOL);

	ptp_msg = (struct ptp_message *)
		(rte_pktmbuf_mtod(created_pkt, char *) +
		sizeof(struct rte_ether_hdr));
	
	ptp_msg->header.msg_type = msg_type;
	ptp_msg->header.ver = 2;
	// ptp_msg->header.message_length = 0;
	ptp_msg->header.domain_number = 0;
	ptp_msg->header.correction = 0;
	ptp_msg->header.source_port_id.port_number = htons(1);
	ptp_msg->header.control = 0;
	ptp_msg->header.seq_id = htons(seq_ptp_counter);
	ptp_msg->header.log_message_interval = 0;

	/* Set up clock id. */
	client_clkid = &ptp_msg->header.source_port_id.clock_id;

	client_clkid->id[0] = eth_hdr->src_addr.addr_bytes[0];
	client_clkid->id[1] = eth_hdr->src_addr.addr_bytes[1];
	client_clkid->id[2] = eth_hdr->src_addr.addr_bytes[2];
	client_clkid->id[3] = 0xFF;
	client_clkid->id[4] = 0xFE;
	client_clkid->id[5] = eth_hdr->src_addr.addr_bytes[3];
	client_clkid->id[6] = eth_hdr->src_addr.addr_bytes[4];
	client_clkid->id[7] = eth_hdr->src_addr.addr_bytes[5];

	switch(ptp_msg->header.msg_type) {
		case SYNC:
			if (ptp_data.ptp_frame_size > sizeof(struct sync_msg))
				ptp_msg->header.message_length = htons(ptp_data.ptp_frame_size);
			else
				ptp_msg->header.message_length = htons(sizeof(struct sync_msg));
			ptp_msg->header.control = 0;
			/* Enable flag for hardware timestamping. */
			created_pkt->ol_flags |= PKT_TX_IEEE1588_TMST;
		    break;
		case DELAY_REQ:
			ptp_msg->header.message_length = htons(sizeof(struct delay_req_msg));
			ptp_msg->header.control = 1;
			break;
		case FOLLOW_UP: 
		case PDELAY_RESP_FOLLOW_UP:
			ptp_msg->header.message_length = htons(sizeof(struct follow_up_msg));
			ptp_msg->header.control = 2;
			break;
		case DELAY_RESP:
			ptp_msg->header.message_length = htons(sizeof(struct delay_resp_msg));
			ptp_msg->header.control = 3;
			/* Enable flag for hardware timestamping. */
			created_pkt->ol_flags |= PKT_TX_IEEE1588_TMST;
			break;
		default:
			ptp_msg->header.message_length = htons(sizeof(struct ptp_message));
			break;
	}

	pkt_size = sizeof(struct rte_ether_hdr) + ntohs(ptp_msg->header.message_length);
	created_pkt->data_len = pkt_size;
	created_pkt->pkt_len = pkt_size;


	return created_pkt;
}

static void resync_clock(uint16_t port_src, uint16_t port_dst) {
	int64_t delta;
	struct timespec src;
	struct timespec dst;
	if (port_src == port_dst)
		return;

	rte_eth_timesync_read_time(port_src, &src);
	rte_eth_timesync_read_time(port_dst, &dst);
	delta = (int64_t)timespec64_to_ns(&src) -
	       (int64_t)timespec64_to_ns(&dst);

	rte_eth_timesync_adjust_time(port_dst, delta);
}

/*
 * The lcore main. This is the main thread that does the work, reading from an
 * input port and writing to an output port.
 */
static __rte_noreturn void
lcore_main(void)
{
	uint16_t portid;
	unsigned nb_rx;
	struct rte_mbuf *m;
	uint32_t seq_ptp_counter = 0;

	/*
	 * Check that the port is on the same NUMA node as the polling thread
	 * for best performance.
	 */
	printf("\nCore %u Waiting for SYNC packets. [Ctrl+C to quit]\n",
			rte_lcore_id());

	/* Run until the application is quit or killed. */
	while (1) {
		/* Read packet from RX queues. */
		for (portid = 0; portid < ptp_enabled_port_nb; portid++) {
			portid = ptp_enabled_ports[portid];
			nb_rx = rte_eth_rx_burst(portid, 0, &m, 1);

			if (likely(nb_rx == 0))
				continue;

			if (m->ol_flags & PKT_RX_IEEE1588_PTP) {
				parse_ptp_frames(portid, m);
			}

			rte_pktmbuf_free(m);
		}

		if ( rte_atomic32_read(&send_ptp_frame_counter)) {
			uint16_t tx_portid = 0;
	//		tx_portid = ptp_data.portid;

			for (portid = 0; portid < ptp_enabled_port_nb; portid++) {
				resync_clock(tx_portid, portid);
			}
			
			struct rte_mbuf *sync_pkt = allocate_ptp_frame(tx_portid, SYNC, seq_ptp_counter);
			struct rte_mbuf *fup_pkt = allocate_ptp_frame(tx_portid, FOLLOW_UP, seq_ptp_counter++);
			struct ptp_message *sync_msg;
			struct follow_up_msg *fup_msg;

			sync_msg = (struct ptp_message *) (rte_pktmbuf_mtod(sync_pkt, char *) +
												sizeof(struct rte_ether_hdr));
			/* PTP TWO STEPS: first SYNC message is followed by a FOLLOW_UP MESSAGE
			 * which contains the origin timestamp of the SYNC message with the same
			 * sequence id.
			 */
			sync_msg->header.flag_field[0] = 0x2; //PTP_TWO_STEPS
			fup_msg = (struct follow_up_msg *) (rte_pktmbuf_mtod(fup_pkt, char *) +
												sizeof(struct rte_ether_hdr));


			do {
				struct timespec tstamp_tx;
				// Read value from NIC to prevent latching with old value.
				rte_eth_timesync_read_tx_timestamp(tx_portid, &tstamp_tx);

				rte_eth_tx_burst(tx_portid, 0, &sync_pkt, 1);
				tstamp_tx.tv_nsec = 0;
				tstamp_tx.tv_sec = 0;

				port_ieee1588_tx_timestamp_check(tx_portid, &tstamp_tx);
				fup_msg->precise_origin_tstamp.ns = htonl(tstamp_tx.tv_nsec);
				fup_msg->precise_origin_tstamp.sec_msb = htons((uint16_t)(tstamp_tx.tv_sec >> 32));
				fup_msg->precise_origin_tstamp.sec_lsb = htonl((uint32_t)(tstamp_tx.tv_sec & UINT32_MAX));

				rte_eth_tx_burst(tx_portid, 0, &fup_pkt, 1);
			} while( !rte_atomic32_dec_and_test(&send_ptp_frame_counter));

		}
	}
}

static void
print_usage(const char *prgname)
{
	printf("%s [EAL options] -- -p PORTMASK -s FRAMESIZE \n"
		" -p PORTMASK: hexadecimal bitmask of ports to configure\n"
		" -s FRAMESIZE: Size of the sync message",
		prgname);
}

static int
ptp_parse_portmask(const char *portmask)
{
	char *end = NULL;
	unsigned long pm;

	/* Parse the hexadecimal string. */
	pm = strtoul(portmask, &end, 16);

	if ((portmask[0] == '\0') || (end == NULL) || (*end != '\0'))
		return 0;

	return pm;
}

static int
parse_ptp_frame_size(const char *param)
{
	char *end = NULL;
	unsigned long pm;

	/* Parse the decimal string. */
	pm = strtoul(param, &end, 10);

	if ((param[0] == '\0') || (end == NULL) || (*end != '\0'))
		return -1;
	return pm;
}
/* Parse the commandline arguments. */
static int
ptp_parse_args(int argc, char **argv)
{
	int opt, ret;
	char **argvopt;
	int option_index;
	char *prgname = argv[0];
	static struct option lgopts[] = { {NULL, 0, 0, 0} };

	argvopt = argv;

	while ((opt = getopt_long(argc, argvopt, "p:T:s:d:",
				  lgopts, &option_index)) != EOF) {

		switch (opt) {
		/* Destination mac address */
		case 'd':
			if (rte_ether_unformat_addr(optarg, &ether_dest)) {
				printf("Invalid ether mac address");
				print_usage(prgname);
				return -1;
			}
			break;
		/* Portmask. */
		case 'p':
			ptp_enabled_port_mask = ptp_parse_portmask(optarg);
			if (ptp_enabled_port_mask == 0) {
				printf("invalid portmask\n");
				print_usage(prgname);
				return -1;
			}
			break;
		case 's':
			ret = parse_ptp_frame_size(optarg);
			if (ret < 0) {
				print_usage(prgname);
				return -1;
			}
			ptp_data.ptp_frame_size = ret;
			break;
		default:
			print_usage(prgname);
			return -1;
		}
	}

	argv[optind-1] = prgname;

	optind = 1; /* Reset getopt lib. */

	return 0;
}

static void send_ptp_frame_sig(int param) {
	if (param == SIGUSR1)
		rte_atomic32_inc(&send_ptp_frame_counter);
}

/*
 * The main function, which does initialization and calls the per-lcore
 * functions.
 */
int
main(int argc, char *argv[])
{
	unsigned nb_ports;

	uint16_t portid;
	struct timespec tp;
	rte_atomic32_init(&send_ptp_frame_counter);

	/* Initialize the Environment Abstraction Layer (EAL). */
	int ret = rte_eal_init(argc, argv);

	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");

	memset(&ptp_data, '\0', sizeof(struct ptpv2_data_slave_ordinary));

	argc -= ret;
	argv += ret;

	ret = ptp_parse_args(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "Error with PTP initialization\n");

	/* Check that there is an even number of ports to send/receive on. */
	nb_ports = rte_eth_dev_count_avail();

	signal(SIGUSR1, send_ptp_frame_sig);

	/* Creates a new mempool in memory to hold the mbufs. */
	mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * nb_ports,
		MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());

	if (mbuf_pool == NULL)
		rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

	/* Initialize all ports. */
	RTE_ETH_FOREACH_DEV(portid) {
		if ((ptp_enabled_port_mask & (1 << portid)) != 0) {
			if (port_init(portid, mbuf_pool) == 0) {
			ptp_enabled_ports[ptp_enabled_port_nb] = portid;
				ptp_enabled_port_nb++;
				clock_gettime(CLOCK_REALTIME, &tp);
				        rte_eth_timesync_write_time(portid, &tp);
			} else {
				rte_exit(EXIT_FAILURE,
					 "Cannot init port %"PRIu8 "\n",
					 portid);
			}
		} else
			printf("Skipping disabled port %u\n", portid);
	}

	if (ptp_enabled_port_nb == 0) {
		rte_exit(EXIT_FAILURE,
			"All available ports are disabled."
			" Please set portmask.\n");
	}

	if (rte_lcore_count() > 1)
		printf("\nWARNING: Too many lcores enabled. Only 1 used.\n");

	/* Call lcore_main on the main core only. */
	lcore_main();

	return 0;
}
