#ifndef __UTIL_H__
#define __UTIL_H__

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <string.h>

#include <rte_ip.h>
#include <rte_eal.h>
#include <rte_log.h>
#include <rte_tcp.h>
#include <rte_flow.h>
#include <rte_mbuf.h>
#include <rte_ring.h>
#include <rte_ether.h>
#include <rte_atomic.h>
#include <rte_ethdev.h>
#include <rte_malloc.h>
#include <rte_cfgfile.h>
#include <rte_mempool.h>
#include <time.h>

// Constants
#define EPSILON						0.00001
#define MAXSTRLEN					128
#define UNIFORM_VALUE				0
#define EXPONENTIAL_VALUE			1
#define IPV4_ADDR(a, b, c, d)		(((d & 0xff) << 24) | ((c & 0xff) << 16) | ((b & 0xff) << 8) | (a & 0xff))
#define BILLION 					1000000000
#define THROUGHPUT_INTERVAL			5 // ms
typedef struct lcore_parameters {
	uint8_t qid;
	uint16_t portid;
	uint64_t nr_elements;
} __rte_cache_aligned lcore_param;

typedef struct timestamp_node_t {
	uint64_t flow_id;
	uint64_t thread_id;
	uint64_t ack_dup;
	uint64_t ack_empty;
	uint64_t timestamp_rx;
	uint64_t timestamp_tx;
	uint64_t nr_never_sent;
} node_t;

extern uint64_t rate;
extern uint16_t portid;
extern uint64_t duration;
extern uint64_t nr_flows;
extern uint64_t nr_queues;
extern uint16_t nr_servers;
extern uint32_t frame_size;
extern uint32_t min_lcores;
extern uint64_t nr_executions;
extern uint32_t tcp_payload_size;

extern uint64_t TICKS_PER_US;
extern uint16_t **flow_indexes_array;
extern uint64_t **interarrival_array;
extern uint64_t *throughputs;

extern uint16_t dst_tcp_port;
extern uint32_t dst_ipv4_addr;
extern uint32_t src_ipv4_addr;
extern struct rte_ether_addr dst_eth_addr;
extern struct rte_ether_addr src_eth_addr;

extern volatile uint8_t quit_rx;
extern volatile uint8_t quit_tx;
extern volatile uint8_t quit_rx_ring;

extern node_t **incoming_array;
extern uint64_t *incoming_idx_array;

void clean_heap();
void wait_timeout();
void print_dpdk_stats();
void print_stats_output();
void process_config_file();
double sample(double lambda);
void allocate_incoming_nodes();
void create_interarrival_array();
void create_flow_indexes_array();
int app_parse_args(int argc, char **argv);
void fill_payload_pkt(struct rte_mbuf *pkt, uint32_t idx, uint64_t value);
uint64_t get_time_ns();
void prepare_throughput_tracking();

#endif // __UTIL_H__
