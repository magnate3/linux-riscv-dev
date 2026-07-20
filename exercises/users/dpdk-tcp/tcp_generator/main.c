#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <pcap.h>
#include <net/if.h>

#include "util.h"
#include "tcp_util.h"
#include "dpdk_util.h"
#include "dpdk.h"
#define TCP_BUF_LEN 1024

// Application parameters
uint64_t rate;
uint64_t duration;
uint64_t nr_flows;
uint64_t nr_queues;
uint16_t nr_servers;
uint32_t min_lcores;
uint32_t frame_size;
uint64_t nr_executions;
uint32_t tcp_payload_size;
uint64_t starting_point;

// General variables
uint64_t TICKS_PER_US;
uint16_t **flow_indexes_array;
uint64_t **interarrival_array;
uint64_t *throughputs;

// Heap and DPDK allocated
node_t **incoming_array;
uint64_t *incoming_idx_array;
struct rte_mempool *pktmbuf_pool;
tcp_control_block_t *tcp_control_blocks;

// Internal threads variables
volatile uint8_t quit_rx = 0;
volatile uint8_t quit_tx = 0;
volatile uint32_t ack_dup = 0;
volatile uint32_t ack_empty = 0;
volatile uint8_t quit_rx_ring = 0;
volatile uint64_t nr_never_sent = 0;
lcore_param lcore_params[RTE_MAX_LCORE];
struct rte_ring *rx_rings[RTE_MAX_LCORE];

// Connection variables
uint16_t dst_tcp_port;
uint32_t dst_ipv4_addr;
uint32_t src_ipv4_addr;
struct rte_ether_addr dst_eth_addr;
struct rte_ether_addr src_eth_addr;
static void tcp_reply(const uint16_t flow_id,lcore_param *rx_conf,uint8_t flag);
#if DBUG_TCP
pcap_dumper_t *         dumper ;
#endif
#ifndef RTE_ARM_EAL_RDTSC_USE_PMU
/**
 *  * This call is portable to any ARMv8 architecture, however, typically
 *   * cntvct_el0 runs at <= 100MHz and it may be imprecise for some tasks.
 *    */
static inline uint64_t
rte_rdtsc(void)
{
        uint64_t tsc;

        asm volatile("mrs %0, cntvct_el0" : "=r" (tsc));
        return tsc;
}
#else
/**
 *  * This is an alternative method to enable rte_rdtsc() with high resolution
 *   * PMU cycles counter.The cycle counter runs at cpu frequency and this scheme
 *    * uses ARMv8 PMU subsystem to get the cycle counter at userspace, However,
 *     * access to PMU cycle counter from user space is not enabled by default in
 *      * arm64 linux kernel.
 *       * It is possible to enable cycle counter at user space access by configuring
 *        * the PMU from the privileged mode (kernel space).
 *         *
 *          * asm volatile("msr pmintenset_el1, %0" : : "r" ((u64)(0 << 31)));
 *           * asm volatile("msr pmcntenset_el0, %0" :: "r" BIT(31));
 *            * asm volatile("msr pmuserenr_el0, %0" : : "r"(BIT(0) | BIT(2)));
 *             * asm volatile("mrs %0, pmcr_el0" : "=r" (val));
 *              * val |= (BIT(0) | BIT(2));
 *               * isb();
 *                * asm volatile("msr pmcr_el0, %0" : : "r" (val));
 *                 *
 *                  */
static inline uint64_t
rte_rdtsc(void)
{
        uint64_t tsc;

        asm volatile("mrs %0, pmccntr_el0" : "=r"(tsc));
        return tsc;
}
#endif
#if DBUG_TCP
static void open_pcap(const char *fname)
{
    char name[IFNAMSIZ +16];
    snprintf(name,IFNAMSIZ +16,"%s-txrx.pcap",fname);
    dumper = pcap_dump_open(pcap_open_dead(DLT_EN10MB, 1600), name);
    if (NULL == dumper)
    {
        printf("dumper is NULL\n");
        return;
    }
}

static void dump_pcap(const char *pkt, int len)
{
    struct timeval tv;
    struct pcap_pkthdr hdr;
    gettimeofday(&tv, NULL);
    hdr.ts.tv_sec = tv.tv_sec;
    hdr.ts.tv_usec = tv.tv_usec;
    hdr.caplen = len;
    hdr.len = len; 
    pcap_dump((u_char*)dumper, &hdr, (u_char*)pkt); 
}
static void dump_tcp_buf(struct rte_mbuf *m)
{
    unsigned int nb_segs; 
    nb_segs = m->nb_segs;
    fprintf(stdout, "dump prim mbuf at %p, addr=%p iova=%" PRIx64 ", buf_len=%u\n", (void *)(uintptr_t)m,
            m->buf_addr, (uint64_t)m->buf_iova, (unsigned)m->buf_len);
    fprintf(stdout, "  pkt_len=%" PRIu32 ", data_len=%"  PRIu32 ", nb_segs=%u, in_port=%u, next %p\n", m->pkt_len,(unsigned)m->data_len,
            (unsigned)m->nb_segs, (unsigned)m->port, m->next);
    
     fprintf(stdout,"mbuf l2 len %u, l3 len %u, l4 len %u \n", m->l2_len,  m->l3_len, m->l4_len); 
    m= m->next;
    nb_segs--;
    while (m && nb_segs != 0) {
        fprintf(stdout, " sub segment at %p, data=%p, data_len=%u,next= %p \n", (void *)(uintptr_t)m,
                rte_pktmbuf_mtod(m, void *), (unsigned)m->data_len,m->next);
        m = m->next;
        nb_segs--;
    }
}
#endif
static inline uint8_t tcp_process_fin(const uint16_t flow_id,lcore_param *rx_conf,tcp_control_block_t *block, uint8_t rx_flags, uint8_t tx_flags) {
    uint16_t state = rte_atomic16_read(&block->tcb_state); 
    uint8_t flags = 0;
#if 1
    switch (state) {
        case TCP_ESTABLISHED:
            if (rx_flags & RTE_TCP_FIN_FLAG) {
		rte_atomic16_set(&block->tcb_state, TCP_LAST_ACK);
                flags = RTE_TCP_FIN_FLAG | RTE_TCP_ACK_FLAG;
                printf("********** recv fin and start  to send fin **** \n");
                tcp_payload_size = 0;
            } else if (tx_flags & RTE_TCP_FIN_FLAG) {
		rte_atomic16_set(&block->tcb_state, TCP_FIN_WAIT_I);
                flags = RTE_TCP_FIN_FLAG | RTE_TCP_ACK_FLAG;
            }
            break;
        case TCP_FIN_WAIT_I:
            if (rx_flags & RTE_TCP_FIN_FLAG) {
                flags =  RTE_TCP_ACK_FLAG;
                /* enter TIME WAIT */
                //socket_close(sk);
                // not need reset, will cause state=TCP_INIT and not enter tcp_process_fin again
                //reset_tcp_blocks(block);
            } else {
                /* wait FIN */
		rte_atomic16_set(&block->tcb_state, TCP_FIN_WAIT_II);
            }
            break;
        case TCP_CLOSING: // half-open
            /* In order to prevent the loss of fin, we make up a FIN, which has no cost */
            rte_atomic16_set(&block->tcb_state, TCP_LAST_ACK);
            flags =  RTE_TCP_FIN_FLAG | RTE_TCP_ACK_FLAG;
            break;
        case TCP_LAST_ACK:
            //socket_close(sk);
            reset_tcp_blocks(block);
             // in TCP_LAST_ACK state ,recv ack not need tcp_reply
            //flags =  RTE_TCP_ACK_FLAG;
            break;
        case TCP_FIN_WAIT_II:
            /* FIN is here */
            if (rx_flags & RTE_TCP_FIN_FLAG) {
                flags =  RTE_TCP_ACK_FLAG;
                /* enter TIME WAIT */
                //socket_close(sk);
                reset_tcp_blocks(block);
            }
            break;
        case TCP_TIME_WAIT:
        default:
            break;
    }
#endif
    flags = tx_flags | flags;
    if(flags){
    tcp_reply(flow_id,rx_conf, flags);
    }
    return 0;
}
static bool check_tcp_close_state(tcp_control_block_t *block){
    uint16_t state = rte_atomic16_read(&block->tcb_state); 
    return (TCP_FIN_WAIT_I == state || TCP_CLOSING == state || TCP_LAST_ACK == state || TCP_FIN_WAIT_II == state || TCP_TIME_WAIT == state);
} 
static void update_tcp_wind(tcp_control_block_t *block,  struct rte_tcp_hdr *tcp_hdr, uint32_t packet_data_size)
{
	// update ACK number in the TCP control block from the packet
	uint32_t ack_cur = rte_be_to_cpu_32(rte_atomic32_read(&block->tcb_next_ack));
	uint32_t ack_hdr = rte_be_to_cpu_32(tcp_hdr->sent_seq) + (packet_data_size);
	if(SEQ_LEQ(ack_cur, ack_hdr)) {
		//rte_atomic32_set(&block->tcb_next_ack, tcp_hdr->sent_seq + rte_cpu_to_be_32(packet_data_size));
		rte_atomic32_set(&block->tcb_next_ack, rte_cpu_to_be_32(rte_be_to_cpu_32(tcp_hdr->sent_seq) + packet_data_size));
	}
         
}
static void update_tcp_fin_wind(tcp_control_block_t *block,  struct rte_tcp_hdr *tcp_hdr, uint32_t packet_data_size)
{
	// update ACK number in the TCP control block from the packet
	rte_atomic32_set(&block->tcb_next_ack, rte_cpu_to_be_32(rte_be_to_cpu_32(tcp_hdr->sent_seq) + packet_data_size));
         
}
// Process the incoming TCP packet
int process_rx_pkt(struct rte_mbuf *pkt, node_t *incoming, uint64_t *incoming_idx,lcore_param *rx_conf) {
	// process only TCP packets
	struct rte_ipv4_hdr *ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
	if(unlikely(ipv4_hdr->next_proto_id != IPPROTO_TCP)) {
		return 0;
	}

	// get TCP header
	struct rte_tcp_hdr *tcp_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_tcp_hdr *, sizeof(struct rte_ether_hdr) + (ipv4_hdr->version_ihl & 0x0f)*4);
       
#if     NIC_SUPPPORT_FLOW_OFFLOAD
	// retrieve the index of the flow from the NIC (NIC tags the packet according the 5-tuple using DPDK rte_flow)
	uint32_t flow_id = pkt->hash.fdir.hi;

	// get control block for the flow
	tcp_control_block_t *block = &tcp_control_blocks[flow_id];
#else
	uint32_t flow_id= 0;
	tcp_control_block_t *block = get_tcp_control_block(ipv4_hdr, tcp_hdr,&flow_id);
        if(NULL == block) {
             printf("find tcp control block fail \n");
             //rte_pktmbuf_free(pkt);
             return 0;
        }
#endif
	// update receive window from the packet
	rte_atomic16_set(&block->tcb_rwin, tcp_hdr->rx_win);

	// get TCP payload size
	uint32_t packet_data_size = rte_be_to_cpu_16(ipv4_hdr->total_length) - ((ipv4_hdr->version_ihl & 0x0f)*4) - ((tcp_hdr->data_off >> 4)*4);

	if((tcp_hdr->tcp_flags & RTE_TCP_FIN_FLAG) || check_tcp_close_state(block)) {
            dpdk_dump_tcph(tcp_hdr,packet_data_size + (tcp_hdr->data_off >> 4)*4);
            if(0 == packet_data_size && (tcp_hdr->tcp_flags & RTE_TCP_FIN_FLAG)){
                 // fin need +1
                 packet_data_size +=1; 
            }
            update_tcp_fin_wind(block,tcp_hdr,packet_data_size);
            return tcp_process_fin(flow_id, rx_conf,block,tcp_hdr->tcp_flags,0); 
        }
	// do not process empty packets
	if(unlikely(packet_data_size == 0)) {
		return 0;
	}
        dpdk_dump_tcph(tcp_hdr,packet_data_size + (tcp_hdr->data_off >> 4)*4);
	// do not process retransmitted packets
	uint32_t seq = rte_be_to_cpu_32(tcp_hdr->sent_seq);
	if(SEQ_LT(block->last_seq_recv, seq)) {
		block->last_seq_recv = seq;
	} else {
		return 0;
	}
        update_tcp_wind(block, tcp_hdr,packet_data_size); 
	// obtain both timestamp from the packet
        //dump_tcp_buf(pkt);
	char *payload = (char*)(((char*) tcp_hdr) + ((tcp_hdr->data_off >> 4)*4));
#if 1
        char buf[TCP_BUF_LEN] = {0};
        memset(buf,0,TCP_BUF_LEN);
        snprintf(buf,packet_data_size, "%s",payload);
        printf("tcp payload offset %u, payload %s \n",((tcp_hdr->data_off >> 4)*4),buf);
#endif
        tcp_reply(flow_id,rx_conf,RTE_TCP_ACK_FLAG);
	return 1;
}

// Start the client establishing all TCP connections
void start_client(uint16_t portid) {
	uint16_t nb_rx;
	uint16_t nb_tx;
	uint64_t ts_syn;
	uint32_t nb_retransmission;
	struct rte_mbuf *pkt;
	tcp_control_block_t *block;
	struct rte_mbuf *pkts[BURST_SIZE];
        int i = 0;
	for(i = 0; i < nr_flows; i++) {
		// get the TCP control block for the flow
		block = &tcp_control_blocks[i];
		// create the TCP SYN packet
		struct rte_mbuf *syn_packet = create_syn_packet(i);
		// insert the rte_flow in the NIC to retrieve the flow id for incoming packets of this flow
#if NIC_SUPPPORT_FLOW_OFFLOAD
		insert_flow(portid, i);
#endif
		// send the SYN packet
		nb_tx = rte_eth_tx_burst(portid, i % nr_queues, &syn_packet, 1);
		if(nb_tx != 1) {
			rte_exit(EXIT_FAILURE, "Error to send the TCP SYN packet.\n");
		}

		// clear the counters
		nb_retransmission = 0;
		ts_syn = rte_rdtsc();

		// change the TCP state to SYN_SENT
		rte_atomic16_set(&block->tcb_state, TCP_SYN_SENT);

		// while not received SYN+ACK packet and TCP state is not ESTABLISHED
		while(rte_atomic16_read(&block->tcb_state) != TCP_ESTABLISHED) {
			// receive TCP SYN+ACK packets from the NIC
			nb_rx = rte_eth_rx_burst(portid, i % nr_queues, pkts, BURST_SIZE);

                        int j = 0;
			for(j = 0; j < nb_rx; j++) {
				// process the SYN+ACK packet, returning the ACK packet to send
				pkt = process_syn_ack_packet(pkts[j]);
				
				if(pkt) {
					// send the TCP ACK packet to the server
					nb_tx = rte_eth_tx_burst(portid, i % nr_queues, &pkt, 1);
					if(nb_tx != 1) {
						rte_exit(EXIT_FAILURE, "Error to send the TCP ACK packet.\n");
					}
				}
                                else
                                {
			            rte_pktmbuf_free(pkts[j]);
                                }
			}
#if 0
			// free packets
			rte_pktmbuf_free_bulk(pkts, nb_rx);
#endif

			if((rte_rdtsc() - ts_syn) > HANDSHAKE_TIMEOUT_IN_US * TICKS_PER_US) {
				nb_retransmission++;
				nb_tx = rte_eth_tx_burst(portid, i % nr_queues, &syn_packet, 1);
				if(nb_tx != 1) {
						rte_exit(EXIT_FAILURE, "Error to send the TCP SYN packet.\n");
				}
				ts_syn = rte_rdtsc();

				if(nb_retransmission == HANDSHAKE_RETRANSMISSION) {
					rte_exit(EXIT_FAILURE, "Cannot establish connection.\n");
				}
			}
		}
	}

	// Discard 3-way handshake packets in the DPDK metrics
	rte_eth_stats_reset(portid);
	rte_eth_xstats_reset(portid);
	
	rte_compiler_barrier();
}
static void tcp_reply(const uint16_t flow_id,lcore_param *rx_conf, uint8_t flag )
{
        const uint8_t qid = rx_conf->qid;
        const uint8_t portid = rx_conf->portid;
        //const uint8_t flag = RTE_TCP_ACK_FLAG;
	uint16_t nb_tx;
	uint16_t nb_pkts = 1;
	// choose the flow to send
	tcp_control_block_t *block = &tcp_control_blocks[flow_id];
        struct rte_mbuf * pkt;
	// generate packets
	pkt = rte_pktmbuf_alloc(pktmbuf_pool);
	// check receive window for that flow
	uint16_t rx_wnd = rte_be_to_cpu_32(rte_atomic16_read(&block->tcb_rwin));
#if 1
        printf("rx_wnd %u, tcp_payload_size: %u \n",rx_wnd , tcp_payload_size);
	if(unlikely(rx_wnd < tcp_payload_size)) { 
             tcp_payload_size = rx_wnd;
	}
#endif
	// fill the packet with the flow information
	fill_tcp_packet(flow_id, pkt,flag);
	//dump_tcp_buf(pkt);
	//fill the payload to gather server information
	// send the batch
        //dump_pcap(rte_pktmbuf_mtod(pkt, char*),pkt->pkt_len);
	nb_tx = rte_eth_tx_burst(portid, qid, &pkt, nb_pkts);
	if(unlikely(nb_tx != nb_pkts)) {
                rte_pktmbuf_free(pkt);
		rte_exit(EXIT_FAILURE, "Cannot send the target packets.\n");
	}
}
// RX processing
static int lcore_rx_ring(void *arg) {
	lcore_param *rx_conf = (lcore_param *) arg;
	uint8_t qid = rx_conf->qid;

	uint16_t nb_rx;
	uint64_t *incoming_idx = &incoming_idx_array[qid];
	node_t *incoming = incoming_array[qid];
	struct rte_mbuf *pkts[BURST_SIZE];
	struct rte_ring *rx_ring = rx_rings[qid];

	while(!quit_rx_ring) {
		// retrieve packets from the RX core
		nb_rx = rte_ring_sc_dequeue_burst(rx_ring, (void**) pkts, BURST_SIZE, NULL); 
                int i = 0;
		for(i = 0; i < nb_rx; i++) {
			rte_prefetch_non_temporal(rte_pktmbuf_mtod(pkts[i], void *));
			// process the incoming packet
			process_rx_pkt(pkts[i], incoming, incoming_idx,rx_conf);
			// free the packet
			rte_pktmbuf_free(pkts[i]);
		}
	}

	// process all remaining packets that are in the RX ring (not from the NIC)
	do{
		nb_rx = rte_ring_sc_dequeue_burst(rx_ring, (void**) pkts, BURST_SIZE, NULL);
                int i = 0;
		for(i = 0; i < nb_rx; i++) {
			rte_prefetch_non_temporal(rte_pktmbuf_mtod(pkts[i], void *));
			// process the incoming packet
			process_rx_pkt(pkts[i], incoming, incoming_idx,rx_conf);
			// free the packet
			rte_pktmbuf_free(pkts[i]);
		}
	} while (nb_rx != 0);

	return 0;
}

// Main RX processing
static int lcore_rx(void *arg) {
	lcore_param *rx_conf = (lcore_param *) arg;
	uint16_t portid = rx_conf->portid;
	uint8_t qid = rx_conf->qid;

	uint16_t nb_rx;
	struct rte_mbuf *pkts[BURST_SIZE];
	struct rte_ring *rx_ring = rx_rings[qid];
	
	while(!quit_rx) {
		// retrieve packets from the NIC
		nb_rx = rte_eth_rx_burst(portid, qid, pkts, BURST_SIZE);
		// retrive the current timestamp 
		if(rte_ring_sp_enqueue_burst(rx_ring, (void* const*) pkts, nb_rx, NULL) != nb_rx) {
			rte_exit(EXIT_FAILURE, "Cannot enqueue the packet to the RX thread: %s.\n", rte_strerror(errno));
		}
	}

	return 0;
}
// main function
int main(int argc, char **argv) {
	// init EAL
	int ret = rte_eal_init(argc, argv);
	if(ret < 0) {
		rte_exit(EXIT_FAILURE, "Invalid EAL parameters\n");
	}
	argc -= ret;
	argv += ret;

	// parse application arguments (after the EAL ones)
	ret = app_parse_args(argc, argv);
	if(ret < 0) {
		rte_exit(EXIT_FAILURE, "Invalid arguments\n");
	}

	// initialize DPDK
	uint16_t portid = 0;
	init_DPDK(portid, nr_queues);

	// allocate nodes for incoming packets
	allocate_incoming_nodes();

	// create flow indexes array
	create_flow_indexes_array();

	// create interarrival array
	create_interarrival_array();
	// prepare throughput tracking
	prepare_throughput_tracking();
	// initialize TCP control blocks
	init_tcp_blocks();

	// start client (3-way handshake for each flow)
	start_client(portid);

	// create the DPDK rings for RX threads
	create_dpdk_rings();

#if DBUG_TCP
	open_pcap("dpdk0");
#endif
	// start RX and TX threads
	uint32_t id_lcore = rte_lcore_id();	
        int i = 0;
	for(i = 0; i < nr_queues; i++) {
		lcore_params[i].portid = portid;
		lcore_params[i].qid = i;
		lcore_params[i].nr_elements = (rate/nr_queues) * duration * nr_executions;

		id_lcore = rte_get_next_lcore(id_lcore, 1, 1);
		rte_eal_remote_launch(lcore_rx_ring, (void*) &lcore_params[i], id_lcore);

		id_lcore = rte_get_next_lcore(id_lcore, 1, 1);
		rte_eal_remote_launch(lcore_rx, (void*) &lcore_params[i], id_lcore);

#if 0
		id_lcore = rte_get_next_lcore(id_lcore, 1, 1);
		rte_eal_remote_launch(lcore_tx, (void*) &lcore_params[i], id_lcore);
#endif
	}

	// wait for duration parameter
	wait_timeout();

	// wait for RX/TX threads
	uint32_t lcore_id;
	RTE_LCORE_FOREACH_WORKER(lcore_id) {
		if(rte_eal_wait_lcore(lcore_id) < 0) {
			return -1;
		}
	}

	// print stats
	//print_stats_output();

	// print DPDK stats
	//print_dpdk_stats(portid);

	// clean up
	clean_heap();
	clean_hugepages();

	return 0;
}
