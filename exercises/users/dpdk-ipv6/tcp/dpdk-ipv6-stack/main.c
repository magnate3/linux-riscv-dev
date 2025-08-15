#include <signal.h>
#include <stdbool.h>
#include <getopt.h>

#include <pcap.h>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include <rte_byteorder.h>
#include <rte_log.h>
#include <rte_common.h>
#include <rte_config.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include "help.h"
#include "util.h"
#include "dpdk.h"
#include "dpdk_ip.h"
#include "dpdk_eth.h"
#include "dpdk_ipv6_ndic.h"
#include "dpdk_work_space.h"
#include "dpdk_config.h"
#include "dpdk_reassembly.h"
#include <rte_cycles.h>
#define US_PER_S 1000000

#define BURST_SIZE 32
#define APP "pingpong"
static struct rte_ether_addr server_ether_addr =
    {{0x44, 0xa1, 0x91, 0xa4, 0x9b, 0xeb}};
uint32_t PINGPONG_LOG_LEVEL = RTE_LOG_DEBUG;

struct netif_port  port0;

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
struct rte_mempool *nat_pktmbuf_pool = NULL;

static volatile bool force_quit;

/* enabled port */
static uint16_t portid = 0;
/* number of packets */
static uint64_t nb_pkts = 100;
/* server mode */
static bool server_mode = false;

static struct rte_eth_dev_tx_buffer *tx_buffer;


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

#define IP_DEFTTL 64 /* from RFC 1340. */
#define IP_VERSION 0x40
#define IP_HDRLEN 0x05 /* default IP header length == five 32-bits words. */
#define IP_VHL_DEF (IP_VERSION | IP_HDRLEN)
#define IP_ADDR_FMT_SIZE 15


static void
signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM)
    {
        rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "\n\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}


#if 0
static void main_loop(void)
{
    uint64_t start_tsc;
    uint64_t end_tsc;
    uint64_t core_cycles;
    start_tsc = rte_rdtsc();
    while (!force_quit) 
    {
        end_tsc = rte_rdtsc();
 	core_cycles = (end_tsc - start_tsc);
        if(core_cycles > 1000000)
        {
            icmp6_ns_request();
            start_tsc = end_tsc;
            //icmp6_echo_request();
        }
   }
}
#else
static void main_loop(void)
{
    while (!force_quit) 
    {
            rte_delay_ms(1000);
            icmp6_ns_request();
            rte_delay_ms(1000);
            //icmp6_big_echo_request();
            //icmp6_echo_request();
   }
}
#endif
void server_loop(void)
{
  
    uint16_t nb_rx;
    //uint16_t i;
    unsigned int lcore_id = rte_lcore_id();
    struct rte_mbuf *bufs[BURST_SIZE];

    //RTE_LOG(INFO, APP, "lcore %u running\n", lcore_id);

    g_work_space = work_space_new(&g_config,lcore_id); 
    while (!force_quit) 
    {


        // 接受数据包
        nb_rx = rte_eth_rx_burst(portid, 0,
            bufs, BURST_SIZE);

        if (unlikely(nb_rx == 0))
            continue;
#if 1
           
           lcore_process_packets(bufs,nb_rx,portid);
           tick_time_update(&g_work_space->time);
           rte_ip_frag_free_death_row(&(netif_port_get(portid)->death_row),
                                PREFETCH_OFFSET);
#else
            struct timeval tv;
            gettimeofday(&tv, NULL);
            char *pktbuf = rte_pktmbuf_mtod(bufs[i], char *);
            dpdk_ip_proto_process_raw(bufs[i]);
#endif
#if 0
            struct  rte_mbuf * arp_pkt=  send_arp(get_mbufpool(0),server_ether_addr.addr_bytes, client_ether_addr.addr_bytes,12345,123455);
            ipv6_xmit(arp_pkt);
            rte_pktmbuf_free(bufs[i]);
#else
#endif
    }

    //RTE_LOG(INFO, APP, "lcore %u exiting\n", lcore_id);
    return ;
}

static int
pong_launch_one_lcore(__attribute__((unused)) void *dummy)
{
    //printf("tid is %lu \n",pthread_self());
    server_loop();
    return 0;
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
static const char short_options[] =
    "p:" /* portmask */
    "n:" /* number of packets */
    "s"  /* server mode */
    ;
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

    /* init log */
    RTE_LOGTYPE_PINGPONG = rte_log_register(APP);
    ret = rte_log_set_level(RTE_LOGTYPE_PINGPONG, PINGPONG_LOG_LEVEL);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Set log level to %u failed\n", PINGPONG_LOG_LEVEL);
    
    nb_lcores = rte_lcore_count();
    if (nb_lcores < 2)
        rte_exit(EXIT_FAILURE, "Number of CPU cores should be no less than 2.");

    nb_ports = rte_eth_dev_count_avail();
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

#if 1
    nat_pktmbuf_pool = rte_pktmbuf_pool_create("nat_mbuf_pool", nb_mbufs,
                                                    MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
                                                    rte_socket_id());
    if (nat_pktmbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot init nat mbuf pool\n");
#else
    nat_pktmbuf_pool = pingpong_pktmbuf_pool;
#endif
    struct rte_eth_rxconf rxq_conf;
    struct rte_eth_txconf txq_conf;
    struct rte_eth_conf local_port_conf = port_conf;
    struct rte_eth_dev_info dev_info;

    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "Initializing port %u...\n", portid);
    fflush(stdout);

    ret = rte_eth_macaddr_get(portid, &server_ether_addr);
    if (ret != 0)
          rte_exit(EXIT_FAILURE, "macaddr get failed\n");
    
#if 0
    char mac[24];
    rte_ether_format_addr(&mac[0], 24, &server_ether_addr);
    printf("port: %d->MAC-> %s\n", portid, mac);
#endif
    //rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "server_ether_addr  %p...\n", &server_ether_addr);
    ether_addr_dump("  port 0 mac addr =", &server_ether_addr);
    /* init port */
    rte_eth_dev_info_get(portid, &dev_info);
    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
        local_port_conf.txmode.offloads |=
            DEV_TX_OFFLOAD_MBUF_FAST_FREE;
    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_IPV4_CKSUM) {
        local_port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM;
        g_dev_tx_offload_ipv4_cksum = 1;
    } else {
        g_dev_tx_offload_ipv4_cksum = 0;
    }

    if ((dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_TCP_CKSUM) &&
        (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_UDP_CKSUM)) {
        g_dev_tx_offload_tcpudp_cksum = 1;
        local_port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_TCP_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM;
    } else {
        g_dev_tx_offload_tcpudp_cksum = 0;
    }

    ret = rte_eth_dev_configure(portid, 1, 1, &local_port_conf);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n",
                 ret, portid);

    ret = rte_eth_dev_adjust_nb_rx_tx_desc(portid, &nb_rxd,
                                           &nb_txd);
    if (ret < 0)
        rte_exit(EXIT_FAILURE,
                 "Cannot adjust number of descriptors: err=%d, port=%u\n",
                 ret, portid);

    /* init one RX queue */
    fflush(stdout);
    rxq_conf = dev_info.default_rxconf;

    rxq_conf.offloads = local_port_conf.rxmode.offloads;
    ret = rte_eth_rx_queue_setup(portid, 0, nb_rxd,
                                 rte_eth_dev_socket_id(portid),
                                 &rxq_conf,
                                 pingpong_pktmbuf_pool);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n",
                 ret, portid);
    init_mbuf(nat_pktmbuf_pool); 
    /* init one TX queue on each port */
    fflush(stdout);
    txq_conf = dev_info.default_txconf;
    txq_conf.offloads = local_port_conf.txmode.offloads;
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

    rte_eth_tx_buffer_init(tx_buffer, MAX_PKT_BURST);

    ret = rte_eth_tx_buffer_set_err_callback(tx_buffer,
                                             rte_eth_tx_buffer_count_callback,
                                             &port_statistics.dropped);
    if (ret < 0)
        rte_exit(EXIT_FAILURE,
                 "Cannot set error callback for tx buffer on port %u\n",
                 portid);
    init_ndisc();
    // init_ndisc  must before init_dpdk_eth_mod for ip_range_init
    init_dpdk_eth_mod();
    init_netif(&port0,nat_pktmbuf_pool,portid,1);
    netif_port_register(&port0);
    init_config();
    /* Start device */
    ret = rte_eth_dev_start(portid);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n",
                 ret, portid);

    // 开启混杂模式
    rte_eth_promiscuous_enable(portid);
    /* initialize port stats */
    initlize_port_statistics();

    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "Initilize port %u done.\n", portid);

    lcore_id = rte_get_next_lcore(0, true, false);

    ret = 0;
    
    rte_eal_remote_launch(pong_launch_one_lcore, NULL, lcore_id);
    //printf("tid is %lu \n",pthread_self());
    main_loop();
    //RTE_LCORE_FOREACH_SLAVE(lcore_id) {
       if (rte_eal_wait_lcore(lcore_id) < 0)
       {
           ret = -1;
       }
    //}
    rte_eth_dev_stop(portid);
    rte_eth_dev_close(portid);
    destroy_port_statistics();
    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_PINGPONG, "Bye.\n");
    
    return 0;
}
