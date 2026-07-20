#include <stdbool.h>
#include <getopt.h>
#include <stdlib.h>

// #include "globals.h"
#include "final.h"

#define APP "dpdk-hping"

uint32_t DPDK_HPING_LOG_LEVEL = RTE_LOG_DEBUG;

/* the client side MAC Adress */
static struct rte_ether_addr target_ether_addr;

/* the server side MAC Adress */
static struct rte_ether_addr my_ether_addr;

#define MAX_PKT_BURST 32
#define MEMPOOL_CACHE_SIZE 128

/*
 * Configurable number of RX/TX ring descriptors
 */
#define RTE_TEST_RX_DESC_DEFAULT 1024
#define RTE_TEST_TX_DESC_DEFAULT 1024
static uint16_t nb_rxd = RTE_TEST_RX_DESC_DEFAULT;
static uint16_t nb_txd = RTE_TEST_TX_DESC_DEFAULT;

int RTE_LOGTYPE_DPDK_HPING;

struct rte_mempool *pktmbuf_pool = NULL;

/* enabled port */
static uint16_t portid = 0;

static struct rte_eth_conf port_config = {
    .rxmode = {
       .mq_mode = RTE_ETH_MQ_RX_NONE,
    },
    .txmode = {
        .mq_mode = RTE_ETH_MQ_TX_NONE,
    },
    .intr_conf = {
     .rxq = 1
    }
};
static int intr_en = 0;

static const char short_options[] =
    "p:" /* port id */
    "m:" /* maximum size of the message */
    "n:" /* minimum size of the message */
    "i:" /* number of interations */
    "c:" /* client mode, with MAC address */
    "s"  /* server mode */
    "l:" /*number of packets to be sent*/
    "W:" /*timeout interval*/
    "a:" /* server ip address */
    "b:" /* client ip address */
    ;

/* display usage */
static void pingpong_usage(const char *prgname)
{
    printf("%s [EAL options] --"
           "\t-p PORTID: port to configure\n"
           "\t-m BYTES: minimum size of the message\n"
           "\t-n BYTES: maximum size of the message\n"
           "\t-i ITERS: number of iterations\n"
           "\t-c TARGET_MAC: target MAC address\n"
           "\t-s: enable server mode\n",
           //
           //    "\t-t: time interval for which client pings\n",

           "\t-l: number of packets to be sent\n",
           "\t-W: timeout interval",

           //
           prgname);
}

/* Parse the argument given in the command line of the application */
static int dpdk_hping_parse_args(int argc, char **argv)
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

        case 's':
            server_mode = true;
            break;

        case 'c':
        {
            const char *PARSE_STRING = "%02X:%02X:%02X:%02X:%02X:%02X";
            sscanf(optarg, PARSE_STRING,
                   &target_ether_addr.addr_bytes[0],
                   &target_ether_addr.addr_bytes[1],
                   &target_ether_addr.addr_bytes[2],
                   &target_ether_addr.addr_bytes[3],
                   &target_ether_addr.addr_bytes[4],
                   &target_ether_addr.addr_bytes[5]);
            break;
        }

        case 'l':
        {
            total_packets = (uint16_t)strtol(optarg, NULL, 10);
            is_not_limited = false;
            break;
        }

        case 'W':
        {
            time_out_value = (unsigned int)strtol(optarg, NULL, 10);
            break;
        }

        case 'a':
        {
            const char *PARSE_STRING = "%d.%d.%d.%d";
            int ip_parts[4];
            sscanf(optarg, PARSE_STRING,
                   &ip_parts[0],
                   &ip_parts[1],
                   &ip_parts[2],
                   &ip_parts[3]);
            server_ip_addr = RTE_IPV4(ip_parts[0], ip_parts[1], ip_parts[2], ip_parts[3]);
            break;
        }

        case 'b':
        {
            const char *PARSE_STRING = "%d.%d.%d.%d";
            int ip_parts[4];
            sscanf(optarg, PARSE_STRING,
                   &ip_parts[0],
                   &ip_parts[1],
                   &ip_parts[2],
                   &ip_parts[3]);
            client_ip_addr = RTE_IPV4(ip_parts[0], ip_parts[1], ip_parts[2], ip_parts[3]);
            break;
        }

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

/* construct ping packet */
static struct rte_mbuf *create_packet(unsigned pkt_size)
{
    struct rte_mbuf *pkt;
    struct rte_ether_hdr *eth_hdr;

    pkt = rte_pktmbuf_alloc(pktmbuf_pool);
    if (!pkt)
        rte_log(RTE_LOG_ERR, RTE_LOGTYPE_DPDK_HPING, "fail to alloc mbuf for packet\n");

    pkt->next = NULL;

    /* Initialize Ethernet header. */
    eth_hdr = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr *);
    rte_ether_addr_copy(&target_ether_addr, &eth_hdr->dst_addr);
    rte_ether_addr_copy(&my_ether_addr, &eth_hdr->src_addr);
    eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

    l2_len = sizeof(struct rte_ether_hdr);

    uint32_t pkt_len = sizeof(struct rte_ether_hdr);

    if (!opt_only_eth)
    {
        pkt_len += add_ip(pkt, client_ip_addr, server_ip_addr);
    }

    pkt->data_len = pkt_len;

    return pkt;
}

/* main client loop */
static void client_main_loop(uint64_t nb_bytes)
{
    unsigned nb_rx, nb_tx;
    const uint64_t tsc_hz = rte_get_tsc_hz();
    struct rte_ether_hdr *eth_hdr;
    struct rte_mbuf *pkts_burst[MAX_PKT_BURST];
    int i = 0;
    int num_packets = 0;
    // change this
    unsigned nb_pkts = (nb_bytes + RTE_ETHER_MTU - 1) / RTE_ETHER_MTU;
    struct rte_mbuf **pkts = malloc(nb_pkts * sizeof(struct rte_mbuf *));
    for (i = 0; i < nb_pkts - 1; ++i)
    {
        pkts[i] = create_packet(RTE_ETHER_MTU);
    }
    pkts[nb_pkts - 1] = create_packet(nb_bytes % RTE_ETHER_MTU);
    // change this

    Signal(SIGALRM, timeout_handler);
    Signal(SIGINT, print_statistics);
    Signal(SIGTERM, print_statistics);

    for (num_packets = 0; (num_packets < total_packets) || is_not_limited; num_packets++)
    {
        /* do ping */
        nb_tx = 0;
        while (nb_tx < nb_pkts)
        {
            nb_tx += rte_eth_tx_burst(portid, 0, pkts + nb_tx, nb_pkts - nb_tx);
            alarm(time_out_value);
            delaytable_add(seq_nb, 0, time(NULL), get_usec(), S_SENT);
            packets_sent++;
        }

        /* wait for pong */
        nb_rx = 0;
        while (nb_rx < nb_pkts && !is_timed_out)
        {
            unsigned nb_rx_once = rte_eth_rx_burst(portid, 0, pkts_burst, MAX_PKT_BURST);
            if (nb_rx_once)
            {
                for (i = 0; i < nb_rx_once; ++i)
                {
                    pkts[nb_rx + i] = pkts_burst[i];
                    eth_hdr = rte_pktmbuf_mtod(pkts[nb_rx + i], struct rte_ether_hdr *);
                    /* compare mac, confirm it is a pong packet */
                    // assert(rte_is_same_ether_addr(&eth_hdr->dst_addr, &my_ether_addr));

                    if (rte_is_same_ether_addr(&eth_hdr->dst_addr, &my_ether_addr))
                    {
                        rte_ether_addr_copy(&eth_hdr->src_addr, &eth_hdr->dst_addr);
                        rte_ether_addr_copy(&my_ether_addr, &eth_hdr->src_addr);
                        parse_client(pkts[nb_rx + i]);

                        if (correct_packet)
                        {
                            correct_packet = true;
                            nb_rx++;
                            packets_recv++;
                        }
                    }
                }
            }
        }
        if (is_timed_out)
        {
            printf("Request timed out\n");
            is_timed_out = false;
            update_seq_nb(pkts[0]);
        }
    }

    print_statistics(0);

    for (i = 0; i < nb_pkts; ++i)
    {
        rte_pktmbuf_free(pkts[i]);
    }
    free(pkts);
}
static int event_register(uint16_t portid, uint8_t queueid)
{
        uint32_t data;
        int ret;
        data = portid << CHAR_BIT | queueid;
        ret = rte_eth_dev_rx_intr_ctl_q(portid, queueid, RTE_EPOLL_PER_THREAD, RTE_INTR_EVENT_ADD, (void *)((uintptr_t)data));
        return ret;
}
static void turn_on_off_intr(uint16_t portid, uint8_t queueid, int on)
{
        int ret= 0;
        if(on)
        {
                ret = rte_eth_dev_rx_intr_enable(portid, queueid);
                if(ret < 0) 
                   printf("rx interrupt enable fail \n");
        }
        else
                rte_eth_dev_rx_intr_disable(portid, queueid);
}
static int
sleep_until_rx_interrupt(int num, int lcore)
{
        /*
 *          * we want to track when we are woken up by traffic so that we can go
 *                   * back to sleep again without log spamming. Avoid cache line sharing
 *                            * to prevent threads stepping on each others' toes.
 *                                     */
        static struct {
                bool wakeup;
        } __rte_cache_aligned status[RTE_MAX_LCORE];
        struct rte_epoll_event event[num];
        int n, i;
        uint16_t port_id;
        uint8_t queue_id;
        void *data;

        if (status[lcore].wakeup) {
                rte_log(RTE_LOG_INFO, RTE_LOGTYPE_DPDK_HPING,
                                "lcore %u sleeps until interrupt triggers\n",
                                rte_lcore_id());
        }

        n = rte_epoll_wait(RTE_EPOLL_PER_THREAD, event, num, 10);
        for (i = 0; i < n; i++) {
                data = event[i].epdata.data;
                port_id = ((uintptr_t)data) >> CHAR_BIT;
                queue_id = ((uintptr_t)data) &
                        RTE_LEN2MASK(CHAR_BIT, uint8_t);
                rte_log(RTE_LOG_INFO, RTE_LOGTYPE_DPDK_HPING,
                        "lcore %u is waked up from rx interrupt on"
                        " port %d queue %d\n",
                        rte_lcore_id(), port_id, queue_id);
        }
        status[lcore].wakeup = n != 0;

        return 0;
}
/* main server loop  */
static void server_main_loop(uint64_t nb_bytes)
{
    unsigned nb_rx, nb_tx;
    struct rte_mbuf *m = NULL;
    struct rte_ether_hdr *eth_hdr;
    struct rte_mbuf *pkts_burst[MAX_PKT_BURST];
    int i = 0;
    struct rte_mbuf **pkts = malloc(MAX_PKT_BURST * sizeof(struct rte_mbuf *));
        if(event_register(portid, 0) < 0)
        {
             printf("interrupt event register fails \n");
        }
        else
        {
             intr_en = 1;
             printf("interrupt event register \n");
        }
    /* wait for pong */
    while (true)
    {
        if(1 == intr_en)
        {
            turn_on_off_intr(portid, 0, 1);
            sleep_until_rx_interrupt(1, rte_lcore_id());
            turn_on_off_intr(portid, 0, 0);
        }
        /* wait for ping */
        unsigned nb_rx_once = rte_eth_rx_burst(portid, 0, pkts_burst, MAX_PKT_BURST);
        if (nb_rx_once)
        {
            for (i = 0; i < nb_rx_once; ++i)
            {
                pkts[nb_rx + i] = pkts_burst[i];
                eth_hdr = rte_pktmbuf_mtod(pkts[nb_rx + i], struct rte_ether_hdr *);

                /* compare mac, confirm it is a pong packet */
                if (rte_is_same_ether_addr(&eth_hdr->dst_addr, &my_ether_addr))
                {
                    rte_ether_addr_copy(&eth_hdr->src_addr, &eth_hdr->dst_addr);
                    rte_ether_addr_copy(&my_ether_addr, &eth_hdr->src_addr);

                    parser_server(pkts[nb_rx + i]);
                }
            }

            nb_tx = 0;
            while (nb_tx < nb_rx_once)
            {
                nb_tx += rte_eth_tx_burst(portid, 0, pkts + nb_tx, nb_rx_once - nb_tx);
            }
        }
    }

    free(pkts);
}

static int client_launch_one_lcore(__attribute__((unused)) void *dummy)
{
    unsigned lcore_id;
    lcore_id = rte_lcore_id();

    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_DPDK_HPING,
            "entering ping loop on lcore %u\n", lcore_id);
    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_DPDK_HPING,
            "target MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n",
            target_ether_addr.addr_bytes[0],
            target_ether_addr.addr_bytes[1],
            target_ether_addr.addr_bytes[2],
            target_ether_addr.addr_bytes[3],
            target_ether_addr.addr_bytes[4],
            target_ether_addr.addr_bytes[5]);

    client_main_loop(data_size);

    return 0;
}

static int server_launch_one_lcore(__attribute__((unused)) void *dummy)
{
    unsigned lcore_id;
    lcore_id = rte_lcore_id();

    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_DPDK_HPING, "entering pong loop on lcore %u\n", lcore_id);
    rte_log(RTE_LOG_INFO, RTE_LOGTYPE_DPDK_HPING, "waiting ping packets\n");

    server_main_loop(data_size);

    return 0;
}

int main(int argc, char **argv)
{
    int return_value;
    uint16_t nb_ports;
    unsigned int nb_mbufs;
    unsigned int lcore_id;

    srand(time(NULL));

    /* init EAL */
    return_value = rte_eal_init(argc, argv);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE, "Invalid EAL arguments\n");
    argc -= return_value;
    argv += return_value;

    /* init log */
    RTE_LOGTYPE_DPDK_HPING = rte_log_register(APP);
    return_value = rte_log_set_level(RTE_LOGTYPE_DPDK_HPING, DPDK_HPING_LOG_LEVEL);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE, "Set log level to %u failed\n", DPDK_HPING_LOG_LEVEL);

    nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0)
        rte_exit(EXIT_FAILURE, "No Ethernet ports, bye...\n");

    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_DPDK_HPING, "%u port(s) available\n", nb_ports);

    /* parse application arguments (after the EAL ones) */
    return_value = dpdk_hping_parse_args(argc, argv);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE, "Invalid arguments\n");
    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_DPDK_HPING, "Enabled port: %u\n", portid);
    if (portid > nb_ports - 1)
        rte_exit(EXIT_FAILURE, "Invalid port id %u, port id should be in range [0, %u]\n", portid, nb_ports - 1);

    nb_mbufs = RTE_MAX((unsigned int)(nb_ports * (nb_rxd + nb_txd + MAX_PKT_BURST + MEMPOOL_CACHE_SIZE)), 8192U);
    pktmbuf_pool = rte_pktmbuf_pool_create("mbuf_pool", nb_mbufs,
                                           MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
                                           rte_socket_id());
    if (pktmbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot init mbuf pool\n");

    struct rte_eth_rxconf rxq_conf;
    struct rte_eth_txconf txq_conf;
    struct rte_eth_conf local_port_config = port_config;
    struct rte_eth_dev_info dev_info;

    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_DPDK_HPING, "Initializing port %u...\n", portid);
    fflush(stdout);

    /* init port */
    rte_eth_dev_info_get(portid, &dev_info); // DEV_TX_OFFLOAD_MBUF_FAST_FREE
    if (dev_info.tx_offload_capa & RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE)
        local_port_config.txmode.offloads |=
            RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;

    return_value = rte_eth_dev_configure(portid, 1, 1, &local_port_config);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE, "Cannot configure device: err=%d, port=%u\n",
                 return_value, portid);

    return_value = rte_eth_dev_adjust_nb_rx_tx_desc(portid, &nb_rxd,
                                                    &nb_txd);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE,
                 "Cannot adjust number of descriptors: err=%d, port=%u\n",
                 return_value, portid);

    return_value = rte_eth_macaddr_get(portid, &my_ether_addr);
    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_DPDK_HPING,
            "my MAC address:     %02X:%02X:%02X:%02X:%02X:%02X\n",
            my_ether_addr.addr_bytes[0],
            my_ether_addr.addr_bytes[1],
            my_ether_addr.addr_bytes[2],
            my_ether_addr.addr_bytes[3],
            my_ether_addr.addr_bytes[4],
            my_ether_addr.addr_bytes[5]);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE,
                 "Cannot get MAC address: err=%d, port=%u\n",
                 return_value, portid);

    /* init one RX queue */
    fflush(stdout);
    rxq_conf = dev_info.default_rxconf;

    rxq_conf.offloads = local_port_config.rxmode.offloads;
    return_value = rte_eth_rx_queue_setup(portid, 0, nb_rxd,
                                          rte_eth_dev_socket_id(portid),
                                          &rxq_conf,
                                          pktmbuf_pool);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n",
                 return_value, portid);

    /* init one TX queue on each port */
    fflush(stdout);
    txq_conf = dev_info.default_txconf;
    txq_conf.offloads = local_port_config.txmode.offloads;
    return_value = rte_eth_tx_queue_setup(portid, 0, nb_txd,
                                          rte_eth_dev_socket_id(portid),
                                          &txq_conf);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup:err=%d, port=%u\n",
                 return_value, portid);

    if (return_value < 0)
        rte_exit(EXIT_FAILURE,
                 "Cannot set error callback for tx buffer on port %u\n",
                 portid);

    /* Start device */
    return_value = rte_eth_dev_start(portid);
    if (return_value < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n",
                 return_value, portid);

    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_DPDK_HPING, "Initilize port %u done.\n", portid);

    lcore_id = rte_get_next_lcore(0, true, false);

    return_value = 0;
#if 1
    if (server_mode)
    {
        rte_eal_remote_launch(server_launch_one_lcore, NULL, lcore_id);
        //rte_eal_remote_launch(server_launch_one_lcore, NULL, CALL_MAIN);
        //rte_eal_mp_remote_launch(server_launch_one_lcore, NULL, CALL_MAIN);
    }
    else
    {
        rte_eal_remote_launch(client_launch_one_lcore, NULL, lcore_id);
    }

    if (rte_eal_wait_lcore(lcore_id) < 0)
    {
        return_value = -1;
    }
#else
        if(event_register(portid, 0) < 0)
        {
             printf("interrupt event register fails \n");
        }
        else
        {
             intr_en = 1;
             printf("interrupt event register \n");
        }
        server_launch_one_lcore(NULL);
#endif

    rte_eth_dev_stop(portid);
    rte_eth_dev_close(portid);
    rte_log(RTE_LOG_DEBUG, RTE_LOGTYPE_DPDK_HPING, "Bye.\n");

    return 0;
}
