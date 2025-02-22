#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <stdint.h>
#include <inttypes.h>

#include <rte_eal.h>
#include <rte_common.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_log.h>
#include <rte_mbuf.h>

#include <pcap.h>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>
#include "dpdk_ip.h"
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250

#define RX_RING_SIZE 512
#define TX_RING_SIZE 512
#define BURST_SIZE 32
#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1
#define MAX_PKT_BURST 32
//#define RTE_DEV_NAME_MAX_LEN 512
static struct rte_eth_dev_tx_buffer *tx_buffer =NULL;
static volatile bool force_quit;
bool bRunning = false;
static  struct rte_eth_conf port_conf_default;

pcap_dumper_t *dumper =NULL;
int g_nCapPort = 0;
int lcore_main(void *arg);
void dumpFile(const u_char *pkt, int len, time_t tv_sec, suseconds_t tv_usec);

void openFile(const char *fname)
{
    dumper = pcap_dump_open(pcap_open_dead(DLT_EN10MB, 1600), fname);
    if (NULL == dumper)
    {
        printf("dumper is NULL\n");
        return;
    }
}

void dumpFile(const u_char *pkt, int len, time_t tv_sec, suseconds_t tv_usec)
{
    struct pcap_pkthdr hdr;
    hdr.ts.tv_sec = tv_sec;
    hdr.ts.tv_usec = tv_usec;
    hdr.caplen = len;
    hdr.len = len; 

    pcap_dump((u_char*)dumper, &hdr, pkt); 
}

// 输出设备的mac地址
static void print_mac(unsigned int port_id)
{
    struct rte_ether_addr dev_eth_addr;
    rte_eth_macaddr_get(port_id, &dev_eth_addr);
    printf("port id %d MAC address: %02X:%02X:%02X:%02X:%02X:%02X\n\n",
        (unsigned int) port_id,
        dev_eth_addr.addr_bytes[0],
        dev_eth_addr.addr_bytes[1],
        dev_eth_addr.addr_bytes[2],
        dev_eth_addr.addr_bytes[3],
        dev_eth_addr.addr_bytes[4],
        dev_eth_addr.addr_bytes[5]);
}

// 在程序运行结束后统计收发包信息
static void print_stats(void)
{
    struct rte_eth_stats stats;
    printf("\nStatistics for port %u\n", g_nCapPort);
    rte_eth_stats_get(g_nCapPort, &stats);
    printf("Rx:%9"PRIu64" Tx:%9"PRIu64" dropped:%9"PRIu64"\n", stats.ipackets, stats.opackets, stats.imissed);
}

// 处理中断信号，统计信息并终止程序
static void signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM) {
        printf("\n\nSignal %d received, preparing to exit...\n",
            signum);
        force_quit = true;
        print_stats();

        if (!bRunning)
        {
            exit(0);
        }
    }
}
#if 1
#define RTE_TEST_RX_DESC_DEFAULT 1024
#define RTE_TEST_TX_DESC_DEFAULT 4
static uint16_t nb_rxd = RTE_TEST_RX_DESC_DEFAULT;
static uint16_t nb_txd = RTE_TEST_TX_DESC_DEFAULT;
static uint64_t nb_pkts = 100;
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
static inline int port_init(uint8_t portid, struct rte_mempool *mbuf_pool)
{
    int ret;
    struct rte_eth_rxconf rxq_conf;
    struct rte_eth_txconf txq_conf;
    struct rte_eth_conf local_port_conf = port_conf;
    struct rte_eth_dev_info dev_info;
    /* init port */
    rte_eth_dev_info_get(portid, &dev_info);
    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
        local_port_conf.txmode.offloads |=
            DEV_TX_OFFLOAD_MBUF_FAST_FREE;

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
                                 mbuf_pool);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup:err=%d, port=%u\n",
                 ret, portid);

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

    /* Start device */
    ret = rte_eth_dev_start(portid);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start:err=%d, port=%u\n",
                 ret, portid);

}
#else
// 初始化设备端口，配置收发队列
static inline int port_init(uint8_t port, struct rte_mempool *mbuf_pool)
{
    uint8_t nb_ports = rte_eth_dev_count_avail();
    if (port < 0 || port >= nb_ports)
    {
        printf("port is not right \n");
        return -1;
    }

    port_conf_default.rxmode.max_rx_pkt_len = RTE_ETHER_MAX_LEN;
    struct rte_eth_conf port_conf = port_conf_default;

    struct rte_eth_dev_info dev_info;
    struct rte_eth_txconf txq_conf;
    rte_eth_dev_info_get(port, &dev_info);
    if (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MBUF_FAST_FREE)
    {
        port_conf.txmode.offloads |= DEV_TX_OFFLOAD_MBUF_FAST_FREE;
    }
    
    const uint16_t nb_rx_queues = 1;
    const uint16_t nb_tx_queues = 1;
    int ret;
    uint16_t q;

    // 配置设备
    ret = rte_eth_dev_configure(port, nb_rx_queues, nb_tx_queues, &port_conf);
    if (ret != 0)
    {
        printf("rte_eth_dev_configure failed \n");
        return ret;
    }

    uint16_t nb_rxd = RX_RING_SIZE;
    uint16_t nb_txd = TX_RING_SIZE;
    rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);

    // 配置收包队列
    for (q = 0; q < nb_rx_queues; q++) 
    {
        ret= rte_eth_rx_queue_setup(port, q, RX_RING_SIZE, rte_eth_dev_socket_id(port),NULL, mbuf_pool);
        if (ret < 0)
        {
            printf("rte_eth_rx_queue_setup failed \n");
            return ret;
        }
    }

    txq_conf = dev_info.default_txconf;
    txq_conf.offloads = port_conf.txmode.offloads;
    // 配置发包队列
    for (q = 0; q < nb_tx_queues; q++) 
    {
        ret= rte_eth_tx_queue_setup(port, q, TX_RING_SIZE, rte_eth_dev_socket_id(port), &txq_conf);
        if (ret < 0)
        {
            printf("rte_eth_tx_queue_setup failed \n");
            return ret;
        }
    }
        /* Initialize TX buffers */
    tx_buffer = rte_zmalloc_socket("tx_buffer",
                                   RTE_ETH_TX_BUFFER_SIZE(MAX_PKT_BURST), 0,
                                   rte_eth_dev_socket_id(port));
    if (tx_buffer == NULL)
        rte_exit(EXIT_FAILURE, "Cannot allocate buffer for tx on port %u\n",
                 port);

    rte_eth_tx_buffer_init(tx_buffer, MAX_PKT_BURST);
    // 启动设备
    ret = rte_eth_dev_start(port);
    if (ret < 0)
    {
        printf("rte_eth_dev_start failed \n");
        return ret;
    }

    // 开启混杂模式
    rte_eth_promiscuous_enable(port);

    return 0;
}
#endif
int lcore_main(void *arg)
{

    int i;
    unsigned int lcore_id = rte_lcore_id();
    //创建pcap文件
    if (NULL == dumper)
    {
        openFile("test.pcap");
    }

    RTE_LOG(INFO, APP, "lcore %u running\n", lcore_id);

    while (!force_quit) 
    {
        bRunning = true;

        struct rte_mbuf *bufs[BURST_SIZE];
        uint16_t nb_rx;
        uint16_t nb_tx;
        uint16_t buf;

        // 接受数据包
        nb_rx = rte_eth_rx_burst(g_nCapPort, 0,
            bufs, BURST_SIZE);

        if (unlikely(nb_rx == 0))
            continue;

        for (i= 0; i < nb_rx; i++)
        {
            struct timeval tv;
            gettimeofday(&tv, NULL);

            char *pktbuf = rte_pktmbuf_mtod(bufs[i], char *);
            
            dumpFile((const u_char*)pktbuf, bufs[i]->data_len, tv.tv_sec, tv.tv_usec);
            dpdk_ip_proto_process_raw(bufs[i]);
        }
    }

    RTE_LOG(INFO, APP, "lcore %u exiting\n", lcore_id);
    return 0;
}


int main(int argc, char *argv[])
{
    int ret;
    int i; 
    struct rte_mempool *mbuf_pool;

    // 初始化DPDK
    ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "EAL Init failed\n");

    argc -= ret;
    argv += ret;

    // 注册中断信号处理函数
    force_quit = false;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("\n\n");
    uint8_t nb_ports = rte_eth_dev_count_avail();
    for (i = 0; i < nb_ports; i++) 
    {
        char dev_name[RTE_DEV_NAME_MAX_LEN];
        rte_eth_dev_get_name_by_port(i, dev_name);
        printf("number %d:  %s  ", i, dev_name);
        print_mac(i);
    }

    printf("choose a port, enter the port number: \n");
    scanf("%d",&g_nCapPort);

    // 申请mbuf内存池
    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL",
        NUM_MBUFS * nb_ports,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id());

    if (mbuf_pool == NULL)
    {
        rte_exit(EXIT_FAILURE, "mbuf_pool create failed\n");
    }

    if (port_init(g_nCapPort, mbuf_pool) != 0)
    {
        rte_exit(EXIT_FAILURE, "port init failed\n");
    }

    // 线程核心绑定，循环处理数据包
    rte_eal_mp_remote_launch(lcore_main, NULL, SKIP_MASTER);
    rte_eal_mp_wait_lcore();

    exit(0);
}
