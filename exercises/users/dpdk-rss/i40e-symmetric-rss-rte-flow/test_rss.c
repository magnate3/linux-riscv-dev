//
// Created by luks on 23.10.21.
//

#include <signal.h>
#include <unistd.h>

#include <rte_ethdev.h>
#include <rte_malloc.h>
#include <rte_hash.h>
#include <rte_ip.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_launch.h>
#include <rte_lcore.h>
#include <rte_log.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_flow.h>



#define BURST_SIZE   32
struct rte_fdir_conf fdir_conf = {
        .mode = RTE_FDIR_MODE_NONE,
        .pballoc = RTE_FDIR_PBALLOC_64K,
        .status = RTE_FDIR_REPORT_STATUS,
        .mask = {
                .vlan_tci_mask = 0xFFEF,
                .ipv4_mask     = {
                        .src_ip = 0xFFFFFFFF,
                        .dst_ip = 0xFFFFFFFF,
                },
                .ipv6_mask     = {
                        .src_ip = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
                        .dst_ip = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF},
                },
                .src_port_mask = 0xFFFF,
                .dst_port_mask = 0xFFFF,
                .mac_addr_byte_mask = 0xFF,
                .tunnel_type_mask = 1,
                .tunnel_id_mask = 0xFFFFFFFF,
        },
        .drop_queue = 127,
};

#define MTU_SIZE             3000
#define NB_RX_DESC           8192
#define NB_TS_DESC           8192
#define MEMPOOL_SIZE         262143
#define MEMPOOL_CACHE_SIZE   500
#define MULTICAST_ENABLE     1
#define PROMISC_ENABLE       1

#define I40E_RSS_HKEY_LEN 52

static volatile int g_should_stop = 0;

static void handle_sig(int sig)
{
    switch (sig) {
        case SIGINT:
        case SIGTERM:
            g_should_stop = 1;
            break;
    }
}

static void handle_broken_pmd(int sig)
{
#define HINT "segfault, try --no-pci or start as root\n"

    if (sig != SIGSEGV)
        return;

    write(STDERR_FILENO, HINT, sizeof(HINT) - 1);
    signal(sig, SIG_DFL);
    kill(getpid(), sig);
}

static int should_stop(void)
{
    return g_should_stop;
}

struct context {
    int socket_id;
    uint16_t nb_queues;
    const char *port1_name;
    uint16_t port1_id;
    uint16_t nb_rx_desc;
    uint16_t nb_tx_desc;
    uint32_t mempool_size;
    uint32_t mempool_cache_size;
    struct rte_mempool **port1_pkt_mempool_array; // toto  potom treba prerobit na pole poli poli
    rte_atomic16_t queue_id; // from zero to lcore count
};

static void thread_loop(uint16_t port_id, uint16_t queue_id)
{
    int retval;
    struct rte_mbuf *pkts[BURST_SIZE] = { NULL };
    uint16_t rx_count;
    uint64_t tot_rx_count = 0;

    while (!should_stop()) {
        rx_count = rte_eth_rx_burst(port_id, queue_id, pkts, BURST_SIZE);
        if (rx_count > 0) {
            tot_rx_count += rx_count;
            printf("Received %d packet/s on lcore %d\n", rx_count, rte_lcore_id());
            for (int i = 0; i < rx_count; i++)
                rte_pktmbuf_free(pkts[i]);
        }
    }

    printf("Lcoreid %d total rx count: %lu\n", rte_lcore_id(), tot_rx_count);
}

static int thread_main(void *arg)
{
    int switch_ifaces = 0;
    struct context *ctx = arg;
    int qid = rte_atomic16_add_return(&ctx->queue_id, 1);

    thread_loop(ctx->port1_id, qid);
    printf("Thread %d finished\n", rte_lcore_id());
    return 0;
}

static int device_init_port_conf(const struct context *ctx, const struct rte_eth_dev_info *dev_info,
                                 struct rte_eth_conf *port_conf)
{
    *port_conf = (struct rte_eth_conf){
            .rxmode = {
                    .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
                    .offloads = DEV_RX_OFFLOAD_CHECKSUM,
            },
            .txmode = {
            },
    };

    // offload checksum validation
    port_conf->rxmode.offloads |= DEV_RX_OFFLOAD_CHECKSUM;

    port_conf->fdir_conf = fdir_conf;
    return 0;
}

int device_configure_queues(struct context *ctx, const struct rte_eth_dev_info *dev_info,
                            const struct rte_eth_conf *port_conf)
{
    int retval;
    struct rte_eth_rxconf rxq_conf;
    struct rte_eth_txconf txq_conf;

    ctx->port1_pkt_mempool_array =
            rte_calloc("mempool_array", ctx->nb_queues, sizeof(struct rte_mempool *), 0);
    if (unlikely(ctx->port1_pkt_mempool_array == NULL)) {
        fprintf(stderr, "Could not allocate memory for packet mempool pointers\n");
        return -ENOMEM;
    }

    for (uint16_t queue_id = 0; queue_id < ctx->nb_queues; queue_id++) {
        char mempool_name[64];
        snprintf(mempool_name, 64, "pktmbuf_pool_p%d_q%d", ctx->port1_id, queue_id);
        printf("Creating a packet mbuf pool %s of size %d, cache size %d, mbuf size %d\n",
               mempool_name, ctx->mempool_size, ctx->mempool_cache_size,
               RTE_MBUF_DEFAULT_BUF_SIZE);

        ctx->port1_pkt_mempool_array[queue_id] = rte_pktmbuf_pool_create(mempool_name, ctx->mempool_size,
                                                                         ctx->mempool_cache_size, 0, RTE_MBUF_DEFAULT_BUF_SIZE, (int)rte_socket_id());
        if (ctx->port1_pkt_mempool_array[queue_id] == NULL) {
            retval = -rte_errno;
            return retval;
        }

        rxq_conf = dev_info->default_rxconf;
        rxq_conf.offloads = port_conf->rxmode.offloads;
        rxq_conf.rx_thresh.hthresh = 0;
        rxq_conf.rx_thresh.pthresh = 0;
        rxq_conf.rx_thresh.wthresh = 0;
        rxq_conf.rx_free_thresh = 0;
        rxq_conf.rx_drop_en = 0;

        printf("Creating Q %d of P %d using desc RX: %d TX: %d RX htresh: %d RX pthresh %d wtresh "
               "%d free_tresh %d drop_en %d Offloads %lu\n",
               queue_id, ctx->port1_id, ctx->nb_rx_desc, ctx->nb_tx_desc,
               rxq_conf.rx_thresh.hthresh, rxq_conf.rx_thresh.pthresh, rxq_conf.rx_thresh.wthresh,
               rxq_conf.rx_free_thresh, rxq_conf.rx_drop_en, rxq_conf.offloads);

        retval = rte_eth_rx_queue_setup(ctx->port1_id, queue_id, ctx->nb_rx_desc, ctx->socket_id,
                                        &rxq_conf, ctx->port1_pkt_mempool_array[queue_id]);
        if (retval < 0) {
            fprintf(stderr, "Error (err=%d) during initialization of device queue %u of port %u\n",
                    retval, queue_id, ctx->port1_id);
            return retval;
        }
    }

    for (uint16_t queue_id = 0; queue_id < ctx->nb_queues; queue_id++) {
        txq_conf = dev_info->default_txconf;
        txq_conf.offloads = port_conf->txmode.offloads;
        retval = rte_eth_tx_queue_setup(
                ctx->port1_id, queue_id, ctx->nb_tx_desc, ctx->socket_id, &txq_conf);
        if (retval < 0) {
            fprintf(stderr, "Error (err=%d) during initialization of device queue %u of port %u\n",
                    retval, queue_id, ctx->port1_id);
            return retval;
        }
    }

    return 0;
}

static int device_configure(struct context *ctx)
{
    // configure device
    int retval;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_conf port_conf;

    retval = rte_eth_dev_get_port_by_name(ctx->port1_name, &(ctx->port1_id));
    if (retval < 0) {
        fprintf(stderr, "Error (err=%d) when getting port id of %s Is device enabled?\n", retval,
                ctx->port1_name);
        return retval;
    }

    if (!rte_eth_dev_is_valid_port(ctx->port1_id)) {
        fprintf(stderr, "Specified port %d is invalid\n", ctx->port1_id);
        return retval;
    }

    retval = rte_eth_dev_socket_id(ctx->port1_id);
    if (retval < 0) {
        fprintf(stderr, "Error (err=%d) invalid socket id (port %u)\n", retval, ctx->port1_id);
        return retval;
    } else {
        ctx->socket_id = retval;
    }

    retval = rte_eth_dev_info_get(ctx->port1_id, &dev_info);
    if (retval != 0) {
        fprintf(stderr, "Error (err=%d) during getting device info (port %u)\n", retval,
                ctx->port1_id);
        return retval;
    }

    retval = device_init_port_conf(ctx, &dev_info, &port_conf);
    if (retval != 0) {
        fprintf(stderr, "Error (err=%d) during port init (port %u)\n", retval,
                ctx->port1_id);
        return retval;
    }

    retval = rte_eth_dev_configure(ctx->port1_id, ctx->nb_queues, ctx->nb_queues, &port_conf);
    if (retval != 0) {
        fprintf(stderr, "Error (err=%d) during configuring the device (port %u)\n", retval,
                ctx->port1_id);
        return retval;
    }

    retval = rte_eth_dev_adjust_nb_rx_tx_desc(ctx->port1_id, &ctx->nb_rx_desc, &ctx->nb_tx_desc);
    if (retval != 0) {
        fprintf(stderr, "Error (err=%d) during adjustment of device queues descriptors (port %u)\n",
                retval, ctx->port1_id);
        return retval;
    }

    retval = MULTICAST_ENABLE ? rte_eth_allmulticast_enable(ctx->port1_id)
                              : rte_eth_allmulticast_disable(ctx->port1_id);
    if (retval < 0) {
        fprintf(stderr, "Error (err=%d) when en/disabling multicast on port %s\n", retval,
                ctx->port1_name);
        return retval;
    }

    retval = PROMISC_ENABLE ? rte_eth_promiscuous_enable(ctx->port1_id)
                            : rte_eth_promiscuous_disable(ctx->port1_id);
    if (retval < 0) {
        fprintf(stderr, "Error (err=%d) when enabling promiscuous mode on port %u\n", retval,
                ctx->port1_id);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    retval = rte_eth_dev_set_mtu(ctx->port1_id, MTU_SIZE);
    if (retval < 0) {
        fprintf(stderr, "Error (err=%d) when setting MTU to %u on port %u\n", retval, MTU_SIZE,
                ctx->port1_id);
        return retval;
    }


    retval = device_configure_queues(ctx, &dev_info, &port_conf);
    if (retval < 0) {
        fprintf(stderr, "Error (err=%d) when configuring queues on port %u\n", retval, ctx->port1_id);
        return retval;
    }

    return 0;
}

static void device_rss_configure(int portid, int nb_queues) {
    int retval;
    struct rte_flow_item pattern[] = { { 0 }, { 0 }, { 0 }, { 0 } };
    struct rte_flow_action action[] = { { 0 }, { 0 } };
    struct rte_flow_action_rss rss_action_conf = { 0 };
    struct rte_flow_attr attr = { 0 };
    struct rte_flow_error create_error = { 0 };
    struct rte_flow_error valid_error = { 0 };
    struct rte_flow_error flush_error = { 0 };
    uint16_t queues[RTE_MAX_QUEUES_PER_PORT];
    uint8_t rss_hkey[I40E_RSS_HKEY_LEN];
    struct rte_eth_rss_conf rss_conf = {
            .rss_key = rss_hkey,
            .rss_key_len = I40E_RSS_HKEY_LEN,
    };
    struct rte_flow *flow;

    retval = rte_eth_dev_rss_hash_conf_get(portid, &rss_conf);
    if (retval != 0) {
        fprintf(stderr, "FATALERROR!!! HASH CONF NO SUCCESS\n");
        exit(1);
    }

    for (uint16_t i = 0; i < nb_queues; ++i) {
        queues[i] = i;
    }

    rss_action_conf.func = RTE_ETH_HASH_FUNCTION_DEFAULT;
    rss_action_conf.level = 0;
    rss_action_conf.types = 0; // queues region can not be configured with types
    rss_action_conf.key = rss_conf.rss_key;
    rss_action_conf.key_len = rss_conf.rss_key_len;
    rss_action_conf.queue_num = nb_queues;
    rss_action_conf.queue = queues;

    attr.ingress = 1;
    pattern[0].type = RTE_FLOW_ITEM_TYPE_END;

    action[0].type = RTE_FLOW_ACTION_TYPE_RSS;
    action[0].conf = &rss_action_conf;

    action[1].type = RTE_FLOW_ACTION_TYPE_END;
    flow = rte_flow_create(portid, &attr, pattern, action, &create_error);
    if (flow == NULL) {
        fprintf(stderr, "ERROR Create errror: %s\n", create_error.message);
        int ret = rte_flow_validate(portid, &attr, pattern, action, &valid_error);
        fprintf(stderr, "FATALERROR!!! Err on flow validation: %s \n errmsg: %s\n",
                rte_strerror(-ret), valid_error.message);
        exit(1);
    } else {
        printf("RULE1 created\n");
    }

    memset(pattern, 0, sizeof(pattern));
    memset(action, 0, sizeof(action));
    memset(&rss_action_conf, 0, sizeof(rss_action_conf));


    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_END;

    rss_action_conf.func = RTE_ETH_HASH_FUNCTION_SYMMETRIC_TOEPLITZ;
    rss_action_conf.level = 0;
    rss_action_conf.types = ETH_RSS_NONFRAG_IPV4_OTHER;
    rss_action_conf.key_len = rss_conf.rss_key_len;
    rss_action_conf.key = rss_conf.rss_key;
    rss_action_conf.queue_num = 0;
    rss_action_conf.queue = NULL;

    action[0].type = RTE_FLOW_ACTION_TYPE_RSS;
    action[0].conf = &rss_action_conf;
    action[1].type = RTE_FLOW_ACTION_TYPE_END;

    flow = rte_flow_create(portid, &attr, pattern, action, &create_error);
    if (flow == NULL) {
        fprintf(stderr, "ERROR Create errror: %s\n", create_error.message);
        int ret = rte_flow_validate(portid, &attr, pattern, action, &valid_error);
        fprintf(stderr, "FATALERROR!!! Err on flow validation: %s \n errmsg: %s\n",
                rte_strerror(-ret), valid_error.message);
        exit(1);
    } else {
        printf("RULE2 created\n");
    }


    memset(pattern, 0, sizeof(pattern));
    memset(action, 0, sizeof(action));
    memset(&rss_action_conf, 0, sizeof(rss_action_conf));

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_TCP;
    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

    rss_action_conf.func = RTE_ETH_HASH_FUNCTION_SYMMETRIC_TOEPLITZ;
    rss_action_conf.level = 0;
    rss_action_conf.types = ETH_RSS_NONFRAG_IPV4_TCP;
    rss_action_conf.key_len = rss_conf.rss_key_len;
    rss_action_conf.key = rss_conf.rss_key;
    rss_action_conf.queue_num = 0;
    rss_action_conf.queue = NULL;

    action[0].type = RTE_FLOW_ACTION_TYPE_RSS;
    action[0].conf = &rss_action_conf;
    action[1].type = RTE_FLOW_ACTION_TYPE_END;

    flow = rte_flow_create(portid, &attr, pattern, action, &create_error);
    if (flow == NULL) {
        fprintf(stderr, "ERROR Create errror: %s\n", create_error.message);
        int ret = rte_flow_validate(portid, &attr, pattern, action, &valid_error);
        fprintf(stderr, "FATALERROR!!! Err on flow validation: %s \n errmsg: %s\n",
                rte_strerror(-ret), valid_error.message);
        exit(1);
    } else {
        printf("RULE3 created\n");
    }


    memset(pattern, 0, sizeof(pattern));
    memset(action, 0, sizeof(action));
    memset(&rss_action_conf, 0, sizeof(rss_action_conf));

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;
    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

    rss_action_conf.func = RTE_ETH_HASH_FUNCTION_SYMMETRIC_TOEPLITZ;
    rss_action_conf.level = 0;
    rss_action_conf.types = ETH_RSS_NONFRAG_IPV4_UDP;
    rss_action_conf.key_len = rss_conf.rss_key_len;
    rss_action_conf.key = rss_conf.rss_key;
    rss_action_conf.queue_num = 0;
    rss_action_conf.queue = NULL;

    action[0].type = RTE_FLOW_ACTION_TYPE_RSS;
    action[0].conf = &rss_action_conf;
    action[1].type = RTE_FLOW_ACTION_TYPE_END;

    flow = rte_flow_create(portid, &attr, pattern, action, &create_error);
    if (flow == NULL) {
        fprintf(stderr, "ERROR Create errror: %s\n", create_error.message);
        int ret = rte_flow_validate(portid, &attr, pattern, action, &valid_error);
        fprintf(stderr, "FATALERROR!!! Err on flow validation: %s \n errmsg: %s\n",
                rte_strerror(-ret), valid_error.message);
        exit(1);
    } else {
        printf("RULE4 created\n");
    }
}

int main(int argc, char *argv[])
{
    struct context ctx;
    char *iface1 = NULL;
    char *iface2 = NULL;
    int proc_type;
    int args;
    int retval;

    rte_log_set_global_level(RTE_LOG_WARNING);

    signal(SIGSEGV, &handle_broken_pmd);

    args = rte_eal_init(argc, argv);
    if (args < 0) {
        fprintf(stderr, "rte_eal_init() has failed: %d\n", args);
        return EXIT_FAILURE;
    }

    signal(SIGSEGV, SIG_DFL);

    argc -= args;
    argv += args;

    if (argc <= 1) {
        fprintf(stderr, "Use: %s PCIe_address#1 [PCIe_address#2]\n", argv[0]);
        fprintf(stderr, "The second NIC then enables the IPS mode\n");
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    iface1 = argv[1];
    ctx.port1_name = iface1;
    ctx.socket_id = (int)rte_socket_id();

    ctx.nb_queues = rte_lcore_count();
    ctx.nb_rx_desc = NB_RX_DESC;
    ctx.nb_tx_desc = NB_TS_DESC;
    ctx.mempool_size = MEMPOOL_SIZE;
    ctx.mempool_cache_size = MEMPOOL_CACHE_SIZE;
    rte_atomic16_init(&ctx.queue_id);
    rte_atomic16_set(&ctx.queue_id, -1);

    retval = device_configure(&ctx);
    if (retval < 0) {
        fprintf(stderr, "device_configure(): unable to configure the device, %d\n", retval);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    proc_type = rte_eal_process_type();
    if (proc_type != RTE_PROC_PRIMARY) {
        fprintf(stderr, "invalid process type %d, primary required\n", proc_type);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    retval = rte_eth_dev_start(ctx.port1_id);
    if (retval < 0) {
        fprintf(stderr, "Error (err=%d) during device startup of port %u", retval, ctx.port1_id);
        rte_eal_cleanup();
        return EXIT_FAILURE;
    }

    // Configure the RSS via RTE_FLOW
    device_rss_configure(ctx.port1_id, ctx.nb_queues);

    printf("\nStarting all cores ... [Ctrl+C to quit]\n\n\n");

    signal(SIGINT, &handle_sig);
    signal(SIGTERM, &handle_sig);

    rte_eal_mp_remote_launch(thread_main, &ctx, CALL_MASTER);
    rte_eal_mp_wait_lcore();

    struct rte_flow_error flush_error = { 0 };
    retval = rte_flow_flush(ctx.port1_id, &flush_error);
    if (retval != 0) {
        fprintf(stderr, "FATALERROR!!! Err on flow flush2: %s \n errmsg: %s\n",
                   rte_strerror(-retval), flush_error.message);
    }

    rte_eal_cleanup();

    return EXIT_SUCCESS;
}
