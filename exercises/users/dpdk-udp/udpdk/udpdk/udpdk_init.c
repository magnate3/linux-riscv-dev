//
// Created by leoll2 on 9/25/20.
// Copyright (c) 2020 Leonardo Lai. All rights reserved.
//

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <rte_common.h>
#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_launch.h>
#include <rte_lcore.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memzone.h>

#include "udpdk_list.h"
#include "udpdk_api.h"
#include "udpdk_args.h"
#include "udpdk_constants.h"
#include "udpdk_bind_table.h"
#include "udpdk_monitor.h"
#include "udpdk_poller.h"
#include "udpdk_sync.h"
#include "udpdk_types.h"

#define RTE_LOGTYPE_INIT RTE_LOGTYPE_USER1
#define RTE_LOGTYPE_CLOSE RTE_LOGTYPE_USER1
#define RTE_LOGTYPE_INTR RTE_LOGTYPE_USER1

extern int interrupted;
extern struct exch_zone_info *exch_zone_desc;
extern struct exch_slot *exch_slots;
extern struct rte_mempool *rx_pktmbuf_pool;
extern struct rte_mempool *tx_pktmbuf_pool;
extern struct rte_mempool *tx_pktmbuf_direct_pool;
extern struct rte_mempool *tx_pktmbuf_indirect_pool;
extern udpdk_list_t **sock_bind_table;
extern int primary_argc;
extern int secondary_argc;
extern char *primary_argv[MAX_ARGC];
extern char *secondary_argv[MAX_ARGC];
extern struct rte_ring *ipc_app_to_pol;
extern struct rte_ring *ipc_pol_to_app;
extern struct rte_mempool *ipc_msg_pool;
static pid_t poller_pid;


/* Get the name of the rings of exchange slots */
static inline const char * get_exch_ring_name(unsigned id, enum exch_ring_func func)
{
    static char buffer[sizeof(EXCH_RX_RING_NAME) + 8];

    if (func == EXCH_RING_RX) {
        snprintf(buffer, sizeof(buffer), EXCH_RX_RING_NAME, id);
    } else {
        snprintf(buffer, sizeof(buffer), EXCH_TX_RING_NAME, id);
    }
    return buffer;
}

/* Initialize a pool of mbuf for reception and transmission */
static int init_mbuf_pools(void)
{
    const unsigned int num_mbufs_rx = NUM_RX_DESC_DEFAULT;
    const unsigned int num_mbufs_tx = NUM_TX_DESC_DEFAULT;  // TODO size properly
    const unsigned int num_mbufs_cache = 2 * MBUF_CACHE_SIZE;
    const unsigned int num_mbufs = num_mbufs_rx + num_mbufs_tx + num_mbufs_cache;
    const int socket = rte_socket_id();

    rx_pktmbuf_pool = rte_pktmbuf_pool_create(PKTMBUF_POOL_RX_NAME, num_mbufs, MBUF_CACHE_SIZE, 0,
            RTE_MBUF_DEFAULT_BUF_SIZE, socket);
    if (rx_pktmbuf_pool == NULL) {
        RTE_LOG(ERR, INIT, "Failed to allocate RX pool: %s\n", rte_strerror(rte_errno));
        return -1;
    }

    tx_pktmbuf_pool = rte_pktmbuf_pool_create(PKTMBUF_POOL_TX_NAME, num_mbufs, MBUF_CACHE_SIZE, 0,
            RTE_MBUF_DEFAULT_BUF_SIZE, socket);  // used by the app (sendto) // TODO size properly
    if (tx_pktmbuf_pool == NULL) {
        RTE_LOG(ERR, INIT, "Failed to allocate TX pool: %s\n", rte_strerror(rte_errno));
        return -1;
    }

    tx_pktmbuf_direct_pool = rte_pktmbuf_pool_create(PKTMBUF_POOL_DIRECT_TX_NAME, num_mbufs, MBUF_CACHE_SIZE, 0,
            RTE_MBUF_DEFAULT_BUF_SIZE, socket);  // used by the poller       // TODO size properly
    if (tx_pktmbuf_direct_pool == NULL) {
        RTE_LOG(ERR, INIT, "Failed to allocate TX direct pool: %s\n", rte_strerror(rte_errno));
        return -1;
    }

    tx_pktmbuf_indirect_pool = rte_pktmbuf_pool_create(PKTMBUF_POOL_INDIRECT_TX_NAME, num_mbufs, MBUF_CACHE_SIZE, 0,
            RTE_MBUF_DEFAULT_BUF_SIZE, socket);  // used by the poller      // TODO size properly
    if (tx_pktmbuf_indirect_pool == NULL) {
        RTE_LOG(ERR, INIT, "Failed to allocate TX indirect pool: %s\n", rte_strerror(rte_errno));
        return -1;
    }

    return 0;
}

/* Initialize a DPDK port */
static int init_port(uint16_t port_num)
{
    struct rte_eth_dev_info dev_info;
    // TODO add RSS support
    const uint16_t rx_rings = 1;
    const uint16_t tx_rings = 1;
    uint16_t rx_ring_size = NUM_RX_DESC_DEFAULT;
    uint16_t tx_ring_size = NUM_TX_DESC_DEFAULT;
    uint16_t q;
    int retval;

    // Check port validity
    if (!rte_eth_dev_is_valid_port(port_num)) {
        RTE_LOG(ERR, INIT, "Port %d is invalid (out of range or not attached)\n", port_num);
        return -1;
    }

    // Retrieve port info
    retval = rte_eth_dev_info_get(port_num, &dev_info);
    if (retval != 0) {
        RTE_LOG(ERR, INIT, "Error during getting device (port %u) info: %s\n",
                port_num, strerror(-retval));
        return retval;
    }

    const struct rte_eth_conf port_conf = {
        .rxmode = {
            .mq_mode = ETH_MQ_RX_RSS,
            .max_rx_pkt_len = RTE_MIN(JUMBO_FRAME_MAX_SIZE, dev_info.max_rx_pktlen),
            .split_hdr_size = 0,
            .offloads = (DEV_RX_OFFLOAD_CHECKSUM |
                         DEV_RX_OFFLOAD_SCATTER |
                         DEV_RX_OFFLOAD_JUMBO_FRAME),
        },
        .txmode = {
            .offloads = DEV_TX_OFFLOAD_MULTI_SEGS,
        }
    };

    // Configure mode and number of rings
    retval = rte_eth_dev_configure(port_num, rx_rings, tx_rings, &port_conf);
    if (retval != 0) {
        RTE_LOG(ERR, INIT, "Could not configure port %d\n", port_num);
        return retval;
    }

    // Adjust the number of descriptors
    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port_num, &rx_ring_size, &tx_ring_size);
    if (retval != 0) {
        RTE_LOG(ERR, INIT, "Could not adjust rx/tx descriptors on port %d\n", port_num);
        return retval;
    }

    // Setup the RX queues
    for (q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port_num, q, rx_ring_size,
                rte_eth_dev_socket_id(port_num), NULL, rx_pktmbuf_pool);
        if (retval < 0) {
            RTE_LOG(ERR, INIT, "Could not setup RX queue %d on port %d\n", q, port_num);
            return retval;
        }
    }

    // Setup the TX queues
    for (q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port_num, q, tx_ring_size,
                rte_eth_dev_socket_id(port_num), NULL); // no particular configuration needed
        if (retval < 0) {
            RTE_LOG(ERR, INIT, "Could not setup TX queue %d on port %d\n", q, port_num);
            return retval;
        }
    }

    // Enable promiscuous mode
    retval = rte_eth_promiscuous_enable(port_num);
    if (retval < 0) {
        RTE_LOG(ERR, INIT, "Could not set port %d to promiscous mode\n", port_num);
        return retval;
    }

    // Start the DPDK port
    retval = rte_eth_dev_start(port_num);
    if (retval < 0) {
        RTE_LOG(ERR, INIT, "Could not start port %d\n", port_num);
        return retval;
    }

    RTE_LOG(INFO, INIT, "Initialized port %d.\n", port_num);
    return 0;
}

/* Initialize a shared memory region to contain descriptors for the exchange slots */
static int init_exch_memzone(void)
{
    const struct rte_memzone *mz;

    mz = rte_memzone_reserve(EXCH_MEMZONE_NAME, sizeof(*exch_zone_desc), rte_socket_id(), 0);
    if (mz == NULL) {
        RTE_LOG(ERR, INIT, "Cannot allocate shared memory for exchange slot descriptors\n");
        return -1;
    }
    memset(mz->addr, 0, sizeof(*exch_zone_desc));
    exch_zone_desc = mz->addr;

    return 0;
}

static int destroy_exch_memzone(void)
{
    const struct rte_memzone *mz;

    mz = rte_memzone_lookup(EXCH_MEMZONE_NAME);
    return rte_memzone_free(mz);
}

/* Initialize a shared memory region to store the L4 switching table */
static int init_udp_bind_table(void)
{
    const struct rte_memzone *mz;

    mz = rte_memzone_reserve(UDP_BIND_TABLE_NAME, UDP_MAX_PORT * sizeof(struct udpdk_list_t *), rte_socket_id(), 0);
    if (mz == NULL) {
        RTE_LOG(ERR, INIT, "Cannot allocate shared memory for L4 switching table\n");
        return -1;
    }
    sock_bind_table = mz->addr;
    btable_init();
    return 0;
}

/* Destroy table for UDP port switching */
static int destroy_udp_bind_table(void)
{
    const struct rte_memzone *mz;

    btable_destroy();

    mz = rte_memzone_lookup(UDP_BIND_TABLE_NAME);
    return rte_memzone_free(mz);
}

/* Initialize (statically) the slots to exchange packets between the application and the poller */
static int init_exchange_slots(void)
{
    unsigned i;
    unsigned socket_id;
    const char *q_name;

    socket_id = rte_socket_id();

    // Allocate enough memory to store the exchange slots
    exch_slots = rte_malloc(EXCH_SLOTS_NAME, sizeof(*exch_slots) * NUM_SOCKETS_MAX, 0);
    if (exch_slots == NULL) {
        RTE_LOG(ERR, INIT, "Cannot allocate memory for exchange slots\n");
        return -1;
    }

    // Create a rte_ring for each RX and TX slot
    for (i = 0; i < NUM_SOCKETS_MAX; i++) {
        q_name = get_exch_ring_name(i, EXCH_RING_RX);
        exch_slots[i].rx_q = rte_ring_create(q_name, EXCH_RING_SIZE, socket_id, RING_F_SP_ENQ | RING_F_SC_DEQ);
        q_name = get_exch_ring_name(i, EXCH_RING_TX);
        exch_slots[i].tx_q = rte_ring_create(q_name, EXCH_RING_SIZE, socket_id, RING_F_SP_ENQ | RING_F_SC_DEQ);
        if (exch_slots[i].rx_q == NULL || exch_slots[i].tx_q == NULL) {
            RTE_LOG(ERR, INIT, "Cannot create exchange RX/TX exchange rings (index %d)\n", i);
            return -1;
        }
    }
    return 0;
}
#define PRIM_SENC_SYNC 1
/* Initialize UDPDK */
int udpdk_init(int argc, char *argv[])
{
    int retval;
#if PRIM_SENC_SYNC
    int fd[2];
    pipe(fd);
#endif
    // Parse and initialize the arguments
    if (udpdk_parse_args(argc, argv) < 0) {  // initializes primary and secondary argc argv
        RTE_LOG(ERR, INIT, "Invalid arguments for UDPDK\n");
        
        //return -1;
        goto err1;
    }

    // Start the secondary process
    poller_pid = fork();
    if (poller_pid < 0){
        RTE_LOG(ERR, INIT, "UDPDK fork process fail\n");
        //return -1;
        goto err1;
    }
    if (poller_pid != 0) {  // parent -> application
#if PRIM_SENC_SYNC
        close(fd[0]);
#endif
        // Initialize EAL (returns how many arguments it consumed)
        if (rte_eal_init(primary_argc, (char **)primary_argv) < 0) {
            RTE_LOG(ERR, INIT, "Cannot initialize EAL\n");
            return -1;
        }

        // Initialize the list allocators
        udpdk_list_init();

        // Initialize pools of mbuf
        retval = init_mbuf_pools();
        if (retval < 0) {
            RTE_LOG(ERR, INIT, "Cannot initialize pools of mbufs\n");
            return -1;
        }

        // Initialize DPDK ports
        retval = init_port(PORT_RX);
        if (retval < 0) {
            RTE_LOG(ERR, INIT, "Cannot initialize RX port %d\n", PORT_RX);
            return -1;
        }
        check_port_link_status(PORT_RX);

        if (PORT_TX != PORT_RX) {
            retval = init_port(PORT_TX);
            if (retval < 0) {
                RTE_LOG(ERR, INIT, "Cannot initialize TX port %d\n", PORT_TX);
                return -1;
            }
            check_port_link_status(PORT_TX);
        } else {
            RTE_LOG(INFO, INIT, "Using the same port for RX and TX\n");
        }

        // Initialize IPC channel to synchronize with the poller
        retval = init_ipc_channel();
        if (retval < 0) {
            RTE_LOG(ERR, INIT, "Cannot initialize IPC channel for app-poller synchronization\n");
            return -1;
        }

        // Initialize memzone for exchange
        retval = init_exch_memzone();
        if (retval < 0) {
            RTE_LOG(ERR, INIT, "Cannot initialize memzone for exchange zone descriptors\n");
            return -1;
        }

        retval = init_udp_bind_table();
        if (retval < 0) {
            RTE_LOG(ERR, INIT, "Cannot create table for UDP port switching\n");
            return -1;
        }

        retval = init_exchange_slots();
        if (retval < 0) {
            RTE_LOG(ERR, INIT, "Cannot initialize exchange slots\n");
            return -1;
        }

        // Let the poller process resume initialization
#if PRIM_SENC_SYNC
        write(fd[1],"hello",5);
#endif
        ipc_notify_to_poller();
        
        // Wait for the poller to be fully initialized
        RTE_LOG(INFO, INIT, "Waiting for the poller to complete its inialization...\n");
        ipc_wait_for_poller();
    } else {  // child -> packet poller
#if PRIM_SENC_SYNC 
        char buf[8]={0};
        int ret = 0;
        close(fd[1]);
        ret = read(fd[0],buf,sizeof(buf));
        if(ret <= 0){
             RTE_LOG(INFO, INIT, "Poller initialization, read pipe failed\n");
             exit(0);
        }
#endif
        if (poller_init(secondary_argc, (char **)secondary_argv) < 0) {
            RTE_LOG(INFO, INIT, " !!! Poller initialization failed !!!!\n");
            return -1;
        }
        poller_body();
      
    }
    // The parent process (application) returns immediately from init; instead, poller doesn't till it dies (or error)
    return 0;
err1:
#if PRIM_SENC_SYNC
    close(fd[0]);
    close(fd[1]);
#endif
    return -1;
}

/* Signal UDPDK poller to stop */
void udpdk_interrupt(int signum)
{
    RTE_LOG(INFO, INTR, "Killing the poller process (%d)...\n", poller_pid);
    interrupted = 1;
}

/* Close all the open sockets */
static void udpdk_close_all_sockets(void)
{
    int s =0;
    for (s = 0; s < NUM_SOCKETS_MAX; s++) {
        if (exch_zone_desc->slots[s].bound) {
            RTE_LOG(INFO, CLOSE, "Closing socket %d that was left open\n", s);
            udpdk_close(s);
        }
    }
}

/* Release all the memory and data structures used by UDPDK */
void udpdk_cleanup(void)
{
    uint16_t port_id;
    pid_t pid;

    // Kill the poller process
    RTE_LOG(INFO, CLOSE, "Killing the poller process (%d)...\n", poller_pid);
    kill(poller_pid, SIGTERM);
    pid = waitpid(poller_pid, NULL, 0);
    if (pid < 0) {
        RTE_LOG(WARNING, CLOSE, "Failed killing the poller process\n");
    } else {
        RTE_LOG(INFO, CLOSE, "...killed!\n");
    }

    // Stop and close DPDK ports
    RTE_ETH_FOREACH_DEV(port_id) {
        rte_eth_dev_stop(port_id);
        rte_eth_dev_close(port_id);
    }

    // Close all open sockets
    udpdk_close_all_sockets();

    // Free the memory of L4 switching table
    destroy_udp_bind_table();
 
    // Free the memory for exch zone
    destroy_exch_memzone();

    // Release linked-list memory allocators
    udpdk_list_deinit();
}
