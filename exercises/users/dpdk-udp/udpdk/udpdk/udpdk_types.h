//
// Created by leoll2 on 9/25/20.
// Copyright (c) 2020 Leonardo Lai. All rights reserved.
//

#ifndef UDPDK_TYPES_H
#define UDPDK_TYPES_H

//#include <rte_common.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_launch.h>
#include <rte_lcore.h>
#include <rte_log.h>
#include <rte_malloc.h>
#include <rte_memory.h>
#include <rte_memzone.h>

#include <netinet/in.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include "udpdk_constants.h"
#include "udpdk_list.h"

enum exch_ring_func {EXCH_RING_RX, EXCH_RING_TX};

/* Descriptor for a binding of a socket to (IP, port) */
struct bind_info {
    int sockfd;         // socket fd of the (addr, port) pair
    struct in_addr ip_addr;     // IPv4 address associated to the socket
    bool reuse_addr;    // SO_REUSEADDR
    bool reuse_port;    // SO_REUSEPORT
    bool closed;        // mark this binding as closed
};

/* Descriptor of a socket (current state and options) */
struct exch_slot_info {
    int used;       // used by an open socket
    int bound;      // used by a socket that did 'bind'
    int sockfd;     // NOTE: redundant atm because it matches the slot index in the current impl
    int udp_port;   // UDP port associated to the socket (only if bound)
    struct in_addr ip_addr;     // IPv4 address associated to the socket (only if bound)
    int so_options; // socket options
} __rte_cache_aligned;

/* Descriptor of the zone in shared memory where packets are exchanged between app and poller */
struct exch_zone_info {
    uint64_t n_zones_active;
    struct exch_slot_info slots[NUM_SOCKETS_MAX];
};

/* Descriptor of the exchange zone queues and buffers for a socket */
struct exch_slot {
    struct rte_ring *rx_q;                      // RX queue
    struct rte_ring *tx_q;                      // TX queue
    struct rte_mbuf *rx_buffer[EXCH_BUF_SIZE];  // buffers storing rx packets before flushing to rt_ring
    uint16_t rx_count;                          // current number of packets in the rx buffer
} __rte_cache_aligned;

/* Global configuration (parsed from file) */
typedef struct {
    struct rte_ether_addr src_mac_addr;
    struct rte_ether_addr dst_mac_addr;
    struct in_addr src_ip_addr;
    char lcores_primary[MAX_ARG_LEN];
    char lcores_secondary[MAX_ARG_LEN];
    int n_mem_channels;
} configuration;

#endif //UDPDK_TYPES_H
