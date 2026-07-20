/*
 * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Jianzhang Peng (pengjianzhang@baidu.com)
 */

#ifndef __WORK_SPACE_H
#define __WORK_SPACE_H

#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <rte_mbuf.h>
#include <rte_ethdev.h>
#include "dpdk.h"
#include "mbuf_cache.h"
#include "dpdk_csum.h"
#include "dpdk_tcp.h"
#include "dpdk_eth.h"
#include "dpdk_common.h"
#include "tick.h"
#include "stat.h"
#include "dpdk_socket.h"
#include "ip_list.h"

#define NB_RXD              4096
#define RX_BURST_MAX        NB_RXD
struct socket_table;

extern __thread struct work_space *g_work_space;
#define g_current_ticks (g_work_space->time.tick.count)
#define g_current_seconds (g_work_space->time.second.count)

#define work_space_tsc(ws) (ws->time.tsc)
#define NB_TXD              4096
#define TX_QUEUE_SIZE       NB_TXD
struct tx_queue {
    uint16_t head;
    uint16_t tail;
    uint16_t tx_burst;
    struct rte_mbuf *tx[TX_QUEUE_SIZE];
};
struct work_space {
    /* read mostly */
    uint8_t id;
    uint8_t ipv6:1;
    uint8_t server:1;
    uint8_t kni:1;
    uint8_t change_dip:1;
    uint8_t http:1;
    uint8_t flood:1;
    uint8_t tos;
    uint8_t port_id;
    uint8_t queue_id;

    uint16_t ip_id;
    bool lldp;
    bool exit;
    bool stop;
    bool start;
    uint16_t vlan_id;
    uint32_t vni:24;
    uint32_t vxlan:8;
    uint32_t vtep_ip; /* each queue has a vtep ip */
    struct tick_time time;
#if 0
    struct cpuload load;
    struct client_launch client_launch;
#endif
    struct mbuf_cache tcp_opt;
    struct mbuf_cache tcp_data;
    union {
        struct mbuf_cache udp;
        struct mbuf_cache tcp;
    };

    FILE *log;
    struct config *cfg;
    struct netif_port *port;
    void (*run_loop)(struct work_space *ws);

    struct {
        int next;
        struct socket *sockets[TCP_ACK_DELAY_MAX];
    } ack_delay;
    struct tx_queue tx_queue;
    struct rte_mbuf *mbuf_rx[NB_RXD];
    struct ip_list  dip_list;
    struct socket_table socket_table;
};
struct work_space *work_space_new(struct config *cfg, int id);
void work_space_close(struct work_space *ws);
static inline bool work_space_in_duration(struct work_space *ws)
{
    if ((ws->time.second.count < (uint64_t)(g_config.duration)) && (ws->stop == false)) {
        return true;
    } else {
        return false;
    }
}

static inline void work_space_tx_flush(struct work_space *ws)
{
    int i = 0;
    int n = 0;
    int num = 0;
    struct tx_queue *queue = &ws->tx_queue;
    struct rte_mbuf **tx = NULL;

    if (queue->head == queue->tail) {
        return;
    }

    for (i = 0; i < 8; i++) {
        num = queue->tail - queue->head;
        if (num > queue->tx_burst) {
            num = queue->tx_burst;
        }

        tx = &queue->tx[queue->head];
        dump_pcap(netif_port_get(DEFAULT_PORTID),rte_pktmbuf_mtod(tx[0],u_char*),tx[0]->pkt_len);
        n = rte_eth_tx_burst(g_work_space->port_id, g_work_space->queue_id, tx, num);
        printf("%s, port id %d , queue id %d, send %d burst packet, num of success %d \n",__func__,g_work_space->port_id,
                                        g_work_space->queue_id,num, n);
        queue->head += n;
        if (queue->head == queue->tail) {
            queue->head = 0;
            queue->tail = 0;
            return;
        } else if (queue->tail < TX_QUEUE_SIZE) {
            return;
        }
    }

    num = queue->tail - queue->head;
    net_stats_tx_drop(num);
    for (i = queue->head; i < queue->tail; i++) {
        rte_pktmbuf_free(queue->tx[i]);
    }
    queue->head = 0;
    queue->tail = 0;
}

static inline void work_space_tx_send(struct work_space *ws, struct rte_mbuf *mbuf)
{
    struct tx_queue *queue = &ws->tx_queue;

    if (ws->vlan_id) {
        mbuf->ol_flags |= RTE_MBUF_F_TX_VLAN;
        mbuf->vlan_tci = ws->vlan_id;
    }

    net_stats_tx(mbuf);
    queue->tx[queue->tail] = mbuf;
    queue->tail++;
    if (((queue->tail - queue->head) >= queue->tx_burst) || (queue->tail == TX_QUEUE_SIZE)) {
        work_space_tx_flush(ws);
    }
}
static inline void work_space_tx_send_tcp(struct work_space *ws, struct rte_mbuf *mbuf)
{
    uint64_t ol_flags = RTE_MBUF_F_TX_TCP_CKSUM;

    if (ws->vxlan) {
        ol_flags = RTE_MBUF_F_TX_UDP_CKSUM;
    }

    csum_offload_ip_tcpudp(mbuf, ol_flags);
    net_stats_tcp_tx();
    work_space_tx_send(ws, mbuf);
}
#endif
