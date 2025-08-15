
#include "dpdk_work_space.h"
#include "dpdk_socket.h"
#include "dpdk_eth.h"
#include "dpdk_common.h"

#include <stdio.h>
#include <rte_cycles.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_malloc.h>
#include <pthread.h>
#define THREAD_NUM_MAX      64
__thread struct work_space *g_work_space;
static struct work_space *g_work_space_all[THREAD_NUM_MAX];
static void work_space_init_rss(struct work_space *ws);
static struct rte_mbuf *work_space_alloc_mbuf(struct work_space *ws)
{
    struct rte_mempool *p = NULL;

    p = port_get_mbuf_pool(ws->port_id, ws->queue_id);
    if (p) {
        return rte_pktmbuf_alloc(p);
    }

    return NULL;
}
static struct work_space *work_space_alloc(struct config *cfg, int id)
{
    size_t size = 0;
    uint32_t socket_num = 0;
    struct work_space *ws = NULL;

    socket_num = config_get_total_socket_num(cfg, id);
    size = sizeof(struct work_space) + socket_num * sizeof(struct socket);

    ws = (struct work_space *)rte_calloc("work_space", 1, size, CACHE_ALIGN_SIZE);
    if (ws != NULL) {
        printf("socket allocation succeeded, size %0.2fGB num %u\n", size * 1.0 / (1024 * 1024 * 1024), socket_num);
        ws->socket_table.socket_pool.num = socket_num;
    } else {
        printf("socket allocation failed, size %0.2fGB num %u\n", size * 1.0 / (1024 * 1024 * 1024), socket_num);
    }

    return ws;
}
static void work_space_init_time(struct work_space *ws)
{
#if 0
    uint32_t cpu_num = 0;
    uint64_t us = 0;

    cpu_num = ws->cfg->cpu_num;
    us = (1000ul * 1000ul * 1000ul) / (1000ul * cpu_num);

    work_space_wait_all(ws);
    usleep(ws->id * us);
#endif
    socket_timer_init();
    tick_time_init(&ws->time);
}

struct work_space *work_space_new(struct config *cfg, int id)
{
    struct work_space *ws = NULL;

    ws = work_space_alloc(cfg, id);
    if (ws == NULL) {
        return NULL;
    }

    g_work_space = ws;
    g_work_space_all[id] = ws;
    ws->cfg = cfg;
    ws->server = cfg->server;
    ws->vlan_id = cfg->vlan_id;
    ws->id = id;
    ws->ipv6 = cfg->af == AF_INET6;
    ws->port_id = DEFAULT_PORTID; 
    ws->port = netif_port_get(ws->port_id);
    ws->queue_id = DEFAULT_QUEUEID;
    ws->kni = false;
    ws->http = false;
    ws->log = stderr;
    ws->tx_queue.tx_burst = cfg->tx_burst;
    if (tcp_init(ws) < 0) {
        printf("tcp_init error");
        goto err;
    }
    work_space_init_time(ws);
    if (socket_table_init(ws) < 0) {
        goto err;
    }
    work_space_init_rss(ws);
    return ws;
err:
    work_space_close(ws);
    return NULL;
}
static void work_space_close_log(struct work_space *ws)
{
    if (ws && ws->log) {
        fclose(ws->log);
        ws->log = NULL;
    }
}
void work_space_close(struct work_space *ws)
{
    work_space_close_log(ws);
}

static void work_space_init_rss(struct work_space *ws)
{
    int i = 0;
    int idx = 0;
    struct work_space *ws2 = NULL;
    struct socket_table *st = NULL;
    struct socket_table *st2 = NULL;

    st = &ws->socket_table;
    st->rss = ws->cfg->rss;
    st->rss_id = ws->queue_id;
    st->rss_num = ws->port->queue_num;

    for (i = 0; i < THREAD_NUM_MAX; i++) {
        ws2 = g_work_space_all[i];
        if ((ws2 == NULL) || (ws2->port != ws->port)) {
            continue;
        }

        st2 = &ws2->socket_table;
        if (st->rss == RSS_L3) {
            idx = ntohl(st2->server_ip) & 0xff;
        } else {
            idx = ws2->queue_id;
        }
        st->socket_table_hash[idx] = st2;
    }
}
