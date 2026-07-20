
#ifndef __CONFIG_H
#define __CONFIG_H
#include<stdbool.h>
#include "ip_range.h"
#include "ip_range.h"
#define CACHE_ALIGN_SIZE    64
#define TCP_WIN             (1460 * 40)
#define NETWORK_PORT_NUM    65536

#define PACKET_SIZE_MAX     1514
#define DEFAULT_CPS         1000
#define DEFAULT_INTERVAL    1       /* 1s */
#define DEFAULT_DURATION    60
#define DEFAULT_TTL         64
#define ND_TTL              255
#define DEFAULT_LAUNCH      4
#define DELAY_SEC           4
#define WAIT_DEFAULT        3
#define SLOW_START_DEFAULT  30
#define SLOW_START_MIN      10
#define SLOW_START_MAX      600
#define KEEPALIVE_REQ_NUM   32767  /* 15 bits */

#define JUMBO_FRAME_MAX_LEN 0x2600
#define JUMBO_PKT_SIZE_MAX  (JUMBO_FRAME_MAX_LEN - ETHER_CRC_LEN)
#define JUMBO_MTU           (JUMBO_PKT_SIZE_MAX - 14)
#define JUMBO_MBUF_SIZE     (1024 * 11)
#define MBUF_DATA_SIZE      (1024 * 10)

#define MSS_IPV4            (PACKET_SIZE_MAX - 14 - 20 - 20)
#define MSS_IPV6            (PACKET_SIZE_MAX - 14 - 40 - 20)
#define MSS_JUMBO_IPV4      (JUMBO_PKT_SIZE_MAX - 14 - 20 - 20)
#define MSS_JUMBO_IPV6      (JUMBO_PKT_SIZE_MAX - 14 - 40 - 20)

#define DEFAULT_WSCALE      13
#define NETWORK_PORT_NUM    65536
#define TCP_ACK_DELAY_MAX   1024
#define RSS_NONE            0
#define RSS_L3              1
#define RSS_L3L4            2
#define RSS_AUTO            3
#define DEFAULT_TTL         64
struct config {
    bool server;
    bool tcp_rst ;
    bool keepalive;
    bool client_hop;
    bool vxlan;
    bool jumbo;
    uint8_t rss;
    uint8_t protocol;   /* TCP/UDP */
    uint8_t tos;
    uint8_t tx_burst;
    int lport_min;
    int lport_max;
    int listen;
    int listen_num;
    int af;
    uint16_t vlan_id;
    int duration;
    int mss;
    uint64_t keepalive_request_interval_us;

    /* tsc */
    uint64_t keepalive_request_interval;
    int keepalive_request_num;
    struct ip_group client_ip_group;
    struct ip_group server_ip_group;
};
extern struct config g_config;
uint32_t config_get_total_socket_num(struct config *cfg, int id);
//static struct netif_port *config_port_get(struct config *cfg, int thread_id, int *p_queue_id)
//{
//}
int init_config(void);
#endif
