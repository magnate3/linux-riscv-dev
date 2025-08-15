#include "dpdk_config.h"
#include "dpdk_eth.h"
#include "dpdk_common.h"
extern struct in6_addr net_ip6 ;
struct config g_config = {
    .tcp_rst = true,
    .server = true,
    .vxlan = false,
    .jumbo = false,
    .rss =  RSS_NONE,
    .keepalive = 10,
    .vlan_id = 0,
    // use ipv6
    .af = AF_INET6,
    .mss = MSS_IPV6,
    .protocol = IPPROTO_TCP, 
    .listen = 80,
    .listen_num = 2,
    // lport_min is client port min, lport_max is client port max
    .lport_min = 8788,
    .lport_max = 8798,
    .tx_burst = 1,
};
static uint32_t config_client_ip_range_socket_num(struct config *cfg, struct ip_range *ip_range)
{
    /*
 *      * client-ip-num * client-port-num * server-ip-num * server-listen-port-num
 *           * client-port-num: 1-65535, skip port 0
 *                * server-ip-num: 1, each thread using one server-ip
 *                     * */
    return ip_range->num * cfg->listen_num * (cfg->lport_max - cfg->lport_min + 1);
}

static struct netif_port *config_port_get(struct config *cfg, int thread_id, int *p_queue_id)
{
    return netif_port_get(DEFAULT_PORTID);
}
uint32_t config_get_total_socket_num(struct config *cfg, int id)
{
    uint32_t num = 0;
    struct ip_range *client_ip_range = NULL;
#if 0
    struct netif_port *port = NULL;
    port = config_port_get(cfg, id, NULL);
#endif
    if (cfg->server) {
        /*
 *          * the DUT(eg load balancer) may connect to all servers
 *                   * */
        for_each_ip_range(&cfg->client_ip_group, client_ip_range) {
            num += config_client_ip_range_socket_num(cfg, client_ip_range);
        }
    } else {
#if 0
        client_ip_range = &(port->client_ip_range);
        num = config_client_ip_range_socket_num(cfg, client_ip_range);
#endif
    }

    return num;
}
int init_config(void)
{
    struct config *cfg = &g_config;
    ipaddr_t ip;
    struct ip_range *ip_range = NULL;
    struct ip_group *ip_group = &cfg->client_ip_group;
    struct netif_port *port = NULL;
    port = config_port_get(cfg, 0, NULL);
    if (inet_pton(AF_INET6, TCP_CLIENT82_IP6, &ip.in6)< 0) 
    {
        return -1;
    }
    ip_range = &ip_group->ip_range[ip_group->num];
    ip_range_init(ip_range, ip, 1);
    ip_group->num++;
    if (ip_range_init(&port->client_ip_range, ip, 1) < 0) {
        printf("bad server ip range \n");
        exit(0);
    }
    ip.in6 = net_ip6;
    if (ip_range_init(&port->server_ip_range, ip, 1) < 0) {
        printf("bad server ip range \n");
        exit(0);
    }
    http_set_test_payload_server(cfg,"hello world");
    return 0;
}
