#ifndef __DPDK_ETH_H__
#define __DPDK_ETH_H__
#include <net/if.h>
#include <net/ethernet.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "conf/common.h"
#include "conf/inet.h" // inet_addr
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_atomic.h>
#define ETHER_TYPE_IPv4 ETHERTYPE_IP
#define ETHER_TYPE_IPv6 ETHERTYPE_IPV6
#define ETHER_TYPE_ARP  ETHERTYPE_ARP
typedef enum {
    ETH_PKT_HOST,
    ETH_PKT_BROADCAST,
    ETH_PKT_MULTICAST,
    ETH_PKT_OTHERHOST,
} eth_type_t;
typedef enum {
    PORT_TYPE_GENERAL,
    PORT_TYPE_BOND_MASTER,
    PORT_TYPE_BOND_SLAVE,
    PORT_TYPE_VLAN,
    PORT_TYPE_TUNNEL,
    PORT_TYPE_INVAL,
} port_type_t;
struct netif_port {
    char                    name[IFNAMSIZ];  /* device name */
    portid_t                id;                         /* device id */
    port_type_t             type;                       /* device type */
    uint16_t                flag;                       /* device flag */
    int                     nrxq;                       /* rx queue number */
    int                     ntxq;                       /* tx queue numbe */
    uint16_t                rxq_desc_nb;                /* rx queue descriptor number */
    uint16_t                txq_desc_nb;                /* tx queue descriptor number */
    struct rte_ether_addr   addr;                       /* MAC address */
    int                     socket;                     /* socket id */
    int                     hw_header_len;              /* HW header length */
    uint16_t                mtu;                        /* device mtu */
    struct rte_mempool      *mbuf_pool;                 /* packet mempool */
    struct rte_eth_dev_info dev_info;                   /* PCI Info + driver name */
    struct rte_eth_conf     dev_conf;                   /* device configuration */
    struct rte_eth_stats    stats;                      /* last device statistics */
    rte_rwlock_t            dev_lock;                   /* device lock */
} __rte_cache_aligned;
struct inet_ifaddr {

    int                     af;
    union inet_addr         addr;       /* primary address of iface */
    uint8_t                 plen;
    union inet_addr         mask;
    union inet_addr         bcast;

    uint8_t                 scope;
    uint32_t                flags;
    rte_atomic32_t          refcnt;
};
void inet_addr_ifa_put(struct inet_ifaddr *ifa);
struct inet_ifaddr *inet_addr_ifa_get(int af, const struct netif_port *dev,
                                      union inet_addr *addr);

struct netif_port* netif_port_get(portid_t id);

#define ETH_ADDR_LEN        6
#define ETH_ADDR_STR_LEN    17

struct eth_addr {
    uint8_t bytes[ETH_ADDR_LEN];
} __attribute__((__packed__));

struct eth_hdr {
    struct eth_addr d_addr;
    struct eth_addr s_addr;
    uint16_t type;
} __attribute__((__packed__));
static inline void eth_addr_copy(struct eth_addr *dst, const struct eth_addr *src)
{
    memcpy((void*)dst, (const void*)src, sizeof(struct eth_addr));
}
static inline void eth_hdr_set(struct eth_hdr *eth, uint16_t type, const struct eth_addr *d_addr,
    const struct eth_addr *s_addr)
{
    eth->type = htons(type);
    eth_addr_copy(&eth->d_addr, d_addr);
    eth_addr_copy(&eth->s_addr, s_addr);
}
static inline int eth_addr_is_zero(const struct eth_addr *ea)
{
    return ((ea)->bytes[0] == 0) && ((ea)->bytes[1] == 0) && ((ea)->bytes[2] == 0) &&
            ((ea)->bytes[3] == 0) && ((ea)->bytes[4] == 0) && ((ea)->bytes[5] == 0);
}
#endif

