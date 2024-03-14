#ifndef __DPDK_ETH_H__
#define __DPDK_ETH_H__
#include <net/if.h>
#include <net/ethernet.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pcap.h>
#include "list.h"
#include "conf/common.h"
#include "conf/inet.h" // inet_addr

#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_mempool.h>
#include <rte_atomic.h>
#define ETHER_TYPE_IPv4 ETHERTYPE_IP
#define ETHER_TYPE_IPv6 ETHERTYPE_IPV6
#define ETHER_TYPE_ARP  ETHERTYPE_ARP
/* max tx/rx queue number for each nic */
#define NETIF_MAX_QUEUES            64
/* max nic number used in the program */
#define NETIF_MAX_PORTS             16
/* maximum pkt number at a single burst */
#define NETIF_MAX_PKT_BURST         32
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
    uint16_t      id;                         /* device id */
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
     struct list_head        list;                       /* device list node hashed by id */
    struct list_head        nlist;                      /* device list node hashed by name */
    pcap_dumper_t *         dumper ;
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

struct netif_port* netif_port_get(uint16_t id);

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
static inline int eth_addr_equal(const struct rte_ether_addr *addr1,
                                 const struct rte_ether_addr *addr2)
{
    const uint16_t *a = (const uint16_t *)addr1;
    const uint16_t *b = (const uint16_t *)addr2;

    return ((a[0]^b[0]) | (a[1]^b[1]) | (a[2]^b[2])) == 0;
}

static inline char *eth_addr_dump(const struct rte_ether_addr *ea,
                                  char *buf, size_t size)
{
    snprintf(buf, size, "%02x:%02x:%02x:%02x:%02x:%02x",
             ea->addr_bytes[0], ea->addr_bytes[1],
             ea->addr_bytes[2], ea->addr_bytes[3],
             ea->addr_bytes[4], ea->addr_bytes[5]);
    return buf;
}
void init_netif(struct netif_port * dev ,struct rte_mempool      *mbuf_pool,uint16_t portid);
int netif_port_register(struct netif_port *port);
void lcore_process_packets(struct rte_mbuf **mbufs,const uint16_t count, uint16_t port_id);
int init_dpdk_eth_mod(void);
int netif_port_register(struct netif_port *port);
struct rte_mbuf * get_mbuf_from_netif(struct netif_port * dev);
uint16_t eth_send_packets(struct rte_mbuf **mbuf, uint16_t port_id,uint16_t queue_id, const uint16_t burst_num);
void dump_pcap( const struct netif_port *dev, const u_char *pkt, int len);
#endif

