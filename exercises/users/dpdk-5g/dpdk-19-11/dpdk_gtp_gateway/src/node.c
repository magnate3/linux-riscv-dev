#include "node.h"

#include <rte_bus_pci.h>

#include "pktbuf.h"

/* GLOBALS */
numa_info_t numa_node_info[GTP_MAX_NUMANODE];

/* EXTERN */
extern app_confg_t app_config;

static const struct rte_eth_conf portConf = {
    .rxmode = {
        // .offloads = DEV_RX_OFFLOAD_CHECKSUM,
        .split_hdr_size   = 0, /**< Header Split disabled */
        // .hw_vlan_filter = 0, /**< VLAN filtering disabled */
        // .jumbo_frame    = 0, /**< Jumbo Frame Support disabled */
        // .hw_strip_crc   = 0, /**< CRC stripped by hardware */
    },
    .txmode = {
        .mq_mode = ETH_MQ_TX_NONE,
        .offloads = DEV_TX_OFFLOAD_IPV4_CKSUM |
                    DEV_TX_OFFLOAD_UDP_CKSUM,
    },
};

int32_t
populate_node_info(void)
{
    int32_t i = 0, socketId = -1, lcoreIndex = 0, enable = 0;
    struct rte_eth_dev_info devInfo;
    struct rte_ether_addr addr;

    /* fetch total lcore count under DPDK */
    uint32_t lc;
    RTE_LCORE_FOREACH(lc) {
        socketId = rte_lcore_to_socket_id(lc);
        lcoreIndex = rte_lcore_index(lc);
        enable = rte_lcore_is_enabled(lc);

        printf("\n Logical %d Physical %d Socket %d enabled %d", lcoreIndex, lc, socketId, enable);

        if (likely(enable)) {
            /* classify the lcore info per NUMA node */
            numa_node_info[socketId].lcoreAvail = numa_node_info[socketId].lcoreAvail | (1 << lcoreIndex);
            numa_node_info[socketId].lcoreTotal += 1;
        } else {
            rte_panic("\nERROR: Lcore %d Socket %d not enabled\n", lcoreIndex, socketId);
            exit(EXIT_FAILURE);
        }
    }
    printf("\n");

    /* Create mempool per numa node based on interface available */
    uint8_t portCount = rte_eth_dev_count_avail();
    for (i = 0; i < portCount; i++) {
        rte_eth_dev_info_get(i, &devInfo);
        rte_eth_macaddr_get(i, &addr);

        if (rte_hash_lookup(app_config.gtp_port_hash, &i) >= 0) {
            printf("\n [Interface %d *GTPGW*]", i);
        } else {
            printf("\n [Interface %d]", i);
        }

        printf("\n - Driver: %s", devInfo.driver_name);
        printf("\n - If index: %d", devInfo.if_index);
        printf("\n - MAC: %02" PRIx8 ":%02" PRIx8 ":%02" PRIx8
               " %02" PRIx8 ":%02" PRIx8 ":%02" PRIx8,
               addr.addr_bytes[0], addr.addr_bytes[1],
               addr.addr_bytes[2], addr.addr_bytes[3],
               addr.addr_bytes[4], addr.addr_bytes[5]);

        const struct rte_pci_device *pci_dev = RTE_DEV_TO_PCI(devInfo.device);
        if (pci_dev) {
            printf("\n - PCI INFO ");
            printf("\n -- ADDR - domain:bus:devid:function %x:%x:%x:%x",
                   pci_dev->addr.domain,
                   pci_dev->addr.bus,
                   pci_dev->addr.devid,
                   pci_dev->addr.function);
            printf("\n == PCI ID - vendor:device:sub-vendor:sub-device %x:%x:%x:%x",
                   pci_dev->id.vendor_id,
                   pci_dev->id.device_id,
                   pci_dev->id.subsystem_vendor_id,
                   pci_dev->id.subsystem_device_id);
            printf("\n -- numa node: %d", devInfo.device->numa_node);
        }

        socketId = (devInfo.device->numa_node == -1) ? 0 : devInfo.device->numa_node;
        numa_node_info[socketId].intfAvail = numa_node_info[socketId].intfAvail | (1 << i);
        numa_node_info[socketId].intfTotal += 1;
        printf("\n");
    }

    /* allocate mempool for numa which has NIC interfaces */
    for (i = 0; i < GTP_MAX_NUMANODE; i++) {
        if (likely(numa_node_info[i].intfAvail)) {
            /* ToDo: per interface */
            uint8_t portIndex = 0;
            char mempoolName[25];

            /* create mempool for TX */
            sprintf(mempoolName, "mbuf_pool-%d-%d-tx", i, portIndex);
            numa_node_info[i].tx[portIndex] = rte_mempool_create(
                mempoolName, NB_MBUF,
                MBUF_SIZE, 64,
                sizeof(struct rte_pktmbuf_pool_private),
                rte_pktmbuf_pool_init, NULL,
                rte_pktmbuf_init, NULL,
                i, /*SOCKET_ID_ANY*/
                0 /*MEMPOOL_F_SP_PUT*/);
            if (unlikely(numa_node_info[i].tx[portIndex] == NULL)) {
                rte_panic("\n ERROR: failed to get mem-pool for tx on node %d intf %d\n", i, portIndex);
                exit(EXIT_FAILURE);
            }

            /* create mempool for RX */
            sprintf(mempoolName, "mbuf_pool-%d-%d-rx", i, portIndex);
            numa_node_info[i].rx[portIndex] = rte_mempool_create(
                mempoolName, NB_MBUF,
                MBUF_SIZE, 64,
                sizeof(struct rte_pktmbuf_pool_private),
                rte_pktmbuf_pool_init, NULL,
                rte_pktmbuf_init, NULL,
                i, /*SOCKET_ID_ANY*/
                0 /*MEMPOOL_F_SP_PUT*/);
            if (unlikely(numa_node_info[i].rx[portIndex] == NULL)) {
                rte_panic("\n ERROR: failed to get mem-pool for rx on node %d intf %d\n", i, portIndex);
                exit(EXIT_FAILURE);
            }
        }
    }

    return 0;
}

int32_t
node_interface_setup(void)
{
    uint8_t portIndex = 0, portCount = rte_eth_dev_count_avail();
    int32_t ret = 0, socket_id = -1;

    for (portIndex = 0; portIndex < portCount; portIndex++) {
        /* fetch the socket Id to which the port the mapped */
        for (ret = 0; ret < GTP_MAX_NUMANODE; ret++) {
            if (numa_node_info[ret].intfTotal) {
                if (numa_node_info[ret].intfAvail & (1 << portIndex)) {
                    socket_id = ret;
                    break;
                }
            }
        }

        ret = rte_eth_dev_configure(portIndex, 1, 1, &portConf);
        if (unlikely(ret < 0)) {
            rte_panic("ERROR: Dev Configure\n");
            return -1;
        }

        ret = rte_eth_rx_queue_setup(portIndex, 0, RTE_TEST_RX_DESC_DEFAULT,
                                     0, NULL, numa_node_info[socket_id].rx[0]);
        if (unlikely(ret < 0)) {
            rte_panic("ERROR: Rx Queue Setup\n");
            return -2;
        }

        ret = rte_eth_tx_queue_setup(portIndex, 0, RTE_TEST_TX_DESC_DEFAULT,
                                     0, NULL);
        if (unlikely(ret < 0)) {
            rte_panic("ERROR: Tx Queue Setup\n");
            return -3;
        }

        rte_eth_promiscuous_enable(portIndex);
        rte_eth_dev_start(portIndex);
    }

    return 0;
}
