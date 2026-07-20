/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/queue.h>

#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_ethdev.h>
#include <rte_log.h>
#include <rte_ether.h>
#define RTE_LOGTYPE_APP RTE_LOGTYPE_USER1
#define RTE_ETHER_ADDR_PRT_FMT     "%02X:%02X:%02X:%02X:%02X:%02X"
#define RTE_ETHER_ADDR_BYTES(mac_addrs) ((mac_addrs)->addr_bytes[0]), \
					 ((mac_addrs)->addr_bytes[1]), \
					 ((mac_addrs)->addr_bytes[2]), \
					 ((mac_addrs)->addr_bytes[3]), \
					 ((mac_addrs)->addr_bytes[4]), \
					 ((mac_addrs)->addr_bytes[5])

#define RTE_ETHER_ADDR_FMT_SIZE         18
#define TX_DESC_PER_QUEUE 512
#define RX_DESC_PER_QUEUE 128
#define VDEV_NAME_FMT "net_pcap_%s_%d"
static struct rte_eth_conf port_conf_default;
static inline int
configure_vdev(uint16_t port_id)
{
	struct rte_ether_addr addr;
	const uint16_t rxRings = 0, txRings = 1;
	int ret;
	uint16_t q;

	if (!rte_eth_dev_is_valid_port(port_id))
		return -1;

	ret = rte_eth_dev_configure(port_id, rxRings, txRings,
					&port_conf_default);
	if (ret != 0)
		rte_exit(EXIT_FAILURE, "dev config failed\n");

	 for (q = 0; q < txRings; q++) {
		ret = rte_eth_tx_queue_setup(port_id, q, TX_DESC_PER_QUEUE,
				rte_eth_dev_socket_id(port_id), NULL);
		if (ret < 0)
			rte_exit(EXIT_FAILURE, "queue setup failed\n");
	}

	ret = rte_eth_dev_start(port_id);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "dev start failed\n");

	ret = rte_eth_macaddr_get(port_id, &addr);
	if (ret != 0)
		rte_exit(EXIT_FAILURE, "macaddr get failed\n");

	printf("Port %u MAC: %02"PRIx8" %02"PRIx8" %02"PRIx8
			" %02"PRIx8" %02"PRIx8" %02"PRIx8"\n",
			port_id,
			addr.addr_bytes[0], addr.addr_bytes[1],
			addr.addr_bytes[2], addr.addr_bytes[3],
			addr.addr_bytes[4], addr.addr_bytes[5]);

	ret = rte_eth_promiscuous_enable(port_id);
	if (ret != 0) {
		rte_exit(EXIT_FAILURE,
			 "promiscuous mode enable failed: %s\n",
			 rte_strerror(-ret));
		return ret;
	}

	return 0;
}
/**
 * Create and initialize vdev for control traffic from a specific port.
 *
 * @param port_id ID of port.
 * @param vport_id Returns ID of vdev.
 * @return int Returns 0 on success.
 */

static int
setup_ct_vdev(uint16_t port_id, uint16_t *vport_id)
{
	int res;
	char portargs[256];
	char portname[32];
        int nb_rxd = 8;
	struct rte_ether_addr addr;
        //struct rte_eth_dev_info port_info;
        //struct rte_eth_conf port_conf;
	/* get MAC address of physical port to use as MAC of virtio_user port */
	rte_eth_macaddr_get(port_id, &addr);

	/* set the name and arguments */
#if 0
	snprintf(portname, sizeof(portname), "virtio_user%u", port_id);
	snprintf(portargs, sizeof(portargs),
			"path=/dev/"
			"vhost-net,queues=1,queue_size=%u,iface=%s,"
			"mac=" RTE_ETHER_ADDR_PRT_FMT,
			nb_rxd, portname, RTE_ETHER_ADDR_BYTES(&addr));
#else
       snprintf(portname, sizeof(portname),VDEV_NAME_FMT, "pdump", 0);
       snprintf(portargs, strlen("tx_pcap=report.pcap") + 1, "tx_pcap=%s", "report.pcap");
#endif

	res = rte_eal_hotplug_add("vdev", portname, portargs);
	if (res < 0) {
                RTE_LOG(INFO, APP, "add hotplug dev fail.\n");
		return -1;
	}
	res = rte_eth_dev_get_port_by_name(portname, vport_id);
        
	if (res != 0) {
                rte_eal_hotplug_remove("vdev", portname);
		return -1;
	}

        RTE_LOG(INFO, APP, "physical dev port addr"  RTE_ETHER_ADDR_PRT_FMT "\n",RTE_ETHER_ADDR_BYTES(&addr));
        RTE_LOG(INFO, APP, "hotplug dev vport id %d.\n",*vport_id);
        if (0 != configure_vdev(*vport_id))
		goto exit ;
#if 0
        if (rte_eth_dev_info_get(*vport_id, &port_info) != 0)
		goto exit ;
        res = rte_eth_promiscuous_enable(*vport_id);
	if (res != 0)
		goto exit ;
        res = rte_eth_dev_start(*vport_id);
	if (res != 0)
		goto exit ;
        res = rte_eth_dev_set_link_up(*vport_id);
	if (res != 0)
		goto stop;
#endif
stop:
        rte_eth_dev_stop(*vport_id);
	//if (res != 0)
	//	goto exit ;
exit:
        rte_eal_hotplug_remove("vdev", portname);
	return 0;
}
static int
lcore_hello(__attribute__((unused)) void *arg)
{
	unsigned lcore_id;
	lcore_id = rte_lcore_id();
	printf("hello from core %u\n", lcore_id);
	return 0;
}

int
main(int argc, char **argv)
{
	int ret;
	unsigned lcore_id;
        uint16_t vport_id = 0, port_id =0;
	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_panic("Cannot init EAL\n");
        setup_ct_vdev(port_id,&vport_id);
	/* call lcore_hello() on every slave lcore */
	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		rte_eal_remote_launch(lcore_hello, NULL, lcore_id);
	}

	/* call it on master lcore too */
	lcore_hello(NULL);
	rte_eal_mp_wait_lcore();
	return 0;
}
