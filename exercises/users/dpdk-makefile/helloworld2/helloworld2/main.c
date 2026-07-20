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
        uint16_t nb_ports =0, phy_port=0;
        struct rte_eth_dev_info dev_info;
        struct rte_eth_link link;
	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_panic("Cannot init EAL\n");
        nb_ports = rte_eth_dev_count_avail();
        printf("Number of Ports: %d\n", nb_ports);
        ret = rte_eth_dev_info_get(0, &dev_info);
        if (ret != 0)
                rte_exit(EXIT_FAILURE,
                        "Error during getting device (port %u) info: %s\n",
                        phy_port, strerror(-ret));
        rte_eth_link_get(phy_port, &link);
         printf("phy_port_ is %u link status up ? %u \n",phy_port, link.link_status == ETH_LINK_UP);
	/* call lcore_hello() on every slave lcore */
	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		rte_eal_remote_launch(lcore_hello, NULL, lcore_id);
	}

	/* call it on master lcore too */
	lcore_hello(NULL);

	rte_eal_mp_wait_lcore();
	return 0;
}
