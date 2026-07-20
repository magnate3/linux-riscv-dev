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
#define MEMPOOL_CACHE_SIZE 128
struct rte_mempool *pool1,*pool2;
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
        struct rte_mbuf *mbuf1 = NULL;
        struct rte_mbuf *mbuf2 = NULL;
	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_panic("Cannot init EAL\n");
#if 0
	/* call lcore_hello() on every slave lcore */
	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		rte_eal_remote_launch(lcore_hello, NULL, lcore_id);
	}
#endif
        pool1 = rte_pktmbuf_pool_create("mbuf_pool1", 1024,
                                                    MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
                                                    rte_socket_id());
        if (pool1 == NULL)
            rte_exit(EXIT_FAILURE, "Cannot init mbuf pool1\n");
        mbuf1 = rte_pktmbuf_alloc(pool1);
        printf("mbuf1 %p \n",mbuf1);
        rte_pktmbuf_free(mbuf1); 
        rte_pktmbuf_free(mbuf1); 
        mbuf1 = rte_pktmbuf_alloc(pool1);
        mbuf2 = rte_pktmbuf_alloc(pool1);
        printf("mbuf1 %p , mbuf2 %p \n",mbuf1,mbuf2);
#if 1
        pool2 = rte_pktmbuf_pool_create("mbuf_pool2", 1024, 0, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
                                                    rte_socket_id());
        if (pool2 == NULL)
            rte_exit(EXIT_FAILURE, "Cannot init mbuf pool2\n");
        printf("------------------- test mem pool2--------------- \n");
        mbuf1 = rte_pktmbuf_alloc(pool2);
        printf("mbuf1 %p \n",mbuf1);
        rte_pktmbuf_free(mbuf1); 
        rte_pktmbuf_free(mbuf1); 
        mbuf1 = rte_pktmbuf_alloc(pool2);
        mbuf2 = rte_pktmbuf_alloc(pool2);
        printf("mbuf1 %p , mbuf2 %p \n",mbuf1,mbuf2);
#endif
	rte_eal_mp_wait_lcore();
        getchar();
	return 0;
}
