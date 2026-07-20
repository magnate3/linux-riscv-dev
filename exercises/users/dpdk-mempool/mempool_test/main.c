/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/queue.h>
#include <signal.h>
#include <stdbool.h>
#include <pthread.h>
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
uint32_t PINGPONG_LOG_LEVEL = RTE_LOG_DEBUG;
int RTE_LOGTYPE_PINGPONG;
struct rte_mempool *pool1,*pool2;
struct rte_mbuf *mbuf1 = NULL;
struct rte_mbuf *mbuf2 = NULL;
static volatile bool force_quit;
static void
signal_handler(int signum)
{
    if (signum == SIGINT || signum == SIGTERM)
    {
        rte_log(RTE_LOG_INFO, RTE_LOGTYPE_PINGPONG, "\n\nSignal %d received, preparing to exit...\n", signum);
        force_quit = true;
    }
}
static int
lcore_hello(__attribute__((unused)) void *arg)
{
     bool eq = false;
     while(!force_quit)
     {
        mbuf2 = rte_pktmbuf_alloc(pool1);
        eq = (mbuf1 == mbuf2); 
        if(eq) 
        printf("tid %lu, mbuf1 %p , mbuf2 %p, mbuf1 == mbuf2 ? %d \n",pthread_self(),mbuf1,mbuf2, eq);
        //printf("tid %lu, mbuf1 %p , mbuf2 %p, mbuf1 == mbuf2 ? %d \n",pthread_self(),mbuf1,mbuf2, mbuf1 == mbuf2);
        rte_pktmbuf_free(mbuf2); 
        if(eq)
            break;
     }
     return 0;
}

int
main(int argc, char **argv)
{
	int ret;
	unsigned lcore_id;
        bool eq = false;
	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_panic("Cannot init EAL\n");
        force_quit = false;
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        pool1 = rte_pktmbuf_pool_create("mbuf_pool1", 256,
                                                    MEMPOOL_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE,
                                                    rte_socket_id());
        if (pool1 == NULL)
            rte_exit(EXIT_FAILURE, "Cannot init mbuf pool1\n");
	/* call lcore_hello() on every slave lcore */
	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		rte_eal_remote_launch(lcore_hello, NULL, lcore_id);
	}
        while(!force_quit)
        {
           mbuf1 = rte_pktmbuf_alloc(pool1);
           eq = (mbuf1 == mbuf2); 
           if(eq) 
           printf("tid %lu, mbuf1 %p , mbuf2 %p, mbuf1 == mbuf2 ? %d \n",pthread_self(),mbuf1,mbuf2, eq);
           //printf("tid %lu, mbuf1 %p , mbuf2 %p, mbuf1 == mbuf2 ? %d \n",pthread_self(),mbuf1,mbuf2, mbuf1 == mbuf2);
           rte_pktmbuf_free(mbuf1); 
           if(eq)
               break;
        }
	rte_eal_mp_wait_lcore();
        //getchar();
	return 0;
}
