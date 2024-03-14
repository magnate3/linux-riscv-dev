/*   SPDX-License-Identifier: BSD-3-Clause
 *   Copyright (C) 2015 Intel Corporation.
 *   All rights reserved.
 */

#include "spdk/stdinc.h"

#include "spdk/config.h"
#include "spdk/env.h"
#include "spdk/util.h"
#include "spdk/env_dpdk.h"
#include "spdk/memory.h"
#include "test/spdk_cunit.h"
#include "spdk_internal/assert.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/queue.h>
#include<stdbool.h>

#include <rte_memory.h>
#include <rte_launch.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_debug.h>
#include <rte_config.h>
#include <rte_malloc.h>

#define __SPDK_ENV_NAME(path)	(strrchr(#path, '/') + 1)
#define _SPDK_ENV_NAME(path)	__SPDK_ENV_NAME(path)
#define SPDK_ENV_NAME		_SPDK_ENV_NAME(SPDK_CONFIG_ENV)
#define CU_ASSERT assert
#define PAGE_ARRAY_SIZE (100)
#define REG_MAP_REGISTERED (1ULL << 62)
uint64_t
test_spdk_vtophys(struct spdk_mem_map *vtophys_map, const void *buf, uint64_t *size)
{
        uint64_t vaddr, paddr_2mb;

        vaddr = (uint64_t)buf;
        paddr_2mb = spdk_mem_map_translate(vtophys_map, vaddr, size);

        /*
 *          * SPDK_VTOPHYS_ERROR has all bits set, so if the lookup returned SPDK_VTOPHYS_ERROR,
 *                   * we will still bitwise-or it with the buf offset below, but the result will still be
 *                            * SPDK_VTOPHYS_ERROR. However now that we do + rather than | (due to PCI vtophys being
 *                                     * unaligned) we must now check the return value before addition.
 *                                              */
#if 0
        SPDK_STATIC_ASSERT(SPDK_VTOPHYS_ERROR == UINT64_C(-1), "SPDK_VTOPHYS_ERROR should be all 1s");
        if (paddr_2mb == SPDK_VTOPHYS_ERROR) {
                return SPDK_VTOPHYS_ERROR;
        } else {
                return paddr_2mb + (vaddr & MASK_2MB);
        }
#endif
       return paddr_2mb;
}
char * g_p = NULL;
static int
test_mem_map_notify(void *cb_ctx, struct spdk_mem_map *map,
                    enum spdk_mem_map_notify_action action,
                    void *vaddr, size_t size)
{
                printf("*************** %s: spdk_mem_map addr %p, %s event \n", __func__,map, SPDK_MEM_MAP_NOTIFY_REGISTER == action ? "register": "unregister");
                switch (action) {
                case SPDK_MEM_MAP_NOTIFY_REGISTER:
                        /* This page should not already be registered */
                        g_p = (char*)malloc(1);
                        printf("g_p addr %p \n", g_p);
                        spdk_mem_map_set_translation(map, (uint64_t)vaddr, size,  (uint64_t)g_p);
                        //spdk_mem_map_translate(map, (uint64_t)vaddr, &size);
                        //spdk_mem_map_clear_translation(map, (uint64_t)vaddr, size);
                        break;
                case SPDK_MEM_MAP_NOTIFY_UNREGISTER:
                        //SPDK_CU_ASSERT_FATAL(spdk_bit_array_get(g_page_array, i) == true);
                        //spdk_bit_array_clear(g_page_array, i);
                        spdk_mem_map_clear_translation(map, (uint64_t)vaddr, size);
                        if (g_p) {
                              free(g_p);
                        }
                        break;
                default:
                        SPDK_UNREACHABLE();
                }
        return 0;
}
static int
test_check_regions_contiguous(uint64_t addr1, uint64_t addr2)
{
        return addr1 == addr2;
}
const struct spdk_mem_map_ops test_mem_map_ops = {
        .notify_cb = test_mem_map_notify,
        .are_contiguous = test_check_regions_contiguous
};
static int 
test_mem_map_registration(void)
{
        struct spdk_mem_map *map;
        int rc = 0;
        char* ptr;
        //uint64_t default_translation = 0xDEADBEEF0BADF00D;
        uint64_t size =0x200000;
        uint64_t paddr;
        map = spdk_mem_map_alloc(SPDK_VTOPHYS_ERROR, &test_mem_map_ops, NULL);
        //  MASK_2MB=1fffff
        printf("MASK_2MB %llx , MASK_2MB & size %llx \n", MASK_2MB, MASK_2MB & size);
        printf("%s: spdk_mem_map addr %p \n", __func__,map);
        SPDK_CU_ASSERT_FATAL(map != NULL);
        //ptr = (char *)malloc(size);
        ptr = (char *)rte_malloc("test",0x200000,0x200000);
        printf("malloc virt addr %p and virt addr & MASK_2MB %llx \n", ptr, MASK_2MB&(uint64_t)ptr);
#if 0
        rc = spdk_mem_register(ptr,  size);
        if (rc) {
             printf("spdk mem register fail \n");
             return 0;
        }
#endif
	paddr = test_spdk_vtophys(map,ptr, &size);
        printf("virt addr %p and phy addr %lx \n", ptr, paddr);
        //*ptr= 'A';
	paddr = test_spdk_vtophys(map,ptr, &size);
        printf("after pagefault, virt addr %p and phy addr %lx \n", ptr, paddr);
        spdk_mem_map_free(&map);
        CU_ASSERT(map == NULL);
        //free(ptr);
        rte_free(ptr);
        return rc;
}
static int
lcore_hello(__attribute__((unused)) void *arg)
{
        test_mem_map_registration();
	return 0;
}
int
main(int argc, char **argv)
{
	int ret;
	unsigned lcore_id;

	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_panic("Cannot init EAL\n");
        spdk_env_dpdk_post_init(false);
#if 0
        ret = mem_map_init(legacy_mem);
	if (ret < 0) {
		SPDK_ERRLOG("Failed to allocate mem_map\n");
		return ret;
	}

	ret = vtophys_init();
	if (ret < 0) {
		SPDK_ERRLOG("Failed to initialize vtophys\n");
		return ret;
	}
#endif
	/* call lcore_hello() on every slave lcore */
	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		rte_eal_remote_launch(lcore_hello, NULL, lcore_id);
	}

	/* call it on master lcore too */
	lcore_hello(NULL);
	rte_eal_mp_wait_lcore();
        spdk_env_dpdk_post_fini();
	return 0;
}
