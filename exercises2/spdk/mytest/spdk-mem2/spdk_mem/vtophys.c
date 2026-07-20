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
#if 1
static int
test_mem_map_notify(void *cb_ctx, struct spdk_mem_map *map,
                    enum spdk_mem_map_notify_action action,
                    void *vaddr, size_t len)
{
        uint32_t i, end;

        SPDK_CU_ASSERT_FATAL(((uintptr_t)vaddr & MASK_2MB) == 0);
        SPDK_CU_ASSERT_FATAL((len & MASK_2MB) == 0);

        /*
 *          * This is a test requirement - the bit array we use to verify
 *                   * pages are valid is only so large.
 *                            */
        //SPDK_CU_ASSERT_FATAL((uintptr_t)vaddr < (VALUE_2MB * PAGE_ARRAY_SIZE));

        i = (uintptr_t)vaddr >> SHIFT_2MB;
        end = i + (len >> SHIFT_2MB);
        for (; i < end; i++) {
                switch (action) {
                case SPDK_MEM_MAP_NOTIFY_REGISTER:
                        /* This page should not already be registered */
                        //SPDK_CU_ASSERT_FATAL(spdk_bit_array_get(g_page_array, i) == false);
                        //SPDK_CU_ASSERT_FATAL(spdk_bit_array_set(g_page_array, i) == 0);
                        break;
                case SPDK_MEM_MAP_NOTIFY_UNREGISTER:
                        //SPDK_CU_ASSERT_FATAL(spdk_bit_array_get(g_page_array, i) == true);
                        //spdk_bit_array_clear(g_page_array, i);
                        break;
                default:
                        SPDK_UNREACHABLE();
                }
        }

        return 0;
}
#else
test_mem_map_notify(void *cb_ctx, struct spdk_mem_map *map,
                    enum spdk_mem_map_notify_action action,
                    void *vaddr, size_t size)
{
                printf("%s: spdk_mem_map addr %p \n", __func__,map);
                switch (action) {
                case SPDK_MEM_MAP_NOTIFY_REGISTER:
                        /* This page should not already be registered */
                        //SPDK_CU_ASSERT_FATAL(spdk_bit_array_get(g_page_array, i) == false);
                        //SPDK_CU_ASSERT_FATAL(spdk_bit_array_set(g_page_array, i) == 0);
                        spdk_mem_map_set_translation(map, (uint64_t)vaddr, size, 0 == (uint64_t)vaddr);
                        break;
                case SPDK_MEM_MAP_NOTIFY_UNREGISTER:
                        //SPDK_CU_ASSERT_FATAL(spdk_bit_array_get(g_page_array, i) == true);
                        //spdk_bit_array_clear(g_page_array, i);
                        spdk_mem_map_clear_translation(map, (uint64_t)vaddr, size);
                        break;
                default:
                        SPDK_UNREACHABLE();
                }
        return 0;
}
#endif
static int
test_check_regions_contiguous(uint64_t addr1, uint64_t addr2)
{
        return addr1 == addr2;
}
const struct spdk_mem_map_ops test_mem_map_ops = {
        .notify_cb = test_mem_map_notify,
        .are_contiguous = test_check_regions_contiguous
};
#if 1
static int 
test_mem_map_registration(void)
{
        struct spdk_mem_map *map;
        int rc = 0;
        char* ptr;
        uint64_t default_translation = 0xDEADBEEF0BADF00D;
        uint64_t size =0x200000;
        uint64_t paddr;
        map = spdk_mem_map_alloc(SPDK_VTOPHYS_ERROR, &test_mem_map_ops, NULL);
        //map = spdk_mem_map_alloc(default_translation, &test_mem_map_ops, NULL);
        //  MASK_2MB=1fffff
        printf("MASK_2MB %llx , MASK_2MB & size %llx \n", MASK_2MB, MASK_2MB & size);
        printf("%s: spdk_mem_map addr %p \n", __func__,map);
        SPDK_CU_ASSERT_FATAL(map != NULL);
        //ptr = (char *)malloc(size);
        ptr = (char *)rte_malloc("test",0x200000,0x200000);
        printf("virt addr %p and virt addr & MASK_2MB %llx \n", ptr, MASK_2MB&(uint64_t)ptr);
        rc = spdk_mem_register(ptr,  size);
        if (rc) {
             printf("spdk mem register fail \n");
             return 0;
        }
	paddr = spdk_vtophys(ptr, &size);
        printf("virt addr %p and phy addr %lx \n", ptr, paddr);
        *ptr= 'A';
	paddr = spdk_vtophys(ptr, &size);
        printf("after pagefault, virt addr %p and phy addr %lx \n", ptr, paddr);
        spdk_mem_map_free(&map);
        CU_ASSERT(map == NULL);
        rte_free(ptr);
        return rc;
}
#else
static int 
test_mem_map_registration(void)
{
        struct spdk_mem_map *map;
        int rc = 0;
        char* ptr;
        uint64_t default_translation = 0xDEADBEEF0BADF00D;
        uint64_t size =0x200000;
        uint64_t paddr;
        map = spdk_mem_map_alloc(SPDK_VTOPHYS_ERROR, &test_mem_map_ops, NULL);
        //map = spdk_mem_map_alloc(default_translation, &test_mem_map_ops, NULL);
        //  MASK_2MB=1fffff
        printf("MASK_2MB %llx , MASK_2MB & size %llx \n", MASK_2MB, MASK_2MB & size);
        printf("%s: spdk_mem_map addr %p \n", __func__,map);
        SPDK_CU_ASSERT_FATAL(map != NULL);
        //ptr = (char *)malloc(size);
        ptr = (char *)rte_malloc("test",0x200000,0x200000);
        printf("virt addr %p and virt addr & MASK_2MB %llx \n", ptr, MASK_2MB&(uint64_t)ptr);
	paddr = spdk_vtophys(ptr, &size);
        printf("virt addr %p and phy addr %lx \n", ptr, paddr);
        *ptr= 'A';
	paddr = spdk_vtophys(ptr, &size);
        printf("after pagefault, virt addr %p and phy addr %lx \n", ptr, paddr);
        spdk_mem_map_free(&map);
        CU_ASSERT(map == NULL);
        rte_free(ptr);
        return rc;
}
#endif
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
