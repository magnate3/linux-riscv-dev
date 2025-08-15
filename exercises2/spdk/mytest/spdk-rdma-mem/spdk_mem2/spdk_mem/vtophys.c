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
#define BAD_PHYS_ADDR 0
#define PFN_MASK_SIZE 8
extern uint64_t vtophys_get_paddr_memseg(uint64_t vaddr);
uint64_t mem_virt2phy(const void *virtaddr)
{
	int fd, retval;
	uint64_t page, physaddr;
	unsigned long long virt_pfn;	// virtual page frame number
	int page_size;
	off_t offset;
        int swapped ;
	int present ;
	/* standard page size */
	page_size = getpagesize();

	fd = open("/proc/self/pagemap", O_RDONLY);
	if (fd < 0) {
		fprintf(stderr, "%s(): cannot open /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	}

	virt_pfn = (unsigned long)virtaddr / page_size;
	//printf("Virtual page frame number is %llu\n", virt_pfn);

	offset = sizeof(uint64_t) * virt_pfn;
	if (lseek(fd, offset, SEEK_SET) == (off_t) - 1) {
		fprintf(stderr, "%s(): seek error in /proc/self/pagemap: %s\n", __func__, strerror(errno));
		close(fd);
		return BAD_PHYS_ADDR;
	}

	retval = read(fd, &page, PFN_MASK_SIZE);
	close(fd);
	if (retval < 0) {
		fprintf(stderr, "%s(): cannot read /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	} else if (retval != PFN_MASK_SIZE) {
		fprintf(stderr, "%s(): read %d bytes from /proc/self/pagemap but expected %d\n", 
		        __func__, retval, PFN_MASK_SIZE);
		return BAD_PHYS_ADDR;
	}

	/*
	 * the pfn (page frame number) are bits 0-54 (see
	 * pagemap.txt in linux Documentation)
	 */
	if ((page & 0x7fffffffffffffULL) == 0) {
		fprintf(stderr, "Zero page frame number\n");
		return BAD_PHYS_ADDR;
	}

	swapped = (page >> 62) & 1;
	present = (page >> 63) & 1;
	physaddr = ((page & 0x7fffffffffffffULL) * page_size) + ((unsigned long)virtaddr % page_size);
	printf("virt addr 0x%lx, phyaddr 0x%lx , swap %d , present %d \n", (unsigned long)virtaddr, physaddr, swapped, present);

	return physaddr;
}
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
#if 1
        SPDK_STATIC_ASSERT(SPDK_VTOPHYS_ERROR == UINT64_C(-1), "SPDK_VTOPHYS_ERROR should be all 1s");
        if (paddr_2mb == SPDK_VTOPHYS_ERROR) {
                return SPDK_VTOPHYS_ERROR;
        } else {
                printf("%s virt addr 0x%lx and call vtophys_get_paddr_memseg phy addr 0x%lx, vaddr & MASK_2MB 0x%llx \n",__func__, vaddr,  paddr_2mb, vaddr & MASK_2MB);
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
    SPDK_CU_ASSERT_FATAL(((uintptr_t)vaddr & MASK_2MB) == 0);
    SPDK_CU_ASSERT_FATAL((size & MASK_2MB) == 0);
    printf("*************** %s: spdk_mem_map addr %p, size 0x%lx, multiple of 2MB: %lu, %s event \n", __func__,map,size,(size >> SHIFT_2MB), SPDK_MEM_MAP_NOTIFY_REGISTER == action ? "register": "unregister");
    uint32_t i, end;
    uint64_t start;
    uint32_t index = 0;
    i = (uintptr_t)vaddr >> SHIFT_2MB;
    end = i + (size >> SHIFT_2MB);
    for (; i < end; i++) {
        switch (action) {
           case SPDK_MEM_MAP_NOTIFY_REGISTER:
                   /* This page should not already be registered */
                   g_p = (char*)malloc(1);
                   
                   start = (uint64_t)vaddr + index*VALUE_2MB;
                   ++index;
                   spdk_mem_map_set_translation(map, start , VALUE_2MB,  vtophys_get_paddr_memseg((uint64_t)start));
                   //spdk_mem_map_set_translation(map, (uint64_t)vaddr, size,  vtophys_get_paddr_memseg((uint64_t)vaddr));
                   //spdk_mem_map_translate(map, (uint64_t)vaddr, &size);
                   //spdk_mem_map_clear_translation(map, (uint64_t)vaddr, size);
                   break;
           case SPDK_MEM_MAP_NOTIFY_UNREGISTER:
                   //SPDK_CU_ASSERT_FATAL(spdk_bit_array_get(g_page_array, i) == true);
                   //spdk_bit_array_clear(g_page_array, i);
                   spdk_mem_map_clear_translation(map, (uint64_t)vaddr, size);
                   break;
           default:
                   SPDK_UNREACHABLE();
           }
       }
        return 0;
}
static int
test_check_regions_contiguous(uint64_t paddr1, uint64_t paddr2)
{
    return (paddr2 - paddr1 == VALUE_2MB);
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
        char* ptr2;
        char* ptr3;
        //uint64_t default_translation = 0xDEADBEEF0BADF00D;
        //uint64_t size =0x100000;
        uint64_t size =VALUE_2MB;
        uint64_t size2 = 8;
        uint64_t paddr;
        map = spdk_mem_map_alloc(0, &test_mem_map_ops, NULL);
        //map = spdk_mem_map_alloc(SPDK_VTOPHYS_ERROR, &test_mem_map_ops, NULL);
        //  MASK_2MB=1fffff
        printf("size %lx, MASK_2MB %llx , MASK_2MB & size %llx \n",size, MASK_2MB, MASK_2MB & size);
        printf("%s: spdk_mem_map addr %p \n", __func__,map);
        SPDK_CU_ASSERT_FATAL(map != NULL);
        //ptr = (char *)malloc(size);
        ptr = (char *)rte_malloc("test",size,size);
        printf("malloc virt addr %p and virt addr & MASK_2MB %llx \n", ptr, MASK_2MB&(uint64_t)ptr);
#if 0
        rc = spdk_mem_register(ptr,  size);
        if (rc) {
             printf("spdk mem register fail \n");
             return 0;
        }
#endif
        mem_virt2phy(ptr); 
        printf("virt addr %p and call rte_mem_virt2phy phy addr 0x%lx \n", ptr,  rte_mem_virt2phy(ptr));
        printf("virt addr %p and call spdk_vtophys phy addr 0x%lx \n", ptr,  spdk_vtophys(ptr,&size));
        printf("virt addr %p and call vtophys_get_paddr_memseg phy addr 0x%lx \n", ptr,  vtophys_get_paddr_memseg((uint64_t)ptr));
	paddr = test_spdk_vtophys(map,ptr, &size);
        printf("virt addr %p and test_spdk_vtophys phy addr 0x%lx \n", ptr, paddr);
        //*ptr= 'A';
	paddr = test_spdk_vtophys(map,ptr, &size);
        printf("after pagefault, virt addr %p and  test_spdk_vtophys phy addr 0x%lx \n", ptr, paddr);
        ///////////////////////////////////////////////////////////////////////////
        printf("-------------- test ptr2 --------------------- \n");
        ptr2 = (char *)rte_malloc("test2",size2,VALUE_2MB);
        printf("virt addr2 %p and call rte_mem_virt2phy phy addr 0x%lx \n", ptr2,  rte_mem_virt2phy(ptr2));
	paddr = test_spdk_vtophys(map,ptr2, &size2);
        printf("virt addr2 %p and test_spdk_vtophys  phy addr 0x%lx \n", ptr2, paddr);
        printf("-------------- test ptr3 --------------------- \n");
        ptr3 = (char *)rte_malloc("test3",size2,size2);
        printf("virt addr3 %p and call rte_mem_virt2phy phy addr 0x%lx \n", ptr3,  rte_mem_virt2phy(ptr3));
	paddr = test_spdk_vtophys(map,ptr2, &size2);
        printf("virt addr3 %p and test_spdk_vtophys  phy addr 0x%lx \n", ptr3, paddr);
        spdk_mem_map_free(&map);
        CU_ASSERT(map == NULL);
        //free(ptr);
        rte_free(ptr);
        rte_free(ptr2);
        rte_free(ptr3);
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
