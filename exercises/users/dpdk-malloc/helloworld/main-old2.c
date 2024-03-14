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
////////////////
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#define HAVE_USE 0
#include <rte_memzone.h>
#include <rte_pause.h>
#include <rte_spinlock.h>
#include <rte_rwlock.h>
#include <rte_tailq.h>
#include "malloc_heap.h"
/* Address of global and public configuration */
static struct rte_config rte_config;

/* Return a pointer to the configuration structure */
struct rte_config *
rte_eal_get_configuration_test(void)
{
        return &rte_config;
}
struct rte_config {
        uint32_t master_lcore;       /**< Id of the master lcore */
        uint32_t lcore_count;        /**< Number of available logical cores. */
        uint32_t numa_node_count;    /**< Number of detected NUMA nodes. */
        uint32_t numa_nodes[RTE_MAX_NUMA_NODES]; /**< List of detected NUMA nodes. */
        uint32_t service_lcore_count;/**< Number of available service cores. */
        enum rte_lcore_role_t lcore_role[RTE_MAX_LCORE]; /**< State of cores. */

        /** Primary or secondary configuration */
        enum rte_proc_type_t process_type;

        /** PA or VA mapping mode */
        enum rte_iova_mode iova_mode;

        /**
 *          * Pointer to memory configuration, which may be shared across multiple
 *                   * DPDK instances
 *                            */
        struct rte_mem_config *mem_config;
} __attribute__((__packed__));
/**
 *  * Memory configuration shared across multiple processes.
 *   */
struct rte_mem_config {
        volatile uint32_t magic;   /**< Magic number - sanity check. */
        uint32_t version;
        /**< Prevent secondary processes using different DPDK versions. */

        /* memory topology */
        uint32_t nchannel;    /**< Number of channels (0 if unknown). */
        uint32_t nrank;       /**< Number of ranks (0 if unknown). */

        /**
 *          * current lock nest order
 *                   *  - qlock->mlock (ring/hash/lpm)
 *                            *  - mplock->qlock->mlock (mempool)
 *                                     * Notice:
 *                                              *  *ALWAYS* obtain qlock first if having to obtain both qlock and mlock
 *                                                       */
        rte_rwlock_t mlock;   /**< used by memzones for thread safety. */
        rte_rwlock_t qlock;   /**< used by tailqs for thread safety. */
        rte_rwlock_t mplock;  /**< used by mempool library for thread safety. */
        rte_spinlock_t tlock; /**< used by timer library for thread safety. */

        rte_rwlock_t memory_hotplug_lock;
        /**< Indicates whether memory hotplug request is in progress. */

        /* memory segments and zones */
        struct rte_fbarray memzones; /**< Memzone descriptors. */

        struct rte_memseg_list memsegs[RTE_MAX_MEMSEG_LISTS];
        /**< List of dynamic arrays holding memsegs */

        struct rte_tailq_head tailq_head[RTE_MAX_TAILQ];
        /**< Tailqs for objects */

        struct malloc_heap malloc_heaps[RTE_MAX_HEAPS];
        /**< DPDK malloc heaps */

        int next_socket_id; /**< Next socket ID for external malloc heap */

        /* rte_mem_config has to be mapped at the exact same address in all
 *          * processes, so we need to store it.
 *                   */
        uint64_t mem_cfg_addr; /**< Address of this structure in memory. */

        /* Primary and secondary processes cannot run with different legacy or
 *          * single file segments options, so to avoid having to specify these
 *                   * options to all processes, store them in shared config and update the
 *                            * internal config at init time.
 *                                     */
        uint32_t legacy_mem; /**< stored legacy mem parameter. */
        uint32_t single_file_segments;
        /**< stored single file segments parameter. */

        uint64_t tsc_hz;
        /**< TSC rate */

        uint8_t dma_maskbits; /**< Keeps the more restricted dma mask. */
};
#if 0
/**< Prevent this segment from being freed back to the OS. */
struct rte_memseg {
        RTE_STD_C11
        union {
                phys_addr_t phys_addr;  /**< deprecated - Start physical address. */
                rte_iova_t iova;        /**< Start IO address. */
        };
        RTE_STD_C11
        union {
                void *addr;         /**< Start virtual address. */
                uint64_t addr_64;   /**< Makes sure addr is always 64 bits */
        };
        size_t len;               /**< Length of the segment. */
        uint64_t hugepage_sz;       /**< The pagesize of underlying memory */
        int32_t socket_id;          /**< NUMA socket ID. */
        uint32_t nchannel;          /**< Number of channels. */
        uint32_t nrank;             /**< Number of ranks. */
        uint32_t flags;             /**< Memseg-specific flags */
} __rte_packed;

/**
 *  * memseg list is a special case as we need to store a bunch of other data
 *   * together with the array itself.
 *    */
struct rte_memseg_list {
        RTE_STD_C11
        union {
                void *base_va;
                /**< Base virtual address for this memseg list. */
                uint64_t addr_64;
                /**< Makes sure addr is always 64-bits */
        };
        uint64_t page_sz; /**< Page size for all memsegs in this list. */
        int socket_id; /**< Socket ID for all memsegs in this list. */
        volatile uint32_t version; /**< version number for multiprocess sync. */
        size_t len; /**< Length of memory area covered by this memseg list. */
        unsigned int external; /**< 1 if this list points to external memory */
        unsigned int heap; /**< 1 if this list points to a heap */
        struct rte_fbarray memseg_arr;
};
#endif
struct internal_config internal_config;
int
rte_eal_hugepage_init_test(void)
{
	struct rte_mem_config *mcfg;
	uint64_t total_mem = 0;
	void *addr;
	unsigned int i, j, seg_idx = 0;

	/* get pointer to global configuration */
	mcfg = rte_eal_get_configuration_test()->mem_config;

	/* for debug purposes, hugetlbfs can be disabled */
	//if (internal_config.no_hugetlbfs) {
		struct rte_memseg_list *msl;
		struct rte_fbarray *arr;
		struct rte_memseg *ms;
		uint64_t page_sz;
		int n_segs, cur_seg;

		/* create a memseg list */
		msl = &mcfg->memsegs[0];

		page_sz = RTE_PGSIZE_4K;
		n_segs = internal_config.memory / page_sz;

		if (rte_fbarray_init(&msl->memseg_arr, "nohugemem", n_segs,
				sizeof(struct rte_memseg))) {
			RTE_LOG(ERR, EAL, "Cannot allocate memseg list\n");
			return -1;
		}

		addr = mmap(NULL, internal_config.memory,
				PROT_READ | PROT_WRITE,
				MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (addr == MAP_FAILED) {
			RTE_LOG(ERR, EAL, "%s: mmap() failed: %s\n", __func__,
					strerror(errno));
			return -1;
		}
		msl->base_va = addr;
		msl->page_sz = page_sz;
		msl->len = internal_config.memory;
		msl->socket_id = 0;
		msl->heap = 1;

		/* populate memsegs. each memseg is 1 page long */
		for (cur_seg = 0; cur_seg < n_segs; cur_seg++) {
			arr = &msl->memseg_arr;

			ms = rte_fbarray_get(arr, cur_seg);
			if (rte_eal_iova_mode() == RTE_IOVA_VA)
				ms->iova = (uintptr_t)addr;
			else
				ms->iova = RTE_BAD_IOVA;
			ms->addr = addr;
			ms->hugepage_sz = page_sz;
			ms->len = page_sz;
			ms->socket_id = 0;

			rte_fbarray_set_used(arr, cur_seg);

			addr = RTE_PTR_ADD(addr, page_sz);
		}
		return 0;
	//}

   
}
static int
lcore_hello(__attribute__((unused)) void *arg)
{
	unsigned lcore_id;
	lcore_id = rte_lcore_id();
	printf("hello from core %u\n", lcore_id);
	return 0;
}
#if 0
int
rte_eal_malloc_heap_init_test(void)
{
        struct rte_mem_config *mcfg = rte_eal_get_configuration()->mem_config;
        unsigned int i;

        if (internal_config.match_allocations) {
                RTE_LOG(DEBUG, EAL, "Hugepages will be freed exactly as allocated.\n");
        }

        if (rte_eal_process_type() == RTE_PROC_PRIMARY) {
                /* assign min socket ID to external heaps */
                mcfg->next_socket_id = EXTERNAL_HEAP_MIN_SOCKET_ID;

                /* assign names to default DPDK heaps */
                for (i = 0; i < rte_socket_count(); i++) {
                        struct malloc_heap *heap = &mcfg->malloc_heaps[i];
                        char heap_name[RTE_HEAP_NAME_MAX_LEN];
                        int socket_id = rte_socket_id_by_idx(i);

                        snprintf(heap_name, sizeof(heap_name),
                                        "socket_%i", socket_id);
                        strlcpy(heap->name, heap_name, RTE_HEAP_NAME_MAX_LEN);
                        heap->socket_id = socket_id;
                }
        }


        if (register_mp_requests()) {
                RTE_LOG(ERR, EAL, "Couldn't register malloc multiprocess actions\n");
                rte_mcfg_mem_read_unlock();
                return -1;
        }

        /* unlock mem hotplug here. it's safe for primary as no requests can
 *          * even come before primary itself is fully initialized, and secondaries
 *                   * do not need to initialize the heap.
 *                            */
        rte_mcfg_mem_read_unlock();

        /* secondary process does not need to initialize anything */
        if (rte_eal_process_type() != RTE_PROC_PRIMARY)
                return 0;

        /* add all IOVA-contiguous areas to the heap */
        return rte_memseg_contig_walk(malloc_add_seg, NULL);
}
#endif
int
main(int argc, char **argv)
{
        
	return 0;
}
