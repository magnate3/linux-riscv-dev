/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/queue.h>

//#include <rte_memory.h>
//#include <rte_launch.h>
//#include <rte_eal.h>
//#include <rte_per_lcore.h>
//#include <rte_lcore.h>
//#include <rte_debug.h>
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
#include "malloc_elem.h"
#include "eal_internal_cfg.h"
#include "eal_private.h"
#include "eal_memcfg.h"
#include "config.h"
static int test_eal_legacy_hugepage_init(void)
{
	struct rte_mem_config *mcfg;
	struct hugepage_file *hugepage = NULL, *tmp_hp = NULL;
#if HAVE_USE
	struct hugepage_info used_hp[MAX_HUGEPAGE_SIZES];
#endif
	struct rte_fbarray *arr;
	struct rte_memseg *ms;

	uint64_t memory[RTE_MAX_NUMA_NODES];

	unsigned hp_offset;
	int i, j;
	int nr_hugefiles, nr_hugepages = 0;
	void *addr;

#if HAVE_USE
	memset(used_hp, 0, sizeof(used_hp));
#endif
	/* get pointer to global configuration */
	mcfg = test_rte_eal_get_configuration()->mem_config;

	/* hugetlbfs can be disabled */
	//if (internal_config.no_hugetlbfs) {
		struct rte_memseg_list *msl;
		int n_segs, cur_seg, fd, flags;
#ifdef MEMFD_SUPPORTED
		int memfd;
#endif
		uint64_t page_sz;

		/* nohuge mode is legacy mode */
		internal_config.legacy_mem = 1;

		/* nohuge mode is single-file segments mode */
		internal_config.single_file_segments = 1;

		/* create a memseg list */
		msl = &mcfg->memsegs[0];

		page_sz = RTE_PGSIZE_4K;
		n_segs = internal_config.memory / page_sz;

		if (rte_fbarray_init(&msl->memseg_arr, "nohugemem", n_segs,
					sizeof(struct rte_memseg))) {
			RTE_LOG(ERR, EAL, "Cannot allocate memseg list\n");
			return -1;
		}

		/* set up parameters for anonymous mmap */
		fd = -1;
		flags = MAP_PRIVATE | MAP_ANONYMOUS;

#ifdef MEMFD_SUPPORTED
		/* create a memfd and store it in the segment fd table */
		memfd = memfd_create("nohuge", 0);
		if (memfd < 0) {
			RTE_LOG(DEBUG, EAL, "Cannot create memfd: %s\n",
					strerror(errno));
			RTE_LOG(DEBUG, EAL, "Falling back to anonymous map\n");
		} else {
			/* we got an fd - now resize it */
			if (ftruncate(memfd, internal_config.memory) < 0) {
				RTE_LOG(ERR, EAL, "Cannot resize memfd: %s\n",
						strerror(errno));
				RTE_LOG(ERR, EAL, "Falling back to anonymous map\n");
				close(memfd);
			} else {
				/* creating memfd-backed file was successful.
				 * we want changes to memfd to be visible to
				 * other processes (such as vhost backend), so
				 * map it as shared memory.
				 */
				RTE_LOG(DEBUG, EAL, "Using memfd for anonymous memory\n");
				fd = memfd;
				flags = MAP_SHARED;
			}
		}
#endif
		addr = mmap(NULL, internal_config.memory, PROT_READ | PROT_WRITE,
				flags, fd, 0);
		if (addr == MAP_FAILED) {
			RTE_LOG(ERR, EAL, "%s: mmap() failed: %s\n", __func__,
					strerror(errno));
			return -1;
		}
		msl->base_va = addr;
		msl->page_sz = page_sz;
		msl->socket_id = 0;
		msl->len = internal_config.memory;
		msl->heap = 1;

		/* we're in single-file segments mode, so only the segment list
		 * fd needs to be set up.
		 */
		if (fd != -1) {
			if (eal_memalloc_set_seg_list_fd(0, fd) < 0) {
				RTE_LOG(ERR, EAL, "Cannot set up segment list fd\n");
				/* not a serious error, proceed */
			}
		}

		/* populate memsegs. each memseg is one page long */
		for (cur_seg = 0; cur_seg < n_segs; cur_seg++) {
			arr = &msl->memseg_arr;

			ms = rte_fbarray_get(arr, cur_seg);
			if (rte_eal_iova_mode() == RTE_IOVA_VA)
				ms->iova = (uintptr_t)addr;
			else
				ms->iova = RTE_BAD_IOVA;
			ms->addr = addr;
			ms->hugepage_sz = page_sz;
			ms->socket_id = 0;
			ms->len = page_sz;

			rte_fbarray_set_used(arr, cur_seg);

			addr = RTE_PTR_ADD(addr, (size_t)page_sz);
		}
		if (mcfg->dma_maskbits &&
		    rte_mem_check_dma_mask_thread_unsafe(mcfg->dma_maskbits)) {
			RTE_LOG(ERR, EAL,
				"%s(): couldn't allocate memory due to IOVA exceeding limits of current DMA mask.\n",
				__func__);
			if (rte_eal_iova_mode() == RTE_IOVA_VA &&
			    rte_eal_using_phys_addrs())
				RTE_LOG(ERR, EAL,
					"%s(): Please try initializing EAL with --iova-mode=pa parameter.\n",
					__func__);
			goto fail;
		}
		return 0;
fail:
    munmap(addr,internal_config.memory);
    return -1;
}
static int test_eal_legacy_hugepage_destroy(void)
{
    struct rte_memseg_list *msl;
    struct rte_mem_config *mcfg;
    mcfg = test_rte_eal_get_configuration()->mem_config;
    msl = &mcfg->memsegs[0];
    munmap(msl->base_va,internal_config.memory);
    return -1;
}
#if 1
int
test_rte_malloc_heap_socket_is_external(int socket_id)
{
        struct rte_mem_config *mcfg = test_rte_eal_get_configuration()->mem_config;
        unsigned int idx;
        int ret = -1;

        if (socket_id == SOCKET_ID_ANY)
                return 0;

        rte_mcfg_mem_read_lock();
        for (idx = 0; idx < RTE_MAX_HEAPS; idx++) {
                struct malloc_heap *tmp = &mcfg->malloc_heaps[idx];

                if ((int)tmp->socket_id == socket_id) {
                        /* external memory always has large socket ID's */
                        ret = tmp->socket_id >= RTE_MAX_NUMA_NODES;
                        break;
                }
        }
        rte_mcfg_mem_read_unlock();

        return ret;
}
/* Free the memory space back to heap */
void test_rte_free(void *addr)
{
        if (addr == NULL) return;
        if (test_malloc_heap_free(malloc_elem_from_data(addr)) < 0)
                RTE_LOG(ERR, EAL, "Error: Invalid memory\n");
}

/*
 *  * Allocate memory on specified heap.
 *   */
void *
test_rte_malloc_socket(const char *type, size_t size, unsigned int align,
                int socket_arg)
{
        /* return NULL if size is 0 or alignment is not power-of-2 */
        if (size == 0 || (align && !rte_is_power_of_2(align)))
                return NULL;

        /* if there are no hugepages and if we are not allocating from an
 *          * external heap, use memory from any socket available. checking for
 *                   * socket being external may return -1 in case of invalid socket, but
 *                            * that's OK - if there are no hugepages, it doesn't matter.
 *                                     */
        if (test_rte_malloc_heap_socket_is_external(socket_arg) != 1 &&
                                !rte_eal_has_hugepages())
                socket_arg = SOCKET_ID_ANY;

        return test_malloc_heap_alloc(type, size, socket_arg, 0,
                        align == 0 ? 1 : align, 0, false);
}
#endif
#define MEMSIZE_IF_NO_HUGE_PAGE (64ULL * 1024ULL * 1024ULL)
struct internal_config internal_config;
int
main(int argc, char **argv)
{
        void * addr;
        const char src[50] = "http://www.runoob.com";
        int loop = 100;
        test_rte_eal_get_configuration()->process_type =  RTE_PROC_PRIMARY;
        internal_config.memory = MEMSIZE_IF_NO_HUGE_PAGE;
        internal_config.memory = MEMSIZE_IF_NO_HUGE_PAGE;
        internal_config.no_hugetlbfs = 1;
        internal_config.legacy_mem = 1;
        //test_rte_eal_hugepage_init();        
        test_eal_legacy_hugepage_init();
        printf("eal init complete \n");
        test_rte_eal_malloc_heap_init();
        printf("heap init complete \n");
        while(--loop >= 0)       
        {
            addr = test_rte_malloc_socket("test", 128,8, 0);
            printf("addr is %p \n", addr);
            memcpy(addr, src, strlen(src)+1);
            printf("dest = %s\n", addr); 
            //test_rte_free(addr);
        }
        test_eal_legacy_hugepage_destroy();
	return 0;
}
