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
#include <rte_malloc.h>
#include "malloc_heap.h"
#include "malloc_elem.h"
#include "eal_internal_cfg.h"
#include "eal_private.h"
#include "eal_memcfg.h"
#include "rte_mempool.h"
#include "config.h"
#define TEST_MEM_1 0
static int print_malloc_add_seg( struct rte_memseg_list *msl, struct rte_memseg *ms);
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
#if TEST_MEM_1
		page_sz = RTE_PGSIZE_64K;
#else
		page_sz = RTE_PGSIZE_4K;
#endif
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
#if TEST_MEM_1
                else
                {
                   printf("****************** mmap addr %p and size %ldk \n", addr, internal_config.memory/1024);
                }
#endif
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
//#if 0
#if TEST_MEM_1
                        printf("****************** curg %d, start addr  %p,  end addr %p \n", cur_seg, addr , RTE_PTR_ADD(addr, (size_t)page_sz));
                       print_malloc_add_seg(msl, ms);
                       
                        *(int*) (addr + 0xff80) = 99;
                        printf("****************** write addr  %p,  end value %d \n", addr + 0xff80, *(int*) (addr + 0xff80));
#endif
			ms->addr = addr;
			ms->hugepage_sz = page_sz;
			ms->socket_id = 0;
			ms->len = page_sz;

			rte_fbarray_set_used(arr, cur_seg);

			addr = RTE_PTR_ADD(addr, (size_t)page_sz);
#if 0
//#if TEST_MEM_1
                       print_malloc_add_seg(msl, ms);
#endif
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
//#define MEMSIZE_IF_NO_HUGE_PAGE (64ULL * 1024ULL)
#define MEMSIZE_IF_NO_HUGE_PAGE (64ULL * 1024ULL * 1024ULL)
static const char *_MSG_POOL = "MSG_POOL";
struct data {
	uint32_t value;
};
struct private_data {
	uint32_t value;
};
struct internal_config internal_config;
int test_mempool()
{
      struct rte_mempool *message_pool;
      void *p  = NULL;
      const unsigned pool_size = 64;
      const unsigned pool_cache = 32;
      const unsigned priv_data_sz = sizeof(struct private_data);
      message_pool = rte_mempool_create(_MSG_POOL, pool_size,
                                sizeof(struct data), pool_cache, priv_data_sz,
                                NULL, NULL, NULL, NULL,
                                0, 0);
                                //rte_socket_id(), 0);
      if(!message_pool) 
      {
          RTE_LOG(ERR, MEMPOOL, "memsegs pool failed \n");
          return 0;
      }
      rte_mempool_get(message_pool, &p);
      RTE_LOG(ERR, MEMPOOL, "addr is %p\n",p);
      rte_mempool_put(message_pool, p);
     return 0;
}
static int print_malloc_add_seg( struct rte_memseg_list *msl, struct rte_memseg *ms)
{
        struct rte_mem_config *mcfg = test_rte_eal_get_configuration()->mem_config;
        struct rte_memseg_list *found_msl;
        struct malloc_heap *heap;
        int msl_idx, heap_idx;

        if (msl->external)
                return 0;

        heap_idx = test_malloc_socket_to_heap_id(msl->socket_id);
        if (heap_idx < 0) {
                RTE_LOG(ERR, EAL, "Memseg list has invalid socket id\n");
                return -1;
        }
        heap = &mcfg->malloc_heaps[heap_idx];

        /* msl is const, so find it */
        msl_idx = msl - mcfg->memsegs;

        if (msl_idx < 0 || msl_idx >= RTE_MAX_MEMSEG_LISTS)
                return -1;

        found_msl = &mcfg->memsegs[msl_idx];

        printf("add  seg memory( msl_idx %d) to heap %d  \n ", msl_idx,  heap_idx);
        //malloc_heap_add_memory(heap, found_msl, ms->addr, len);

        //heap->total_size += len;

        return 0;
}

int
main(int argc, char **argv)
{
        void * addr;
        struct rte_mem_config *mcfg;
        struct rte_memseg_list *msl;
        struct malloc_heap *heap;
        int msl_idx, heap_idx;
        int begin = 128;
        struct malloc_elem *elem;
        heap = &mcfg->malloc_heaps[heap_idx];
        const char src[64] = "http://www.runoob.com";
        int loop = 2;
        test_rte_eal_get_configuration()->process_type =  RTE_PROC_PRIMARY;
        internal_config.memory = MEMSIZE_IF_NO_HUGE_PAGE;
        internal_config.no_hugetlbfs = 1;
        internal_config.legacy_mem = 1;
        printf("rte_socket_count %d \n", rte_socket_count());
        printf("rte_eal_has_hugepages %d \n", rte_eal_has_hugepages());
        while(begin < 1024*1024)
        {
            printf("index of %d :  %d \n ", begin, malloc_elem_free_list_index(begin));
            begin *=2;
        }
        //printf("\n");
        //printf("index of 128 %d , index of 1024 %d \n", malloc_elem_free_list_index(128), malloc_elem_free_list_index(1024));
        //test_rte_eal_hugepage_init();        
        test_eal_legacy_hugepage_init();
        printf("eal init complete \n");
         test_rte_eal_memzone_init();
        test_rte_eal_malloc_heap_init();
        printf("heap init complete \n");
        test_rte_eal_tailqs_init();
        mcfg = test_rte_eal_get_configuration()->mem_config; 
        msl = &mcfg->memsegs[0];
        while(--loop >= 0)       
        {
            addr = test_rte_malloc_socket("test", 128,8, 0);
            printf("addr is %p \n", addr);
#if TEST_MEM_1
            if((unsigned long)addr >= (unsigned long) msl->base_va &&  (unsigned long)addr <= (unsigned long)RTE_PTR_ADD(msl->base_va, (size_t)RTE_PGSIZE_64K) )
            {
                 printf("******************  addr %p is legal \n", addr);
            }
            else
            {
                 printf("******************  addr %p is illegal \n", addr);
                 continue;
            }
            *(int *) addr = 99;
            printf("****************** mmap addr %p and size %ldk \n", addr, internal_config.memory/1024);
#endif 
            memcpy(addr, src, strlen(src)+1);
            printf("dest = %s\n", addr); 
            //test_rte_free(addr);
            elem = malloc_elem_from_data(addr);
            //elem = (struct malloc_elem *) addr;
            printf("######################malloc elem dump ########################## \n");
            malloc_elem_dump(elem, stdout);
        }
        heap_idx = test_malloc_socket_to_heap_id(msl->socket_id);
        if (heap_idx < 0) {
                RTE_LOG(ERR, EAL, "Memseg list has invalid socket id\n");
                return -1;
        }
        heap = &mcfg->malloc_heaps[heap_idx];
        printf("######################malloc heap dump ########################## \n");
        test_malloc_heap_dump(heap,stdout);
        test_mempool(); 
        test_eal_legacy_hugepage_destroy();
	return 0;
}
