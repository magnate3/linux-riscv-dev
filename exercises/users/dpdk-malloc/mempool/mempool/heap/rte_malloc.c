#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/queue.h>

#include <rte_errno.h>
#include <rte_memcpy.h>
#include <rte_memory.h>
#include <rte_eal.h>
#include <rte_eal_memconfig.h>
#include <rte_branch_prediction.h>
#include <rte_debug.h>
#include <rte_launch.h>
#include <rte_per_lcore.h>
#include <rte_lcore.h>
#include <rte_common.h>
#include <rte_spinlock.h>

#include <rte_malloc.h>
#include "malloc_elem.h"
#include "malloc_heap.h"
#include "eal_memalloc.h"
#include "eal_memcfg.h"
#include "eal_private.h"


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
/*
 *  * Return the IO address of a virtual address obtained through rte_malloc
 *   */
rte_iova_t
test_rte_malloc_virt2iova(const void *addr)
{
        const struct rte_memseg *ms;
        struct malloc_elem *elem = malloc_elem_from_data(addr);

        if (elem == NULL)
                return RTE_BAD_IOVA;

        if (!elem->msl->external && rte_eal_iova_mode() == RTE_IOVA_VA)
                return (uintptr_t) addr;

        ms = rte_mem_virt2memseg(addr, elem->msl);
        if (ms == NULL)
                return RTE_BAD_IOVA;

        if (ms->iova == RTE_BAD_IOVA)
                return RTE_BAD_IOVA;

        return ms->iova + RTE_PTR_DIFF(addr, ms->addr);
}
/*
 *  * Allocate zero'd memory on specified heap.
 *   */
void *
test_rte_zmalloc_socket(const char *type, size_t size, unsigned align, int socket)
{
        void *ptr = test_rte_malloc_socket(type, size, align, socket);

#ifdef RTE_MALLOC_DEBUG
        /*
 *          * If DEBUG is enabled, then freed memory is marked with poison
 *                   * value and set to zero on allocation.
 *                            * If DEBUG is not enabled then  memory is already zeroed.
 *                                     */
        if (ptr != NULL)
                memset(ptr, 0, size);
#endif
        return ptr;
}
/*
 *  * Allocate zero'd memory on default heap.
 *   */
void *
test_rte_zmalloc(const char *type, size_t size, unsigned align)
{
        return test_rte_zmalloc_socket(type, size, align, SOCKET_ID_ANY);
}


#endif
