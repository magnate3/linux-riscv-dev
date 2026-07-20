#include "eal_private.h"
#include "eal_memcfg.h"
/* Address of global and public configuration */
static struct rte_mem_config early_mem_config;
/* Address of global and public configuration */
static struct rte_config rte_config = {
                .mem_config = &early_mem_config,
};

/* Return a pointer to the configuration structure */
struct rte_config *
test_rte_eal_get_configuration(void)
{
        return &rte_config;
}
#if 0
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
#endif
